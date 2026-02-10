#include <metal_stdlib>
#include "MPSUniforms.metal"
#include "RTRMetalEngine/Rendering/RayTracingShaderTypes.h"

#if __has_include(<metal_raytracing>)
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif
#include <metal_raytracing>
#define RTR_HAS_RAYTRACING 1
#else
#define RTR_HAS_RAYTRACING 0
#endif

using namespace metal;

namespace {

constant uint kMaxTextureSize = 2048u;
constant uint kLightSamplesPerPixel = 4u;

float srgbChannelToLinear(float value) {
    if (value <= 0.0f) {
        return 0.0f;
    }
    if (value >= 1.0f) {
        return value;
    }
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return pow((value + 0.055f) / 1.055f, 2.4f);
}

simd_float3 sampleSkyColor(float3 /*direction*/) {
    // Hardware Cornell scenes are enclosed; leaking in a bright blue sky causes
    // long-running accumulation to drift toward a tint. Keep a neutral, dim
    // ambient term so demos without a ceiling still have a fallback without
    // overpowering the interior lighting.
    return float3(0.05f, 0.05f, 0.05f);
}

inline float wrapCoordinate(float value) {
    value = value - floor(value);
    return (value < 0.0f) ? value + 1.0f : value;
}

inline RTRRayTracingMaterial makeFallbackMaterial() {
    RTRRayTracingMaterial material;
    material.albedo = float3(0.5f, 0.5f, 0.5f);
    material.roughness = 0.5f;
    material.emission = float3(0.0f);
    material.metallic = 0.0f;
    material.reflectivity = 0.0f;
    material.indexOfRefraction = 1.0f;
    material.textureIndex = RTR_INVALID_TEXTURE_INDEX;
    material.materialFlags = 0u;
    return material;
}

inline float3 computeNormal(uint i0,
                            uint i1,
                            uint i2,
                            float3 v0,
                            float3 v1,
                            float3 v2,
                            const device packed_float3* normals,
                            uint normalCount,
                            float3 bary) {
    if (normals && i0 < normalCount && i1 < normalCount && i2 < normalCount) {
        const float3 n0 = float3(normals[i0]);
        const float3 n1 = float3(normals[i1]);
        const float3 n2 = float3(normals[i2]);
        const float3 blended = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
        if (all(isfinite(blended))) {
            return blended;
        }
    }
    return normalize(cross(v1 - v0, v2 - v0));
}

inline RTRRayTracingMaterial loadMaterial(uint materialIndex,
                                          const device RTRRayTracingMaterial* materials,
                                          uint materialCount,
                                          uint fallbackMaterialIndex) {
    if (!materials || materialCount == 0) {
        return makeFallbackMaterial();
    }
    if (materialIndex == RTR_INVALID_MATERIAL_INDEX) {
        materialIndex = fallbackMaterialIndex;
    }
    if (materialIndex == RTR_INVALID_MATERIAL_INDEX) {
        materialIndex = 0u;
    }
    materialIndex = min(materialIndex, materialCount - 1u);
    return materials[materialIndex];
}

inline float4 readTextureTexel(const RTRRayTracingTextureResource info,
                               uint x,
                               uint y,
                               const device float* pixels) {
    const uint base = info.dataOffset + (y * info.rowPitch + x) * 4u;
    return float4(pixels[base + 0], pixels[base + 1], pixels[base + 2], pixels[base + 3]);
}

inline float4 sampleTexture(uint textureIndex,
                            constant RTRRayTracingTextureResource* infos,
                            uint textureCount,
                            const device float* pixels,
                            float2 uv) {
    if (textureIndex == RTR_INVALID_TEXTURE_INDEX || infos == nullptr || pixels == nullptr) {
        return float4(1.0f);
    }
    if (textureIndex >= textureCount) {
        return float4(1.0f);
    }

    const RTRRayTracingTextureResource info = infos[textureIndex];
    if (info.width == 0u || info.height == 0u || info.rowPitch == 0u) {
        return float4(1.0f);
    }

    const float width = static_cast<float>(min(info.width, kMaxTextureSize));
    const float height = static_cast<float>(min(info.height, kMaxTextureSize));
    const float2 wrapped = float2(wrapCoordinate(uv.x), wrapCoordinate(uv.y));
    const float sampleX = wrapped.x * (width - 1.0f);
    const float sampleY = (1.0f - wrapped.y) * (height - 1.0f);

    const uint x0 = static_cast<uint>(floor(sampleX));
    const uint y0 = static_cast<uint>(floor(sampleY));
    const uint x1 = min(x0 + 1u, info.width - 1u);
    const uint y1 = min(y0 + 1u, info.height - 1u);
    const float tx = fract(sampleX);
    const float ty = fract(sampleY);

    const float4 t00 = readTextureTexel(info, x0, y0, pixels);
    const float4 t10 = readTextureTexel(info, x1, y0, pixels);
    const float4 t01 = readTextureTexel(info, x0, y1, pixels);
    const float4 t11 = readTextureTexel(info, x1, y1, pixels);
    const float4 a = mix(t00, t10, tx);
    const float4 b = mix(t01, t11, tx);
    return mix(a, b, ty);
}

inline float3 sampleMaterialColor(const RTRRayTracingMaterial material,
                                  float2 uv,
                                  constant RTRRayTracingTextureResource* infos,
                                  uint textureCount,
                                  const device float* pixels) {
    if (material.textureIndex == RTR_INVALID_TEXTURE_INDEX) {
        return material.albedo;
    }
    const float4 sampled = sampleTexture(material.textureIndex, infos, textureCount, pixels, uv);
    return float3(sampled.rgb) * material.albedo;
}

inline float3 transformPosition(float4x4 matrix, float3 position) {
    const float4 result = matrix * float4(position, 1.0f);
    return result.xyz;
}

inline float3 transformNormal(float4x4 worldToObject, float3 normal) {
    const float4 transformed = transpose(worldToObject) * float4(normal, 0.0f);
    return normalize(transformed.xyz);
}

inline float geometrySchlickGGX(float nDotV, float k) {
    return nDotV / (nDotV * (1.0f - k) + k);
}

inline float distributionGGX(float nDotH, float alpha) {
    const float a2 = alpha * alpha;
    const float denom = (nDotH * nDotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(3.14159265f * denom * denom, 1.0e-4f);
}

inline float geometrySmith(float nDotV, float nDotL, float roughness) {
    const float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    return geometrySchlickGGX(nDotV, k) * geometrySchlickGGX(nDotL, k);
}

inline float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0f) - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

inline float3 debugInstanceColor(uint value) {
    uint hash = value * 1664525u + 1013904223u;
    const float r = static_cast<float>((hash >> 16) & 0xFFu) / 255.0f;
    hash = hash * 1664525u + 1013904223u;
    const float g = static_cast<float>((hash >> 8) & 0xFFu) / 255.0f;
    hash = hash * 1664525u + 1013904223u;
    const float b = static_cast<float>(hash & 0xFFu) / 255.0f;
    return clamp(float3(r, g, b), 0.05f, 1.0f);
}

inline RTRHardwareAreaLight defaultAreaLight() {
    RTRHardwareAreaLight light;
    light.position = float4(0.0f, 0.95f, -0.9f, 1.0f);
    light.right = float4(0.18f, 0.0f, 0.0f, 0.0f);
    light.up = float4(0.0f, 0.0f, 0.18f, 0.0f);
    light.forward = float4(0.0f, -1.0f, 0.0f, 0.0f);
    light.color = float4(18.0f, 17.5f, 17.0f, 0.0f);
    return light;
}

inline RTRHardwareAreaLight getAreaLight(constant RTRHardwareRayUniforms& uniforms, uint index = 0u) {
    if (uniforms.lightCount == 0u) {
        return defaultAreaLight();
    }
    const uint clamped = min(index, uniforms.lightCount - 1u);
    return uniforms.lights[clamped];
}

inline void writeHitDebug(uint2 gid,
                          uint width,
                          uint height,
                          device uint* buffer,
                          uint value) {
    if (buffer == nullptr || width == 0u || height == 0u) {
        return;
    }
    const uint linearIndex = gid.y * width + gid.x;
    if (linearIndex < width * height) {
        buffer[linearIndex] = value;
    }
}

inline void sampleAreaLight(RTRHardwareAreaLight light,
                            float2 u,
                            float3 position,
                            thread float3& lightDir,
                            thread float3& lightColor,
                            thread float& lightDistance) {
    const float2 mapped = u * 2.0f - 1.0f;
    const float3 samplePoint = light.position.xyz + light.right.xyz * mapped.x + light.up.xyz * mapped.y;
    const float3 toSample = samplePoint - position;
    lightDistance = length(toSample);
    const float invDistance = 1.0f / max(lightDistance, 1.0f);
    lightDir = toSample * invDistance;
    const float3 normal = normalize(light.forward.xyz);
    lightColor = light.color.xyz * (invDistance * invDistance);
    lightColor *= clamp(dot(-lightDir, normal), 0.0f, 1.0f);
}

constant uint kHaltonPrimes[] = {
    2,   3,   5,   7,   11,  13,  17,  19,
    23,  29,  31,  37,  41,  43,  47,  53,
    59,  61,  67,  71,  73,  79,  83,  89,
};

inline float halton(uint index, uint dimension) {
    const uint primeCount = static_cast<uint>(sizeof(kHaltonPrimes) / sizeof(uint));
    const uint primeIndex = (primeCount > 0u) ? (dimension % primeCount) : 0u;
    const uint base = kHaltonPrimes[primeIndex];
    float invBase = (base > 0u) ? (1.0f / static_cast<float>(base)) : 1.0f;
    float fraction = 1.0f;
    float result = 0.0f;
    uint i = index;
    while (i > 0u) {
        fraction *= invBase;
        result += fraction * static_cast<float>(i % base);
        i /= base;
    }
    return result;
}

}  // namespace

#if RTR_HAS_RAYTRACING
using namespace metal::raytracing;

struct HardwareHit {
    bool hit;
    uint primitiveIndex;
    float2 bary;
    float distance;
    uint instanceIndex;
    uint instanceUserId;
};

inline HardwareHit traceScene(acceleration_structure<instancing> accelerationStructure,
                              float3 origin,
                              float3 direction,
                              float minDistance,
                              float maxDistance,
                              uint rayMask = RTR_RAY_MASK_PRIMARY) {
    raytracing::ray sceneRay(origin, direction, minDistance, maxDistance);
    intersector<triangle_data, instancing, world_space_data> tracer;
    tracer.assume_geometry_type(geometry_type::triangle);
    tracer.force_opacity(forced_opacity::opaque);
    tracer.set_triangle_cull_mode(triangle_cull_mode::none);
    tracer.accept_any_intersection(false);

    const auto result = tracer.intersect(sceneRay, accelerationStructure, rayMask);

    HardwareHit hit{};
    if (result.type == intersection_type::triangle) {
        hit.hit = true;
        hit.primitiveIndex = result.primitive_id;
        hit.distance = result.distance;
        hit.bary = result.triangle_barycentric_coord;
        hit.instanceIndex = result.instance_id;
        hit.instanceUserId = result.instance_id;
    }
    return hit;
}

inline bool isOccluded(acceleration_structure<instancing> accelerationStructure,
                       float3 origin,
                       float3 direction,
                       float maxDistance) {
    const float epsilon = 1.0e-3f;
    raytracing::ray shadowRay(origin + direction * epsilon, direction, epsilon, maxDistance - epsilon);
    intersector<triangle_data, instancing, world_space_data> tracer;
    tracer.assume_geometry_type(geometry_type::triangle);
    tracer.force_opacity(forced_opacity::opaque);
    tracer.set_triangle_cull_mode(triangle_cull_mode::none);
    tracer.accept_any_intersection(true);

    const auto result = tracer.intersect(shadowRay, accelerationStructure, RTR_RAY_MASK_SHADOW);
    return result.type != intersection_type::none && result.distance < maxDistance - epsilon;
}


kernel void rayKernel(constant RTRHardwareRayUniforms& uniforms [[buffer(0)]],
                      const device packed_float3* positions [[buffer(1)]],
                      const device packed_float3* normals [[buffer(2)]],
                      const device uint* indices [[buffer(3)]],
                      const device packed_float3* colors [[buffer(4)]],
                      const device packed_float2* texcoords [[buffer(5)]],
                      const device RTRRayTracingMeshResource* meshes [[buffer(6)]],
                      const device RTRRayTracingMaterial* materials [[buffer(7)]],
                      constant RTRRayTracingTextureResource* textureInfos [[buffer(8)]],
                      const device float* texturePixels [[buffer(9)]],
                      constant MPSSceneLimits& limits [[buffer(10)]],
                      device uint* hitDebug [[buffer(11)]],
                      const device RTRRayTracingInstanceResource* instances [[buffer(12)]],
                      acceleration_structure<instancing> accelerationStructure [[buffer(15)]],
                      texture2d<unsigned int> randomTex [[texture(0)]],
                      texture2d<float, access::write> dstTex [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint width = uniforms.camera.width;
    const uint height = uniforms.camera.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    float2 jitter = float2(0.0f);
    if (randomTex.get_width() > 0 && randomTex.get_height() > 0) {
        const uint widthMask = randomTex.get_width();
        const uint heightMask = randomTex.get_height();
        const uint framePhaseX = (uniforms.camera.frameIndex * 12582917u) % max(widthMask, 1u);
        const uint framePhaseY = (uniforms.camera.frameIndex * 4256249u) % max(heightMask, 1u);
        const uint2 coord = uint2((gid.x + framePhaseX) % widthMask, (gid.y + framePhaseY) % heightMask);
        const uint4 noise = randomTex.read(coord);
        jitter.x = (static_cast<float>(noise.x & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
        jitter.y = (static_cast<float>(noise.y & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    }

    const float2 dims = float2(static_cast<float>(max(width, 1u)), static_cast<float>(max(height, 1u)));
    const float2 ndc = ((float2(gid) + 0.5f + jitter) / dims) * 2.0f - 1.0f;
    const float3 eye = uniforms.camera.eye.xyz;
    const float3 forward = uniforms.camera.forward.xyz;
    const float3 right = uniforms.camera.right.xyz;
    const float3 up = uniforms.camera.up.xyz;
    const float3 target = eye + forward + right * (ndc.x * uniforms.camera.imagePlaneHalfExtents.x) +
                          up * (ndc.y * uniforms.camera.imagePlaneHalfExtents.y);
    float3 rayDirection = normalize(target - eye);
    float3 rayOrigin = eye;

    const bool debugAlbedo = (uniforms.camera.flags & RTR_RAY_FLAG_DEBUG) != 0u;
    const bool debugInstanceColors = (uniforms.camera.flags & RTR_RAY_FLAG_INSTANCE_COLOR) != 0u;
    const bool debugInstanceTrace = (uniforms.camera.flags & RTR_RAY_FLAG_INSTANCE_TRACE) != 0u;
    const bool debugPrimitiveTrace = (uniforms.camera.flags & RTR_RAY_FLAG_PRIMITIVE_TRACE) != 0u;

    bool accelerationStructureNull = is_null_instance_acceleration_structure(accelerationStructure);

    if (accelerationStructureNull) {
        dstTex.write(float4(1.0f, 0.0f, 1.0f, 1.0f), gid);
        writeHitDebug(gid, uniforms.camera.width, uniforms.camera.height, hitDebug, 3u);
        return;
    }

    HardwareHit hit = traceScene(accelerationStructure, rayOrigin, rayDirection, 1.0e-3f, FLT_MAX);
    float3 color = sampleSkyColor(rayDirection);
    uint hitDebugValue = 0u;

    if (hit.hit) {
        const bool hasInstanceData = (instances != nullptr && limits.instanceCount > 0u);
        const uint rawInstanceId = hit.instanceUserId;
        const uint safeInstance = hasInstanceData ? min(rawInstanceId, limits.instanceCount - 1u) : 0u;
        const bool instanceOutOfRange = (!hasInstanceData) || (rawInstanceId >= limits.instanceCount);

        RTRRayTracingInstanceResource instanceInfo;
        if (hasInstanceData) {
            instanceInfo = instances[safeInstance];
        } else {
            instanceInfo.objectToWorld = float4x4(1.0f);
            instanceInfo.worldToObject = float4x4(1.0f);
            instanceInfo.materialIndex = RTR_INVALID_MATERIAL_INDEX;
            instanceInfo.meshIndex = 0u;
            instanceInfo.primitiveOffset = 0u;
            instanceInfo.primitiveCount = 0u;
        }

        RTRRayTracingMeshResource meshResource;
        if (meshes != nullptr && limits.meshCount > 0u) {
            const uint safeMesh = min(instanceInfo.meshIndex, limits.meshCount - 1u);
            meshResource = meshes[safeMesh];
        } else {
            meshResource.positionOffset = 0u;
            meshResource.normalOffset = 0u;
            meshResource.texcoordOffset = 0u;
            meshResource.colorOffset = 0u;
            meshResource.indexOffset = 0u;
            meshResource.vertexCount = limits.vertexCount;
            meshResource.indexCount = limits.indexCount;
            meshResource.materialIndex = RTR_INVALID_MATERIAL_INDEX;
        }

        float4x4 objectToWorld = instanceInfo.objectToWorld;
        float4x4 worldToObject = instanceInfo.worldToObject;

        const uint meshPrimitiveCount = max(meshResource.indexCount / 3u, 1u);
        const uint localPrimitive = min(hit.primitiveIndex, meshPrimitiveCount - 1u);
        uint globalPrimitive = instanceInfo.primitiveOffset + localPrimitive;
        if (instanceInfo.primitiveCount > 0u) {
            const uint primitiveEnd = instanceInfo.primitiveOffset + instanceInfo.primitiveCount - 1u;
            globalPrimitive = min(globalPrimitive, primitiveEnd);
        }

        if (debugPrimitiveTrace) {
            hitDebugValue = globalPrimitive + 1u;
            const uint p = globalPrimitive + 1u;
            color = float3(
                static_cast<float>((p * 13u) & 0xFFu) / 255.0f,
                static_cast<float>((p * 29u) & 0xFFu) / 255.0f,
                static_cast<float>((p * 53u) & 0xFFu) / 255.0f);
            writeHitDebug(gid, uniforms.camera.width, uniforms.camera.height, hitDebug, hitDebugValue);
            dstTex.write(float4(color, 1.0f), gid);
            return;
        } else if (debugInstanceTrace) {
            const uint meshBits = min(instanceInfo.meshIndex, 0xFFFFu);
            const uint instanceBits = min(rawInstanceId, 0xFFFFu);
            // Encode as 1-based so 0 remains "miss" in CPU-side debug decoding.
            hitDebugValue = ((instanceBits + 1u) << 16) | (meshBits + 1u);
            if (instanceOutOfRange) {
                hitDebugValue |= 0x80000000u;
            }
        } else {
            hitDebugValue = 1u;
        }

        if (debugInstanceTrace) {
            color = instanceOutOfRange ? float3(1.0f, 0.0f, 0.0f) : debugInstanceColor(instanceInfo.meshIndex);
            writeHitDebug(gid, uniforms.camera.width, uniforms.camera.height, hitDebug, hitDebugValue);
            dstTex.write(float4(color, 1.0f), gid);
            return;
        }

        const uint base = globalPrimitive * 3u;
        if (indices && base + 2u < limits.indexCount) {
            const uint i0 = indices[base + 0];
            const uint i1 = indices[base + 1];
            const uint i2 = indices[base + 2];
            if (i0 < limits.vertexCount && i1 < limits.vertexCount && i2 < limits.vertexCount) {
                const float w = clamp(1.0f - hit.bary.x - hit.bary.y, 0.0f, 1.0f);
                const float3 baryFull = float3(w, hit.bary.x, hit.bary.y);
                const float3 localV0 = float3(positions[i0]);
                const float3 localV1 = float3(positions[i1]);
                const float3 localV2 = float3(positions[i2]);
                const float3 localHit = localV0 * baryFull.x + localV1 * baryFull.y + localV2 * baryFull.z;
                const float3 hitPos = transformPosition(objectToWorld, localHit);

                float2 uv = float2(0.0f);
                if (texcoords && i0 < limits.texcoordCount && i1 < limits.texcoordCount && i2 < limits.texcoordCount) {
                    const float2 t0 = float2(texcoords[i0]);
                    const float2 t1 = float2(texcoords[i1]);
                    const float2 t2 = float2(texcoords[i2]);
                    uv = t0 * baryFull.x + t1 * baryFull.y + t2 * baryFull.z;
                }

                float3 vertexColour = float3(1.0f);
                if (colors && i0 < limits.colorCount && i1 < limits.colorCount && i2 < limits.colorCount) {
                    const float3 c0 = float3(colors[i0]);
                    const float3 c1 = float3(colors[i1]);
                    const float3 c2 = float3(colors[i2]);
                    vertexColour = clamp(c0 * baryFull.x + c1 * baryFull.y + c2 * baryFull.z, 0.0f, 1.0f);
                }

                const float3 localNormal = computeNormal(i0,
                                                        i1,
                                                        i2,
                                                        localV0,
                                                        localV1,
                                                        localV2,
                                                        normals,
                                                        limits.normalCount,
                                                        baryFull);
                float3 normal = transformNormal(worldToObject, localNormal);
                const RTRRayTracingMaterial material = loadMaterial(instanceInfo.materialIndex,
                                                                    materials,
                                                                    limits.materialCount,
                                                                    meshResource.materialIndex);
                const float3 sampledColour = clamp(sampleMaterialColor(material,
                                                                       uv,
                                                                       textureInfos,
                                                                       limits.textureCount,
                                                                       texturePixels),
                                                    0.0f,
                                                    1.0f);
                const float3 baseColour = sampledColour * vertexColour;

                if (debugInstanceColors) {
                    color = debugInstanceColor(instanceInfo.meshIndex);
                } else if (debugAlbedo) {
                    color = baseColour;
                } else {
                    const float3 viewDir = normalize(-rayDirection);
                    float3 lighting = baseColour * 0.05f + material.emission;

                    const RTRHardwareAreaLight light = getAreaLight(uniforms);
                    const uint pixelSeed = gid.x * 73856093u ^ gid.y * 19349663u;
                    const uint frameBase = uniforms.camera.frameIndex * kLightSamplesPerPixel;
                    const float sampleWeight = 1.0f / static_cast<float>(kLightSamplesPerPixel);
                    for (uint sampleIdx = 0u; sampleIdx < kLightSamplesPerPixel; ++sampleIdx) {
                        const uint sequenceIndex = pixelSeed + frameBase + sampleIdx + 1u;
                        const float2 lightSamples = float2(halton(sequenceIndex, 2u),
                                                           halton(sequenceIndex, 3u));
                        float3 lightDir;
                        float3 lightColor;
                        float lightDistance;
                        sampleAreaLight(light, lightSamples, hitPos, lightDir, lightColor, lightDistance);

                        const float NdotL = max(dot(normal, lightDir), 0.0f);
                        const float NdotV = max(dot(normal, viewDir), 0.0f);
                        if (NdotL > 1.0e-4f && NdotV > 1.0e-4f && all(isfinite(lightDir))) {
                            const bool occluded = isOccluded(accelerationStructure, hitPos, lightDir, lightDistance);
                            if (!occluded) {
                                const float3 halfVec = normalize(lightDir + viewDir);
                                const float NdotH = max(dot(normal, halfVec), 0.0f);
                                const float VdotH = max(dot(viewDir, halfVec), 0.0f);
                                const float alpha = max(material.roughness * material.roughness, 1.0e-3f);
                                const float D = distributionGGX(NdotH, alpha);
                                const float G = geometrySmith(NdotV, NdotL, material.roughness);
                                const float3 F = fresnelSchlick(VdotH,
                                                                mix(float3(0.04f), baseColour, material.metallic));
                                const float3 specular = (D * G * F) / max(4.0f * NdotV * NdotL, 1.0e-4f);
                                const float3 kd = (float3(1.0f) - F) * (1.0f - material.metallic);
                                const float3 diffuse = kd * baseColour / 3.14159265f;
                                lighting += (diffuse + specular) * lightColor * NdotL * sampleWeight;
                            }
                        }
                    }

                    const bool reflectiveSurface = (material.materialFlags & RTR_MATERIAL_FLAG_REFLECTIVE) != 0u;
                    const bool refractiveSurface = (material.materialFlags & RTR_MATERIAL_FLAG_REFRACTIVE) != 0u;
                    const float reflectivity = clamp(material.reflectivity, 0.0f, 1.0f);
                    if (!debugInstanceColors && uniforms.maxBounces > 1u) {
                        if (reflectiveSurface && reflectivity > 0.0f) {
                            const float3 reflectedDir = normalize(reflect(rayDirection, normal));
                            HardwareHit bounce = traceScene(accelerationStructure, hitPos, reflectedDir, 1.0e-3f, FLT_MAX);
                            float3 reflection = sampleSkyColor(reflectedDir);
                            if (bounce.hit) {
                                RTRRayTracingInstanceResource bounceInfo;
                                if (instances != nullptr && limits.instanceCount > 0u) {
                                    const uint bounceInstance = min(bounce.instanceUserId, limits.instanceCount - 1u);
                                    bounceInfo = instances[bounceInstance];
                                } else {
                                    bounceInfo.materialIndex = RTR_INVALID_MATERIAL_INDEX;
                                    bounceInfo.meshIndex = 0u;
                                }

                                RTRRayTracingMeshResource bounceMesh;
                                if (meshes != nullptr && limits.meshCount > 0u) {
                                    const uint safeBounceMesh = min(bounceInfo.meshIndex, limits.meshCount - 1u);
                                    bounceMesh = meshes[safeBounceMesh];
                                } else {
                                    bounceMesh.positionOffset = 0u;
                                    bounceMesh.indexOffset = 0u;
                                    bounceMesh.vertexCount = limits.vertexCount;
                                    bounceMesh.indexCount = limits.indexCount;
                                    bounceMesh.materialIndex = RTR_INVALID_MATERIAL_INDEX;
                                }

                                const uint bounceMeshPrimitiveCount = max(bounceMesh.indexCount / 3u, 1u);
                                const uint bounceLocalPrimitive = min(bounce.primitiveIndex, bounceMeshPrimitiveCount - 1u);
                                uint bounceGlobalPrimitive = bounceInfo.primitiveOffset + bounceLocalPrimitive;
                                if (bounceInfo.primitiveCount > 0u) {
                                    const uint bounceEnd = bounceInfo.primitiveOffset + bounceInfo.primitiveCount - 1u;
                                    bounceGlobalPrimitive = min(bounceGlobalPrimitive, bounceEnd);
                                }
                                const uint bounceBase = bounceGlobalPrimitive * 3u;
                                if (bounceBase + 2u < limits.indexCount) {
                                    const uint bi0 = indices[bounceBase + 0];
                                    const uint bi1 = indices[bounceBase + 1];
                                    const uint bi2 = indices[bounceBase + 2];
                                    if (bi0 < limits.vertexCount && bi1 < limits.vertexCount && bi2 < limits.vertexCount) {
                                        const float wBounce = clamp(1.0f - bounce.bary.x - bounce.bary.y, 0.0f, 1.0f);
                                        const float3 baryBounce = float3(wBounce, bounce.bary.x, bounce.bary.y);
                                        const float3 rb0 = float3(positions[bi0]);
                                        const float3 rb1 = float3(positions[bi1]);
                                        const float3 rb2 = float3(positions[bi2]);
                                        const float3 bounceLocalPos = rb0 * baryBounce.x + rb1 * baryBounce.y + rb2 * baryBounce.z;
                                        const float3 bouncePos = transformPosition(bounceInfo.objectToWorld, bounceLocalPos);
                                        float2 bounceUV = float2(0.0f);
                                        if (texcoords && bi0 < limits.texcoordCount && bi1 < limits.texcoordCount && bi2 < limits.texcoordCount) {
                                            const float2 t0 = float2(texcoords[bi0]);
                                            const float2 t1 = float2(texcoords[bi1]);
                                            const float2 t2 = float2(texcoords[bi2]);
                                            bounceUV = t0 * baryBounce.x + t1 * baryBounce.y + t2 * baryBounce.z;
                                        }
                                        const RTRRayTracingMaterial bounceMaterial = loadMaterial(bounceInfo.materialIndex,
                                                                                                   materials,
                                                                                                   limits.materialCount,
                                                                                                   bounceMesh.materialIndex);
                                        reflection = clamp(sampleMaterialColor(bounceMaterial,
                                                                               bounceUV,
                                                                               textureInfos,
                                                                               limits.textureCount,
                                                                               texturePixels),
                                                           0.0f,
                                                           1.0f);
                                        reflection += bounceMaterial.emission;
                                        const float bounceFade = exp(-0.2f * length(bouncePos - hitPos));
                                        reflection *= bounceFade;
                                    }
                                }
                            }
                            lighting += reflection * reflectivity;
                        }

                        if (refractiveSurface) {
                            const float ior = clamp(material.indexOfRefraction, 1.01f, 3.0f);
                            float3 refractionNormal = normal;
                            float cosIncident = dot(-rayDirection, refractionNormal);
                            float eta = 1.0f / ior;
                            if (cosIncident < 0.0f) {
                                cosIncident = -cosIncident;
                                refractionNormal = -refractionNormal;
                                eta = ior;
                            }

                            float3 refractedDir = refract(rayDirection, refractionNormal, eta);
                            if (length_squared(refractedDir) < 1.0e-6f || !all(isfinite(refractedDir))) {
                                refractedDir = normalize(reflect(rayDirection, refractionNormal));
                            } else {
                                refractedDir = normalize(refractedDir);
                            }

                            HardwareHit transmittedHit =
                                traceScene(accelerationStructure, hitPos, refractedDir, 1.0e-3f, FLT_MAX);
                            float3 transmittedColor = sampleSkyColor(refractedDir);
                            if (transmittedHit.hit) {
                                RTRRayTracingInstanceResource transmittedInfo;
                                if (instances != nullptr && limits.instanceCount > 0u) {
                                    const uint transmittedInstance =
                                        min(transmittedHit.instanceUserId, limits.instanceCount - 1u);
                                    transmittedInfo = instances[transmittedInstance];
                                } else {
                                    transmittedInfo.materialIndex = RTR_INVALID_MATERIAL_INDEX;
                                    transmittedInfo.meshIndex = 0u;
                                }

                                RTRRayTracingMeshResource transmittedMesh;
                                if (meshes != nullptr && limits.meshCount > 0u) {
                                    const uint safeTransmittedMesh =
                                        min(transmittedInfo.meshIndex, limits.meshCount - 1u);
                                    transmittedMesh = meshes[safeTransmittedMesh];
                                } else {
                                    transmittedMesh.positionOffset = 0u;
                                    transmittedMesh.indexOffset = 0u;
                                    transmittedMesh.vertexCount = limits.vertexCount;
                                    transmittedMesh.indexCount = limits.indexCount;
                                    transmittedMesh.materialIndex = RTR_INVALID_MATERIAL_INDEX;
                                }

                                const uint transmittedPrimitiveCount = max(transmittedMesh.indexCount / 3u, 1u);
                                const uint transmittedLocalPrimitive =
                                    min(transmittedHit.primitiveIndex, transmittedPrimitiveCount - 1u);
                                uint transmittedGlobalPrimitive =
                                    transmittedInfo.primitiveOffset + transmittedLocalPrimitive;
                                if (transmittedInfo.primitiveCount > 0u) {
                                    const uint transmittedEnd =
                                        transmittedInfo.primitiveOffset + transmittedInfo.primitiveCount - 1u;
                                    transmittedGlobalPrimitive = min(transmittedGlobalPrimitive, transmittedEnd);
                                }
                                const uint transmittedBase = transmittedGlobalPrimitive * 3u;
                                if (transmittedBase + 2u < limits.indexCount) {
                                    const uint ti0 = indices[transmittedBase + 0];
                                    const uint ti1 = indices[transmittedBase + 1];
                                    const uint ti2 = indices[transmittedBase + 2];
                                    if (ti0 < limits.vertexCount && ti1 < limits.vertexCount && ti2 < limits.vertexCount) {
                                        const float wTransmit =
                                            clamp(1.0f - transmittedHit.bary.x - transmittedHit.bary.y, 0.0f, 1.0f);
                                        const float3 baryTransmit =
                                            float3(wTransmit, transmittedHit.bary.x, transmittedHit.bary.y);
                                        float2 transmittedUV = float2(0.0f);
                                        if (texcoords &&
                                            ti0 < limits.texcoordCount &&
                                            ti1 < limits.texcoordCount &&
                                            ti2 < limits.texcoordCount) {
                                            const float2 t0 = float2(texcoords[ti0]);
                                            const float2 t1 = float2(texcoords[ti1]);
                                            const float2 t2 = float2(texcoords[ti2]);
                                            transmittedUV = t0 * baryTransmit.x + t1 * baryTransmit.y + t2 * baryTransmit.z;
                                        }
                                        const RTRRayTracingMaterial transmittedMaterial = loadMaterial(
                                            transmittedInfo.materialIndex,
                                            materials,
                                            limits.materialCount,
                                            transmittedMesh.materialIndex);
                                        transmittedColor = clamp(sampleMaterialColor(transmittedMaterial,
                                                                                     transmittedUV,
                                                                                     textureInfos,
                                                                                     limits.textureCount,
                                                                                     texturePixels),
                                                                 0.0f,
                                                                 1.0f);
                                        transmittedColor += transmittedMaterial.emission;
                                    }
                                }
                            }

                            const float f0Base = (ior - 1.0f) / (ior + 1.0f);
                            const float f0 = f0Base * f0Base;
                            const float fresnel = clamp(f0 + (1.0f - f0) * pow(1.0f - cosIncident, 5.0f), 0.0f, 1.0f);
                            const float transmissionWeight = 1.0f - fresnel;
                            lighting += transmittedColor * transmissionWeight;
                        }
                    }

                    color = clamp(lighting, 0.0f, 4.0f);
                }
            }
        }
    }

    writeHitDebug(gid, uniforms.camera.width, uniforms.camera.height, hitDebug, hitDebugValue);

    dstTex.write(float4(color, 1.0f), gid);
}

#endif // RTR_HAS_RAYTRACING

struct RTRDisplayVertexOutput {
    float4 position [[position]];
    float2 uv;
};

vertex RTRDisplayVertexOutput RTRDisplayVertex(uint vertexID [[vertex_id]]) {
    RTRDisplayVertexOutput out;
    const float2 positions[4] = {
        float2(-1.0f, -1.0f),
        float2(1.0f, -1.0f),
        float2(-1.0f, 1.0f),
        float2(1.0f, 1.0f),
    };
    const float2 uvs[4] = {
        float2(0.0f, 0.0f),
        float2(1.0f, 0.0f),
        float2(0.0f, 1.0f),
        float2(1.0f, 1.0f),
    };
    const uint safeIndex = vertexID & 0x3u;
    out.position = float4(positions[safeIndex], 0.0f, 1.0f);
    out.uv = uvs[safeIndex];
    return out;
}

fragment float4 RTRDisplayFragment(RTRDisplayVertexOutput in [[stage_in]],
                                   texture2d<float> colorTexture [[texture(0)]],
                                   constant float2& invRenderSize [[buffer(0)]]) {
    constexpr sampler displaySampler(address::clamp_to_edge, filter::linear);
    float2 uv = clamp(in.uv, 0.0f, 1.0f);
    if (invRenderSize.x > 0.0f && invRenderSize.y > 0.0f) {
        // Nudge sample into texel centers to avoid sampling issues when renderSize != drawableSize.
        uv = uv * (float2(1.0f) - invRenderSize) + invRenderSize * 0.5f;
    }
    return colorTexture.sample(displaySampler, uv);
}
