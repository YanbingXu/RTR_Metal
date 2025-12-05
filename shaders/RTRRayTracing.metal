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

inline RTRRayTracingMaterial loadMaterial(uint primitiveIndex,
                                          const device uint* primitiveMaterials,
                                          const device RTRRayTracingMaterial* materials,
                                          uint primitiveCount,
                                          uint materialCount) {
    if (!primitiveMaterials || !materials || primitiveCount == 0 || materialCount == 0) {
        return makeFallbackMaterial();
    }
    const uint clampedPrimitive = min(primitiveIndex, primitiveCount - 1u);
    uint materialIndex = primitiveMaterials[clampedPrimitive];
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
    uint instanceId;
};

inline HardwareHit traceScene(acceleration_structure<instancing> accelerationStructure,
                              float3 origin,
                              float3 direction,
                              float minDistance,
                              float maxDistance,
                              uint rayMask = RTR_RAY_MASK_PRIMARY) {
    raytracing::ray sceneRay(origin, direction, minDistance, maxDistance);
    intersector<instancing, triangle_data> tracer;
    tracer.assume_geometry_type(geometry_type::triangle);
    tracer.force_opacity(forced_opacity::opaque);
    tracer.accept_any_intersection(false);

    const auto result = tracer.intersect(sceneRay, accelerationStructure, rayMask);

    HardwareHit hit{};
    if (result.type == intersection_type::triangle) {
        hit.hit = true;
        hit.primitiveIndex = result.primitive_id;
        hit.distance = result.distance;
        hit.bary = result.triangle_barycentric_coord;
        hit.instanceId = result.instance_id;
    }
    return hit;
}

inline bool isOccluded(acceleration_structure<instancing> accelerationStructure,
                       float3 origin,
                       float3 direction,
                       float maxDistance) {
    const float epsilon = 1.0e-3f;
    raytracing::ray shadowRay(origin + direction * epsilon, direction, epsilon, maxDistance - epsilon);
    intersector<instancing, triangle_data> tracer;
    tracer.assume_geometry_type(geometry_type::triangle);
    tracer.force_opacity(forced_opacity::opaque);
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
                      const device uint* primitiveMaterials [[buffer(6)]],
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
        const uint2 coord = uint2(gid.x % randomTex.get_width(), gid.y % randomTex.get_height());
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
    const float3 rayDirection = normalize(target - eye);

    const bool debugAlbedo = (uniforms.camera.flags & RTR_RAY_FLAG_DEBUG) != 0u;

    bool accelerationStructureNull = is_null_instance_acceleration_structure(accelerationStructure);

    if (accelerationStructureNull) {
        dstTex.write(float4(1.0f, 0.0f, 1.0f, 1.0f), gid);
        if (hitDebug != nullptr) {
            const uint width = uniforms.camera.width;
            const uint height = uniforms.camera.height;
            if (width > 0 && height > 0) {
                const uint linearIndex = gid.y * width + gid.x;
                if (linearIndex < width * height) {
                    hitDebug[linearIndex] = 3u;
                }
            }
        }
        return;
    }

    HardwareHit hit = traceScene(accelerationStructure, eye, rayDirection, 1.0e-3f, FLT_MAX);

    if (hitDebug != nullptr) {
        const uint width = uniforms.camera.width;
        const uint height = uniforms.camera.height;
        if (width > 0 && height > 0) {
            const uint linearIndex = gid.y * width + gid.x;
            if (linearIndex < width * height) {
                hitDebug[linearIndex] = hit.hit ? 1u : 0u;
            }
        }
    }
    float3 color = sampleSkyColor(rayDirection);

    if (hit.hit) {
        uint primitiveIndex = hit.primitiveIndex;
        if (instances != nullptr && limits.instanceCount > 0u) {
            const uint safeInstance = min(hit.instanceId, limits.instanceCount - 1u);
            RTRRayTracingInstanceResource instanceInfo = instances[safeInstance];
            const uint primitiveBase = instanceInfo.primitiveOffset + hit.primitiveIndex;
            const uint primitiveCap = (limits.primitiveCount > 0u) ? (limits.primitiveCount - 1u) : 0u;
            primitiveIndex = min(primitiveBase, primitiveCap);
        }

        const uint base = primitiveIndex * 3u;
        if (indices && base + 2u < limits.indexCount) {
            const uint i0 = indices[base + 0];
            const uint i1 = indices[base + 1];
            const uint i2 = indices[base + 2];
            if (i0 < limits.vertexCount && i1 < limits.vertexCount && i2 < limits.vertexCount) {
                const float w = clamp(1.0f - hit.bary.x - hit.bary.y, 0.0f, 1.0f);
                const float3 baryFull = float3(w, hit.bary.x, hit.bary.y);
                const float3 v0 = float3(positions[i0]);
                const float3 v1 = float3(positions[i1]);
                const float3 v2 = float3(positions[i2]);
                const float3 hitPos = v0 * baryFull.x + v1 * baryFull.y + v2 * baryFull.z;

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

                float3 normal = computeNormal(i0, i1, i2, v0, v1, v2, normals, limits.normalCount, baryFull);
                const RTRRayTracingMaterial material = loadMaterial(primitiveIndex,
                                                                    primitiveMaterials,
                                                                    materials,
                                                                    limits.primitiveCount,
                                                                    limits.materialCount);
                const float3 sampledColour = clamp(sampleMaterialColor(material,
                                                                       uv,
                                                                       textureInfos,
                                                                       limits.textureCount,
                                                                       texturePixels),
                                                    0.0f,
                                                    1.0f);
                const float3 baseColour = sampledColour * vertexColour;

                if (debugAlbedo) {
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

                    const float reflectivity = clamp(material.reflectivity, 0.0f, 1.0f);
                    if (reflectivity > 0.0f && uniforms.maxBounces > 1u) {
                        const float3 reflectedDir = normalize(reflect(rayDirection, normal));
                        HardwareHit bounce = traceScene(accelerationStructure, hitPos, reflectedDir, 1.0e-3f, FLT_MAX);
                        float3 reflection = sampleSkyColor(reflectedDir);
                        if (bounce.hit) {
                            uint bouncePrimitiveIndex = bounce.primitiveIndex;
                            if (instances != nullptr && limits.instanceCount > 0u) {
                                const uint bounceInstance = min(bounce.instanceId, limits.instanceCount - 1u);
                                RTRRayTracingInstanceResource bounceInfo = instances[bounceInstance];
                                const uint bounceOffset = bounceInfo.primitiveOffset + bounce.primitiveIndex;
                                const uint primitiveCap = (limits.primitiveCount > 0u) ? (limits.primitiveCount - 1u) : 0u;
                                bouncePrimitiveIndex = min(bounceOffset, primitiveCap);
                            }

                            if (bouncePrimitiveIndex * 3u + 2u < limits.indexCount) {
                                const uint bounceBase = bouncePrimitiveIndex * 3u;
                                const uint bi0 = indices[bounceBase + 0];
                                const uint bi1 = indices[bounceBase + 1];
                                const uint bi2 = indices[bounceBase + 2];
                                if (bi0 < limits.vertexCount && bi1 < limits.vertexCount && bi2 < limits.vertexCount) {
                                    const float wBounce = clamp(1.0f - bounce.bary.x - bounce.bary.y, 0.0f, 1.0f);
                                    const float3 baryBounce = float3(wBounce, bounce.bary.x, bounce.bary.y);
                                    const float3 rb0 = float3(positions[bi0]);
                                    const float3 rb1 = float3(positions[bi1]);
                                    const float3 rb2 = float3(positions[bi2]);
                                    const float3 bouncePos = rb0 * baryBounce.x + rb1 * baryBounce.y + rb2 * baryBounce.z;
                                    float2 bounceUV = float2(0.0f);
                                    if (texcoords && bi0 < limits.texcoordCount && bi1 < limits.texcoordCount && bi2 < limits.texcoordCount) {
                                        const float2 t0 = float2(texcoords[bi0]);
                                        const float2 t1 = float2(texcoords[bi1]);
                                        const float2 t2 = float2(texcoords[bi2]);
                                        bounceUV = t0 * baryBounce.x + t1 * baryBounce.y + t2 * baryBounce.z;
                                    }
                                    const RTRRayTracingMaterial bounceMaterial = loadMaterial(bouncePrimitiveIndex,
                                                                                               primitiveMaterials,
                                                                                               materials,
                                                                                               limits.primitiveCount,
                                                                                               limits.materialCount);
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

                    color = clamp(lighting, 0.0f, 4.0f);
                }
            }
        }
    }

    dstTex.write(float4(color, 1.0f), gid);
}

kernel void accumulateKernel(constant RTRHardwareRayUniforms& uniforms [[buffer(0)]],
                             texture2d<float, access::read> renderTex [[texture(0)]],
                             texture2d<float, access::read_write> sumTex [[texture(1)]],
                             texture2d<float, access::write> accumTex [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    const uint width = uniforms.camera.width;
    const uint height = uniforms.camera.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    float3 currentSum = renderTex.read(gid).xyz;
    if (uniforms.camera.frameIndex > 0u) {
        currentSum += sumTex.read(gid).xyz;
    }

    sumTex.write(float4(currentSum, 1.0f), gid);
    const float frameCount = static_cast<float>(uniforms.camera.frameIndex + 1u);
    const float3 averaged = currentSum / max(frameCount, 1.0f);
    accumTex.write(float4(averaged, 1.0f), gid);
}

#endif // RTR_HAS_RAYTRACING

struct RTRDisplayVertexOutput {
    float4 position [[position]];
    float2 uv;
};

vertex RTRDisplayVertexOutput RTRDisplayVertex(uint vertexID [[vertex_id]]) {
    RTRDisplayVertexOutput out;
    const float2 positions[3] = {
        float2(-1.0f, -1.0f),
        float2(3.0f, -1.0f),
        float2(-1.0f, 3.0f),
    };
    const float2 clipPosition = positions[vertexID];
    out.position = float4(clipPosition, 0.0f, 1.0f);
    // Map clip-space coordinates [-1, 1] directly into texture space [0, 1].
    out.uv = clipPosition * 0.5f + 0.5f;
    return out;
}

fragment float4 RTRDisplayFragment(RTRDisplayVertexOutput in [[stage_in]],
                                   texture2d<float> colorTexture [[texture(0)]]) {
    constexpr sampler displaySampler(address::clamp_to_edge, filter::linear);
    return colorTexture.sample(displaySampler, in.uv);
}
