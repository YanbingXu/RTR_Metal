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
#if RTR_HAS_RAYTRACING
using namespace metal::raytracing;
#endif

namespace {

inline uint mixBits(uint value) {
    value ^= value >> 17;
    value *= 0xed5ad4bbU;
    value ^= value >> 11;
    value *= 0xac4c1b51U;
    value ^= value >> 15;
    value *= 0x31848babU;
    value ^= value >> 14;
    return value;
}

inline float2 jitterForPixel(uint2 gid, constant MPSSamplingUniforms& sampling) {
    const uint sampleIndex = (sampling.samplesPerPixel == 0)
                                 ? sampling.sampleIndex
                                 : min(sampling.sampleIndex, sampling.samplesPerPixel - 1);
    if (sampleIndex == 0 && sampling.baseSeed == 0) {
        return float2(0.0f);
    }
    const uint base = mixBits(gid.x ^ (gid.y << 16) ^ (sampling.baseSeed * 0x9E3779B9U) ^ sampleIndex);
    const uint hashX = mixBits(base ^ 0x68bc21ebu);
    const uint hashY = mixBits(base ^ 0x02e5be93u);
    const float jitterX = (static_cast<float>(hashX & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    const float jitterY = (static_cast<float>(hashY & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    return float2(jitterX, jitterY);
}

struct RTRVertexSample {
    float3 position;
    float3 normal;
    float2 texcoord;
};

inline RTRVertexSample loadVertexSample(const device uchar* base, uint stride, uint index) {
    const device uchar* bytes = base + stride * index;
    RTRVertexSample sample;
    sample.position = *reinterpret_cast<const device float3*>(bytes);
    sample.normal = *reinterpret_cast<const device float3*>(bytes + 16);
    sample.texcoord = *reinterpret_cast<const device float2*>(bytes + 32);
    return sample;
}

inline float wrapCoordinate(float value) {
    value = value - floor(value);
    return (value < 0.0f) ? value + 1.0f : value;
}

inline float3 accumulateColor(texture2d<float, access::read_write> accumulation,
                              uint2 gid,
                              constant RTRRayTracingUniforms& uniforms,
                              float3 sample,
                              bool allowAccumulation) {
    if (accumulation.get_width() == 0 || accumulation.get_height() == 0) {
        return sample;
    }

    if (!allowAccumulation) {
        return sample;
    }

    if (uniforms.frameIndex == 0u) {
        accumulation.write(float4(sample, 1.0f), gid);
        return sample;
    }

    float4 previous = accumulation.read(gid);
    const float prevSamples = max(previous.w, 1.0f);
    const float newSamples = prevSamples + 1.0f;
    const float3 blended = (previous.xyz * prevSamples + sample) / newSamples;
    accumulation.write(float4(blended, newSamples), gid);
    return blended;
}

inline float2 pseudoRandom(uint2 gid, uint frameIndex) {
    const uint base = mixBits(gid.x * 73856093u ^ gid.y * 19349663u ^ ((frameIndex + 1u) * 83492791u));
    const uint hashX = mixBits(base ^ 0x9e3779b9u);
    const uint hashY = mixBits(base ^ 0x7f4a7c15u);
    const float rx = (static_cast<float>(hashX & 0xFFFFFFu) + 0.5f) / 16777216.0f;
    const float ry = (static_cast<float>(hashY & 0xFFFFFFu) + 0.5f) / 16777216.0f;
    return float2(rx, ry);
}

inline bool traceShadowRay(instance_acceleration_structure scene,
                           float3 origin,
                           float3 direction,
                           float maxDistance) {
    if (is_null_instance_acceleration_structure(scene)) {
        return false;
    }
    ray shadowRay(origin, direction, 0.001f, maxDistance);
    intersection_params params;
    params.assume_geometry_type(geometry_type::triangle);
    params.force_opacity(forced_opacity::opaque);
    intersection_query<triangle_data, instancing> shadowQuery;
    shadowQuery.reset(shadowRay, scene, ~0u, params);
    while (shadowQuery.next()) {}
    return shadowQuery.get_committed_intersection_type() == intersection_type::triangle;
}

inline float distributionGGX(float nDotH, float alpha) {
    const float a2 = alpha * alpha;
    const float denom = (nDotH * nDotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(3.14159265f * denom * denom, 1e-4f);
}

inline float geometrySchlickGGX(float nDotV, float k) {
    return nDotV / (nDotV * (1.0f - k) + k);
}

inline float geometrySmith(float nDotV, float nDotL, float roughness) {
    const float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    return geometrySchlickGGX(nDotV, k) * geometrySchlickGGX(nDotL, k);
}

inline float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0f) - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

inline float4 readTextureTexel(const RTRRayTracingTextureResource info,
                               uint x,
                               uint y,
                               device const float* pixels) {
    const uint base = info.dataOffset + (y * info.rowPitch + x) * 4u;
    return float4(pixels[base + 0], pixels[base + 1], pixels[base + 2], pixels[base + 3]);
}

inline float4 sampleTexture(uint textureIndex,
                            constant RTRRayTracingTextureResource* infos,
                            uint textureCount,
                            device const float* pixels,
                            float2 uv) {
    if (textureIndex == RTR_INVALID_TEXTURE_INDEX || infos == nullptr || pixels == nullptr) {
        return float4(1.0f);
    }
    if (textureIndex >= textureCount) {
        return float4(1.0f);
    }

    const RTRRayTracingTextureResource info = infos[textureIndex];
    if (info.width == 0 || info.height == 0 || info.rowPitch == 0) {
        return float4(1.0f);
    }

    const float width = static_cast<float>(info.width);
    const float height = static_cast<float>(info.height);
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

}  // namespace

kernel void rayGenMain(texture2d<float, access::write> output [[texture(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    float2 uv = float2(gid) / float2(output.get_width(), output.get_height());
    output.write(float4(uv, 0.5, 1.0), gid);
}

kernel void missMain(texture2d<float, access::write> output [[texture(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    output.write(float4(0.1, 0.1, 0.4, 1.0), gid);
}

kernel void closestHitMain(texture2d<float, access::write> output [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    output.write(float4(0.8, 0.8, 0.8, 1.0), gid);
}

kernel void rtGradientKernel(constant RTRRayTracingUniforms& uniforms [[buffer(1)]],
                             device const RTRRayTracingResourceHeader* resourceHeader [[buffer(2)]],
                             device const RTRRayTracingMeshResource* meshResources [[buffer(3)]],
                             device const uchar* fallbackVertexBytes [[buffer(4)]],
                             device const uint* fallbackIndices [[buffer(5)]],
                             device const RTRRayTracingInstanceResource* instances [[buffer(6)]],
                             device const RTRRayTracingMaterial* materials [[buffer(7)]],
                             texture2d<float, access::write> output [[texture(0)]],
                             texture2d<float, access::read_write> accumulation [[texture(1)]],
                             texture2d<float, access::read> randomTex [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const bool debugAlbedo = (uniforms.flags & RTR_RAY_FLAG_DEBUG) != 0u;
    const bool accumulationEnabled = (uniforms.flags & RTR_RAY_FLAG_ACCUMULATE) != 0u;
    if (debugAlbedo) {
        const float4 debugColour = float4(1.0f, 0.0f, 0.0f, 1.0f);
        output.write(debugColour, gid);
        if (accumulation.get_width() == uniforms.width && accumulation.get_height() == uniforms.height) {
            accumulation.write(debugColour, gid);
        }
        return;
    }

    const float2 dims = float2(max(uniforms.width, 1u), max(uniforms.height, 1u));
    float2 uv = float2(gid) / dims;

    const uint geometryCount = (resourceHeader != nullptr) ? resourceHeader->geometryCount : 0u;
    const uint randomWidth = (resourceHeader != nullptr) ? resourceHeader->randomTextureWidth : 0u;
    const uint randomHeight = (resourceHeader != nullptr) ? resourceHeader->randomTextureHeight : 0u;

    (void)meshResources;
    (void)fallbackVertexBytes;
    (void)fallbackIndices;
    (void)instances;
    (void)materials;

    const float geomBoost = geometryCount > 0 ? 0.5f : 0.0f;

    const uint noiseWidth = randomWidth > 0 ? randomWidth : randomTex.get_width();
    const uint noiseHeight = randomHeight > 0 ? randomHeight : randomTex.get_height();
    float noise = 0.0f;
    if (noiseWidth > 0 && noiseHeight > 0) {
        const uint2 noiseCoord = uint2(gid.x % noiseWidth, gid.y % noiseHeight);
        noise = randomTex.read(noiseCoord).x;
    }

    float3 colour = float3(uv * (0.5f + geomBoost * 0.5f), 0.35f + 0.4f * sin((float)uniforms.frameIndex * 0.1f + noise));
    colour = clamp(colour, 0.0f, 1.0f);
    float3 finalColour = debugAlbedo ? colour
                                     : accumulateColor(accumulation, gid, uniforms, colour, accumulationEnabled);
    output.write(float4(finalColour, 1.0f), gid);
}

#if RTR_HAS_RAYTRACING
kernel void rtHardwareKernel(instance_acceleration_structure scene [[buffer(0)]],
                             constant RTRRayTracingUniforms& uniforms [[buffer(1)]],
                             device const RTRRayTracingResourceHeader* resourceHeader [[buffer(2)]],
                             device const RTRRayTracingMeshResource* meshResources [[buffer(3)]],
                             device const uchar* fallbackVertexBytes [[buffer(4)]],
                             device const uint* fallbackIndices [[buffer(5)]],
                             device const RTRRayTracingInstanceResource* instances [[buffer(6)]],
                             device const RTRRayTracingMaterial* materials [[buffer(7)]],
                             constant RTRRayTracingTextureResource* textureInfos [[buffer(8)]],
                             device const float* texturePixels [[buffer(9)]],
                             texture2d<float, access::write> output [[texture(0)]],
                             texture2d<float, access::read_write> accumulation [[texture(1)]],
                             texture2d<float, access::read> randomTex [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const float2 dims = float2(max(uniforms.width, 1u), max(uniforms.height, 1u));
    const uint textureCount = (resourceHeader != nullptr) ? resourceHeader->textureCount : 0u;
    const bool debugAlbedo = (uniforms.flags & RTR_RAY_FLAG_DEBUG) != 0u;
    const bool accumulationEnabled = (uniforms.flags & RTR_RAY_FLAG_ACCUMULATE) != 0u;
    const float2 pixel = float2(gid) + 0.5f;
    const float2 ndc = (pixel / dims - 0.5f) * 2.0f;

    const float3 eye = uniforms.eye.xyz;
    const float3 forward = uniforms.forward.xyz;
    const float3 right = uniforms.right.xyz;
    const float3 up = uniforms.up.xyz;
    const float3 target = eye + forward + right * (ndc.x * uniforms.imagePlaneHalfExtents.x) +
                          up * (ndc.y * uniforms.imagePlaneHalfExtents.y);
    const float3 direction = normalize(target - eye);

    const float rayMin = 0.001f;
    const float rayMax = 1.0e6f;
    ray primaryRay(eye, direction, rayMin, rayMax);

    float3 colour = float3(0.04f, 0.06f, 0.10f);

    constexpr uint kInvalidOffset = 0xFFFFFFFFu;

    if (!is_null_instance_acceleration_structure(scene)) {
        intersection_params params;
        params.assume_geometry_type(geometry_type::triangle);
        params.force_opacity(forced_opacity::opaque);

        intersection_query<triangle_data, instancing> query;
        query.reset(primaryRay, scene, ~0u, params);

        // Drive traversal; without intersection functions Metal returns the committed result
        // after the first call to next().
        while (query.next()) {
            // No custom intersection functions, so nothing to do inside the loop.
        }

        if (query.get_committed_intersection_type() == intersection_type::triangle) {
            const uint primitiveID = query.get_committed_primitive_id();
            if (debugAlbedo) {
                colour = float3(1.0f, 0.0f, 0.0f);
            }
            
            const float2 bary = query.get_committed_triangle_barycentric_coord();
            const float w = clamp(1.0f - bary.x - bary.y, 0.0f, 1.0f);
            const float u = clamp(bary.x, 0.0f, 1.0f);
            const float v = clamp(bary.y, 0.0f, 1.0f);

            const float3 palette[3] = {
                float3(0.92f, 0.38f, 0.24f),
                float3(0.24f, 0.88f, 0.46f),
                float3(0.46f, 0.32f, 0.92f),
            };
            float3 albedo = palette[0] * w + palette[1] * u + palette[2] * v;
            float3 emission = float3(0.0f);

            float3 normal = normalize(float3(0.0f, 1.0f, 0.0f));
            uint instanceIndex = 0;
            if (instances && resourceHeader != nullptr && resourceHeader->instanceCount > 0) {
                instanceIndex = min(query.get_committed_user_instance_id(), resourceHeader->instanceCount - 1u);
            }

            RTRRayTracingMeshResource mesh = {};
            RTRRayTracingInstanceResource instance = {};
            RTRRayTracingMaterial materialProps = {};
            materialProps.albedo = float3(1.0f);
            materialProps.textureIndex = RTR_INVALID_TEXTURE_INDEX;
            materialProps.materialFlags = 0u;
            bool haveInstanceData = false;
            float2 interpolatedUV = float2(0.0f);
            bool hasValidUV = false;
            if (meshResources != nullptr && resourceHeader != nullptr && resourceHeader->geometryCount > 0 && instances) {
                instance = instances[instanceIndex];
                haveInstanceData = true;
                const uint meshIndex = min(instance.meshIndex, resourceHeader->geometryCount - 1u);
                mesh = meshResources[meshIndex];

                const bool hasGPUAddresses = mesh.vertexBufferAddress != 0 && mesh.indexBufferAddress != 0;
                const bool hasFallbackVertices = mesh.fallbackVertexOffset != kInvalidOffset && fallbackVertexBytes;
                const bool hasFallbackIndices = mesh.fallbackIndexOffset != kInvalidOffset && fallbackIndices;

                const device uchar* vertexBytes = nullptr;
                const device uint* indexData = nullptr;
                if (hasGPUAddresses) {
                    vertexBytes = reinterpret_cast<const device uchar*>(mesh.vertexBufferAddress);
                    indexData = reinterpret_cast<const device uint*>(mesh.indexBufferAddress);
                } else {
                    if (hasFallbackVertices) {
                        vertexBytes = fallbackVertexBytes + mesh.fallbackVertexOffset;
                    }
                    if (hasFallbackIndices) {
                        indexData = fallbackIndices + mesh.fallbackIndexOffset;
                    }
                }

                if (vertexBytes && indexData && mesh.vertexStride > 0 && mesh.indexCount >= 3 && mesh.vertexCount >= 3) {
                    const uint base = min(primitiveID * 3u, mesh.indexCount - 3u);
                    const uint maxVertex = mesh.vertexCount - 1u;
                    const uint i0 = min(indexData[base + 0], maxVertex);
                    const uint i1 = min(indexData[base + 1], maxVertex);
                    const uint i2 = min(indexData[base + 2], maxVertex);

                    const RTRVertexSample v0 = loadVertexSample(vertexBytes, mesh.vertexStride, i0);
                    const RTRVertexSample v1 = loadVertexSample(vertexBytes, mesh.vertexStride, i1);
                    const RTRVertexSample v2 = loadVertexSample(vertexBytes, mesh.vertexStride, i2);
                    const float3 p0 = v0.position;
                    const float3 p1 = v1.position;
                    const float3 p2 = v2.position;
                    const float3 e1 = p1 - p0;
                    const float3 e2 = p2 - p0;
                    const float3 faceNormal = normalize(cross(e1, e2));
                    float3 interpolatedNormal = normalize(v0.normal * w + v1.normal * u + v2.normal * v);
                    if (!all(isfinite(interpolatedNormal)) || length(interpolatedNormal) < 1e-3f) {
                        interpolatedNormal = faceNormal;
                    }
                    if (all(isfinite(interpolatedNormal))) {
                        normal = interpolatedNormal;
                    } else if (all(isfinite(faceNormal))) {
                        normal = faceNormal;
                    }

                    const float2 uv0 = v0.texcoord;
                    const float2 uv1 = v1.texcoord;
                    const float2 uv2 = v2.texcoord;
                    float2 uvInterp = uv0 * w + uv1 * u + uv2 * v;
                    if (all(isfinite(uvInterp))) {
                        interpolatedUV = uvInterp;
                        hasValidUV = true;
                    }
                }

            const uint materialIndex = instance.materialIndex;
            if (materials && resourceHeader->materialCount > 0) {
                materialProps = materials[min(materialIndex, resourceHeader->materialCount - 1u)];
                albedo = materialProps.albedo;
                emission = materialProps.emission;
            }
        }

        const float distance = query.get_committed_distance();
        float3 worldNormal = normal;
            if (haveInstanceData) {
                const float3x3 objectToWorld3x3 = float3x3(instance.objectToWorld[0].xyz,
                                                           instance.objectToWorld[1].xyz,
                                                           instance.objectToWorld[2].xyz);
                worldNormal = normalize(objectToWorld3x3 * normal);
            }

            float3 baseAlbedo = clamp(albedo, 0.0f, 1.0f);
            if (materialProps.textureIndex != RTR_INVALID_TEXTURE_INDEX && hasValidUV) {
                const float4 textureSample = sampleTexture(materialProps.textureIndex,
                                                          textureInfos,
                                                          textureCount,
                                                          texturePixels,
                                                          interpolatedUV);
                baseAlbedo = clamp(textureSample.xyz, 0.0f, 1.0f);
            }
            float3 worldPosition = primaryRay.origin + primaryRay.direction * distance;
            const float3 V = normalize(-primaryRay.direction);

            const float3 lightPosition = float3(0.0f, 0.92f, -1.05f);
            const float3 lightRight = float3(0.25f, 0.0f, 0.0f);
            const float3 lightForward = float3(0.0f, 0.0f, -0.18f);

            const float2 randSample = pseudoRandom(gid, uniforms.frameIndex + 1u);
            const float3 samplePoint = lightPosition + lightRight * (randSample.x * 2.0f - 1.0f) +
                                      lightForward * (randSample.y * 2.0f - 1.0f);
            const float3 toSample = samplePoint - worldPosition;
            const float distanceToLight = length(toSample);
            float3 lightDir = toSample / max(distanceToLight, 1e-3f);

            float shadowFactor = 1.0f;
            if (!debugAlbedo) {
                const float3 shadowOrigin = worldPosition + worldNormal * 0.002f;
                if (traceShadowRay(scene, shadowOrigin, lightDir, distanceToLight - 0.01f)) {
                    shadowFactor = 0.0f;
                }
            }

            const float nDotL = clamp(dot(worldNormal, lightDir), 0.0f, 1.0f);
            const float nDotV = clamp(dot(worldNormal, V), 0.0f, 1.0f);
            float roughness = clamp(materialProps.roughness, 0.05f, 1.0f);
            const float alpha = roughness * roughness;
            const float3 F0 = mix(float3(0.04f), baseAlbedo, materialProps.metallic);
            const float3 H = normalize(lightDir + V);
            const float nDotH = max(dot(worldNormal, H), 0.0f);
            const float vDotH = max(dot(V, H), 0.0f);
            const float NDF = distributionGGX(nDotH, alpha);
            const float G = geometrySmith(nDotV, nDotL, roughness);
            const float3 F = fresnelSchlick(vDotH, F0);
            const float3 specular = (NDF * G * F) / max(4.0f * nDotV * nDotL + 1e-4f, 1e-4f);
            const float3 kS = F;
            const float3 kD = (float3(1.0f) - kS) * (1.0f - materialProps.metallic);
            const float3 diffuse = kD * baseAlbedo * (1.0f / 3.14159265f);

            const float3 lightColor = float3(12.0f, 11.5f, 11.0f) / (distanceToLight * distanceToLight + 1e-3f);
            const float3 direct = (diffuse + specular) * lightColor * nDotL * shadowFactor;

            float3 colourDirect = direct + emission;

            if (!debugAlbedo) {
                if (materialProps.reflectivity > 0.0f) {
                    const float3 reflectDir = normalize(reflect(primaryRay.direction.xyz, worldNormal));
                    const float reflectStrength = clamp(materialProps.reflectivity, 0.0f, 1.0f);
                    const float skyFactor = clamp(reflectDir.y * 0.5f + 0.5f, 0.0f, 1.0f);
                    const float3 reflectionTint = mix(float3(0.2f, 0.25f, 0.32f), float3(0.7f, 0.75f, 0.8f), skyFactor);
                    colourDirect = mix(colourDirect, reflectionTint, reflectStrength * 0.6f);
                }

                if (materialProps.indexOfRefraction > 1.01f) {
                    const float eta = 1.0f / clamp(materialProps.indexOfRefraction, 1.01f, 2.5f);
                    const float3 refractDir = refract(primaryRay.direction.xyz, worldNormal, eta);
                    if (all(isfinite(refractDir))) {
                        const float refractionWeight = clamp(1.0f - materialProps.roughness, 0.0f, 1.0f);
                        const float3 glassTint = float3(0.9f, 0.95f, 1.0f);
                        colourDirect = mix(colourDirect, glassTint, refractionWeight * 0.35f);
                    }
                }
            }

            if (debugAlbedo) {
                colour = float3(1.0f, 0.0f, 0.0f);
            } else {
                colour = accumulateColor(accumulation, gid, uniforms, colourDirect, accumulationEnabled);
            }
        } else {
            float3 skyColour = float3(0.1f, 0.12f, 0.2f);
            if (!debugAlbedo) {
                const float t = direction.y * 0.5f + 0.5f;
                const float3 skyTop = float3(0.45f, 0.55f, 0.85f);
                const float3 skyBottom = float3(0.1f, 0.12f, 0.2f);
                skyColour = mix(skyBottom, skyTop, t);
            }
            colour = debugAlbedo ? float3(0.0f, 1.0f, 0.0f)
                                 : accumulateColor(accumulation, gid, uniforms, skyColour, accumulationEnabled);
        }
    }

    float3 mappedColour = colour;
    if (!debugAlbedo) {
        mappedColour = colour / (colour + float3(1.0f));
        mappedColour = pow(clamp(mappedColour, 0.0f, 1.0f), float3(1.0f / 2.2f));
    }
    output.write(float4(mappedColour, 1.0f), gid);
}
#endif // RTR_HAS_RAYTRACING

kernel void mpsRayKernel(device MPSRayOriginMaskDirectionMaxDistance* rays [[buffer(0)]],
                         constant MPSCameraUniforms& uniforms [[buffer(1)]],
                         constant MPSSamplingUniforms& sampling [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const float2 jitter = jitterForPixel(gid, sampling);
    const float2 pixel = float2(gid) + 0.5f + jitter;
    const float2 ndc = (pixel / float2(uniforms.width, uniforms.height) - 0.5f) * 2.0f;
    const float3 eye = uniforms.eye.xyz;
    const float3 forward = uniforms.forward.xyz;
    const float3 right = uniforms.right.xyz;
    const float3 up = uniforms.up.xyz;
    const float3 target = eye + forward + right * (ndc.x * uniforms.imagePlaneHalfExtents.x) +
                          up * (ndc.y * uniforms.imagePlaneHalfExtents.y);
    const float3 direction = normalize(target - eye);

    const uint index = gid.y * uniforms.width + gid.x;
    rays[index].origin.x = eye.x;
    rays[index].origin.y = eye.y;
    rays[index].origin.z = eye.z;
    rays[index].direction.x = direction.x;
    rays[index].direction.y = direction.y;
    rays[index].direction.z = direction.z;
    rays[index].mask = 0xFFFFFFFFu;
    rays[index].maxDistance = FLT_MAX;
}

kernel void mpsShadeKernel(const device MPSIntersectionData* intersections [[buffer(0)]],
                           const device packed_float3* positions [[buffer(1)]],
                           const device uint* indices [[buffer(2)]],
                           const device packed_float3* colors [[buffer(3)]],
                           device float4* outRadiance [[buffer(4)]],
                           constant MPSCameraUniforms& uniforms [[buffer(5)]],
                           constant MPSSceneLimits& limits [[buffer(6)]],
                           device float4* debugBuffer [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= uniforms.width * uniforms.height) {
        return;
    }

    const bool isDebugPixel = (gid == (225 * uniforms.width + 153));

    const MPSIntersectionData isect = intersections[gid];
    float4 colour = float4(0.08f, 0.08f, 0.12f, 1.0f);

    if (isfinite(isect.distance) && isect.distance < FLT_MAX && isect.primitiveIndex != UINT_MAX &&
        isect.primitiveIndex < limits.primitiveCount && limits.vertexCount > 0 && limits.indexCount > 0) {
        const uint primitiveIndex = isect.primitiveIndex;
        const uint base = primitiveIndex * 3;
        if ((base + 2) < limits.indexCount) {
            const uint i0 = indices[base + 0];
            const uint i1 = indices[base + 1];
            const uint i2 = indices[base + 2];

            if (i0 < limits.vertexCount && i1 < limits.vertexCount && i2 < limits.vertexCount) {
                const float3 v0 = float3(positions[i0]);
                const float3 v1 = float3(positions[i1]);
                const float3 v2 = float3(positions[i2]);
                const float3 e1 = v1 - v0;
                const float3 e2 = v2 - v0;
                const float3 normal = normalize(cross(e1, e2));

                const float lightIntensity = max(0.0f, dot(normal, normalize(float3(0.2f, 0.8f, 0.6f))));
                const float shading = lightIntensity * 0.8f + 0.2f;

                const float u = isect.coordinates.x;
                const float v = isect.coordinates.y;
                const float w = 1.0f - u - v;

                const float3 fallbackPalette[3] = {
                    float3(0.85f, 0.4f, 0.25f),
                    float3(0.25f, 0.85f, 0.4f),
                    float3(0.4f, 0.25f, 0.85f),
                };
                const float3 c0 = (i0 < limits.colorCount) ? float3(colors[i0]) : fallbackPalette[0];
                const float3 c1 = (i1 < limits.colorCount) ? float3(colors[i1]) : fallbackPalette[1];
                const float3 c2 = (i2 < limits.colorCount) ? float3(colors[i2]) : fallbackPalette[2];
                const float3 interpolatedColor = c0 * w + c1 * u + c2 * v;
                const float3 hitColour = clamp(interpolatedColor * shading, 0.0f, 1.0f);
                colour = float4(hitColour, 1.0f);

                if (isDebugPixel) {
                    debugBuffer[0] = float4(u, v, w, 0.0);
                    debugBuffer[1] = float4(float(i0), float(i1), float(i2), 0.0);
                    debugBuffer[2] = float4(c0, 0.0);
                    debugBuffer[3] = float4(c1, 0.0);
                    debugBuffer[4] = float4(c2, 0.0);
                    debugBuffer[5] = float4(normal, 0.0);
                    debugBuffer[6] = float4(shading, 0.0, 0.0, 0.0);
                    debugBuffer[7] = float4(interpolatedColor, 0.0);
                }
            }
        }
    }

    outRadiance[gid] = colour;
}

kernel void mpsAccumulateKernel(device float4* accumulation [[buffer(0)]],
                                device float4* current [[buffer(1)]],
                                constant MPSAccumulationUniforms& uniforms [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    float4 sample = current[gid];
    if (uniforms.reset != 0) {
        accumulation[gid] = sample;
        return;
    }

    const float frameCount = static_cast<float>(uniforms.frameIndex);
    float4 accum = accumulation[gid];
    float4 blended = (accum * frameCount + sample) / (frameCount + 1.0f);
    accumulation[gid] = blended;
    current[gid] = blended;
}

struct RTRDisplayVertexOutput {
    float4 position [[position]];
    float2 texcoord;
};

vertex RTRDisplayVertexOutput RTRDisplayVertex(uint vertexId [[vertex_id]]) {
    const float2 positions[3] = {
        {-1.0f, -1.0f},
        { 3.0f, -1.0f},
        {-1.0f,  3.0f},
    };
    RTRDisplayVertexOutput output;
    output.position = float4(positions[vertexId], 0.0f, 1.0f);
    output.texcoord = float2((positions[vertexId].x + 1.0f) * 0.5f,
                             (positions[vertexId].y + 1.0f) * 0.5f);
    return output;
}

fragment float4 RTRDisplayFragment(RTRDisplayVertexOutput in [[stage_in]],
                                   texture2d<float, access::sample> source [[texture(0)]]) {
    constexpr sampler textureSampler(filter::linear,
                                     address::clamp_to_edge,
                                     coord::normalized);
    const float3 colour = clamp(source.sample(textureSampler, in.texcoord).xyz, 0.0f, 1.0f);
    return float4(colour, 1.0f);
}
