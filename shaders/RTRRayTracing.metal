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

constant uint kHaltonPrimes[] = {
    2,   3,   5,   7,   11,  13,  17,  19,
    23,  29,  31,  37,  41,  43,  47,  53,
    59,  61,  67,  71,  73,  79,  83,  89,
};

inline uint mixBits(uint value) {
    value ^= value >> 17;
    value *= 0xed5ad4bbu;
    value ^= value >> 11;
    value *= 0xac4c1b51u;
    value ^= value >> 15;
    value *= 0x31848babu;
    value ^= value >> 14;
    return value;
}

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

inline float wrapCoordinate(float value) {
    value = value - floor(value);
    return (value < 0.0f) ? value + 1.0f : value;
}

inline float3 sampleSkyColor(float3 direction) {
    const float t = clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
    const float3 skyTop = float3(0.45f, 0.55f, 0.85f);
    const float3 skyBottom = float3(0.1f, 0.12f, 0.2f);
    return mix(skyBottom, skyTop, t);
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

#if RTR_HAS_RAYTRACING
inline RTRHardwareAreaLight defaultAreaLight();
inline RTRHardwareAreaLight getAreaLight(constant RTRHardwareRayUniforms& uniforms, uint index);
inline void sampleAreaLight(RTRHardwareAreaLight light,
                            float2 u,
                            float3 position,
                            thread float3& lightDir,
                            thread float3& lightColor,
                            thread float& lightDistance);
#endif

inline float2 jitterForPixel(uint2 gid, constant MPSSamplingUniforms& sampling) {
    const uint sampleIndex = (sampling.samplesPerPixel == 0)
                                 ? sampling.sampleIndex
                                 : min(sampling.sampleIndex, sampling.samplesPerPixel - 1u);
    if (sampleIndex == 0u && sampling.baseSeed == 0u) {
        return float2(0.0f);
    }
    const uint base = mixBits(gid.x ^ (gid.y << 16) ^ (sampling.baseSeed * 0x9e3779b9u) ^ sampleIndex);
    const uint hashX = mixBits(base ^ 0x68bc21ebu);
    const uint hashY = mixBits(base ^ 0x02e5be93u);
    const float jitterX = (static_cast<float>(hashX & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    const float jitterY = (static_cast<float>(hashY & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    return float2(jitterX, jitterY);
}

struct TraceHit {
    bool hit;
    uint primitiveIndex;
    float distance;
    float2 bary;
};

inline bool intersectTriangle(const float3 rayOrigin,
                              const float3 rayDir,
                              const float3 v0,
                              const float3 v1,
                              const float3 v2,
                              thread float& tOut,
                              thread float2& baryOut) {
    const float3 edge1 = v1 - v0;
    const float3 edge2 = v2 - v0;
    const float3 pvec = cross(rayDir, edge2);
    const float det = dot(edge1, pvec);
    const float epsilon = 1e-5f;
    if (fabs(det) < epsilon) {
        return false;
    }

    const float invDet = 1.0f / det;
    const float3 tvec = rayOrigin - v0;
    const float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const float3 qvec = cross(tvec, edge1);
    const float v = dot(rayDir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    const float t = dot(edge2, qvec) * invDet;
    if (t < epsilon) {
        return false;
    }

    tOut = t;
    baryOut = float2(u, v);
    return true;
}

inline TraceHit traceScene(const float3 rayOrigin,
                           const float3 rayDir,
                           const device packed_float3* positions,
                           const device uint* indices,
                           uint primitiveCount,
                           uint indexCount,
                           uint vertexCount) {
    TraceHit hit{};
    hit.hit = false;
    hit.distance = FLT_MAX;

    const uint maxIndex = min(indexCount, primitiveCount * 3u);
    for (uint base = 0; base + 2 < maxIndex; base += 3) {
        const uint i0 = indices[base + 0];
        const uint i1 = indices[base + 1];
        const uint i2 = indices[base + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
            continue;
        }

        const float3 v0 = float3(positions[i0]);
        const float3 v1 = float3(positions[i1]);
        const float3 v2 = float3(positions[i2]);
        float t = 0.0f;
        float2 bary = float2(0.0f);
        if (intersectTriangle(rayOrigin, rayDir, v0, v1, v2, t, bary) && t < hit.distance) {
            hit.hit = true;
            hit.distance = t;
            hit.bary = bary;
            hit.primitiveIndex = base / 3u;
        }
    }

    return hit;
}

inline RTRRayTracingMaterial loadMaterial(uint primitiveIndex,
                                          const device uint* primitiveMaterials,
                                          const device RTRRayTracingMaterial* materials,
                                          uint primitiveCount,
                                          uint materialCount) {
    RTRRayTracingMaterial material{};
    material.albedo = float3(0.8f);
    material.roughness = 0.5f;
    material.emission = float3(0.0f);
    material.metallic = 0.0f;
    material.reflectivity = 0.0f;
    material.indexOfRefraction = 1.0f;
    material.textureIndex = RTR_INVALID_TEXTURE_INDEX;
    material.materialFlags = 0u;

    if (primitiveMaterials != nullptr && primitiveIndex < primitiveCount) {
        const uint matIndex = primitiveMaterials[primitiveIndex];
        if (matIndex != RTR_INVALID_TEXTURE_INDEX && matIndex < materialCount && materials != nullptr) {
            material = materials[matIndex];
        }
    }
    return material;
}

inline float3 sampleMaterialColor(const RTRRayTracingMaterial material,
                                  const float2 uv,
                                  constant RTRRayTracingTextureResource* textureInfos,
                                  uint textureCount,
                                  const device float* texturePixels) {
    float3 materialColour = clamp(material.albedo, 0.0f, 1.0f);
    if (material.textureIndex != RTR_INVALID_TEXTURE_INDEX && textureInfos != nullptr && texturePixels != nullptr &&
        material.textureIndex < textureCount) {
        const float4 texSample = sampleTexture(material.textureIndex,
                                               textureInfos,
                                               textureCount,
                                               texturePixels,
                                               uv);
        materialColour = clamp(texSample.xyz, 0.0f, 1.0f);
    }
    return materialColour;
}

inline float3 computeNormal(uint i0,
                            uint i1,
                            uint i2,
                            float3 v0,
                            float3 v1,
                            float3 v2,
                            const device packed_float3* normals,
                            uint normalCount,
                            float3 barycentric) {
    float3 normal = normalize(cross(v1 - v0, v2 - v1));
    if (normalCount > 0 && normals != nullptr) {
        const float3 n0 = (i0 < normalCount) ? float3(normals[i0]) : normal;
        const float3 n1 = (i1 < normalCount) ? float3(normals[i1]) : normal;
        const float3 n2 = (i2 < normalCount) ? float3(normals[i2]) : normal;
        normal = normalize(n0 * barycentric.z + n1 * barycentric.x + n2 * barycentric.y);
    }
    return normal;
}

inline float3 shadeHit(uint primitiveIndex,
                       float2 bary,
                       float3 rayDir,
                       const device packed_float3* positions,
                       const device packed_float3* normals,
                       const device uint* indices,
                       const device packed_float3* colors,
                       const device packed_float2* texcoords,
                       const device RTRRayTracingMaterial* materials,
                       const device uint* primitiveMaterials,
                       constant RTRRayTracingTextureResource* textureInfos,
                       uint textureCount,
                       const device float* texturePixels,
                       constant MPSSceneLimits& limits) {
    const uint base = primitiveIndex * 3u;
    if (base + 2u >= limits.indexCount) {
        return float3(0.08f, 0.08f, 0.12f);
    }

    const uint i0 = indices[base + 0];
    const uint i1 = indices[base + 1];
    const uint i2 = indices[base + 2];
    if (i0 >= limits.vertexCount || i1 >= limits.vertexCount || i2 >= limits.vertexCount) {
        return float3(0.08f, 0.08f, 0.12f);
    }

    const float3 v0 = float3(positions[i0]);
    const float3 v1 = float3(positions[i1]);
    const float3 v2 = float3(positions[i2]);
    const float w = clamp(1.0f - bary.x - bary.y, 0.0f, 1.0f);
    const float3 baryFull = float3(w, bary.x, bary.y);
    const float3 normal = computeNormal(i0, i1, i2, v0, v1, v2, normals, limits.normalCount, baryFull);

    float2 uv = float2(0.0f);
    if (limits.texcoordCount > 0 && texcoords != nullptr) {
        const float2 uv0 = (i0 < limits.texcoordCount) ? float2(texcoords[i0]) : float2(0.0f);
        const float2 uv1 = (i1 < limits.texcoordCount) ? float2(texcoords[i1]) : float2(0.0f);
        const float2 uv2 = (i2 < limits.texcoordCount) ? float2(texcoords[i2]) : float2(0.0f);
        uv = uv0 * baryFull.x + uv1 * baryFull.y + uv2 * baryFull.z;
    }

    const float3 fallbackPalette[3] = {
        float3(0.85f, 0.4f, 0.25f),
        float3(0.25f, 0.85f, 0.4f),
        float3(0.4f, 0.25f, 0.85f),
    };
    const float3 c0 = (i0 < limits.colorCount && colors != nullptr) ? float3(colors[i0]) : fallbackPalette[0];
    const float3 c1 = (i1 < limits.colorCount && colors != nullptr) ? float3(colors[i1]) : fallbackPalette[1];
    const float3 c2 = (i2 < limits.colorCount && colors != nullptr) ? float3(colors[i2]) : fallbackPalette[2];
    const float3 vertexColour = clamp(c0 * baryFull.x + c1 * baryFull.y + c2 * baryFull.z, 0.0f, 1.0f);

    const RTRRayTracingMaterial material = loadMaterial(primitiveIndex,
                                                        primitiveMaterials,
                                                        materials,
                                                        limits.primitiveCount,
                                                        limits.materialCount);
    const float3 materialColour = sampleMaterialColor(material, uv, textureInfos, limits.textureCount, texturePixels);
    const float3 baseColour = clamp(materialColour * vertexColour, 0.0f, 1.0f);

    const float3 lightDir = normalize(float3(0.2f, 0.8f, 0.6f));
    const float nDotL = max(0.0f, dot(normal, lightDir));
    const float shading = nDotL * 0.8f + 0.2f;
    const float3 diffuse = clamp(baseColour * shading + material.emission, 0.0f, 1.0f);

    return diffuse;
}

inline float3 accumulateSpecularBounces(TraceHit startHit,
                                        float3 rayDir,
                                        float3 throughput,
                                        uint maxAdditionalBounces,
                                        const device packed_float3* positions,
                                        const device packed_float3* normals,
                                        const device uint* indices,
                                        const device packed_float3* colors,
                                        const device packed_float2* texcoords,
                                        const device RTRRayTracingMaterial* materials,
                                        const device uint* primitiveMaterials,
                                        constant RTRRayTracingTextureResource* textureInfos,
                                        uint textureCount,
                                        const device float* texturePixels,
                                        constant MPSSceneLimits& limits,
                                        RTRHardwareAreaLight light,
                                        uint randomSeed) {
    float3 accum = float3(0.0f);
    TraceHit currentHit = startHit;
    float3 currentDir = rayDir;
    float3 currentThroughput = throughput;

    for (uint bounce = 0; bounce < maxAdditionalBounces && currentHit.hit; ++bounce) {
        const float w = clamp(1.0f - currentHit.bary.x - currentHit.bary.y, 0.0f, 1.0f);
        const float3 baryFull = float3(w, currentHit.bary.x, currentHit.bary.y);
        const uint base = currentHit.primitiveIndex * 3u;
        const uint i0 = indices[base + 0];
        const uint i1 = indices[base + 1];
        const uint i2 = indices[base + 2];
        const float3 v0 = float3(positions[i0]);
        const float3 v1 = float3(positions[i1]);
        const float3 v2 = float3(positions[i2]);
        const float3 normal = computeNormal(i0, i1, i2, v0, v1, v2, normals, limits.normalCount, baryFull);
        const float3 hitPos = v0 * baryFull.x + v1 * baryFull.y + v2 * baryFull.z;
        const float3 viewDir = normalize(-currentDir);

        const RTRRayTracingMaterial material = loadMaterial(currentHit.primitiveIndex,
                                                            primitiveMaterials,
                                                            materials,
                                                            limits.primitiveCount,
                                                            limits.materialCount);

        float2 uv = float2(0.0f);
        if (limits.texcoordCount > 0 && texcoords != nullptr) {
            const float2 uv0 = (i0 < limits.texcoordCount) ? float2(texcoords[i0]) : float2(0.0f);
            const float2 uv1 = (i1 < limits.texcoordCount) ? float2(texcoords[i1]) : float2(0.0f);
            const float2 uv2 = (i2 < limits.texcoordCount) ? float2(texcoords[i2]) : float2(0.0f);
            uv = uv0 * baryFull.x + uv1 * baryFull.y + uv2 * baryFull.z;
        }

        float3 vertexColour = float3(1.0f);
        if (limits.colorCount > 0 && colors != nullptr) {
            const float3 c0 = float3(colors[i0]);
            const float3 c1 = float3(colors[i1]);
            const float3 c2 = float3(colors[i2]);
            vertexColour = clamp(c0 * baryFull.x + c1 * baryFull.y + c2 * baryFull.z, 0.0f, 1.0f);
        }

        const float3 materialColour = sampleMaterialColor(material, uv, textureInfos, textureCount, texturePixels);
        const float3 baseColour = clamp(materialColour * vertexColour, 0.0f, 1.0f);

        float3 lighting = currentThroughput * (material.emission + baseColour * 0.08f);
        lighting += currentThroughput * sampleSkyColor(currentDir) * 0.02f;

        uint seed = mixBits(randomSeed ^ bounce * 0x9e3779b9u ^ currentHit.primitiveIndex);
        const float2 lightSamples = float2(halton(seed, 2u), halton(seed ^ 0x85ebca6bu, 3u));
        float3 lightDir;
        float3 lightColor;
        float lightDistance;
        sampleAreaLight(light, lightSamples, hitPos, lightDir, lightColor, lightDistance);

        const float NdotL = max(dot(normal, lightDir), 0.0f);
        const float NdotV = max(dot(normal, viewDir), 0.0f);
        if (lightDistance > 0.0f && NdotL > 1.0e-4f && NdotV > 1.0e-4f && all(isfinite(lightDir))) {
            TraceHit shadowHit = traceScene(hitPos + normal * 1.0e-3f,
                                            lightDir,
                                            positions,
                                            indices,
                                            limits.primitiveCount,
                                            limits.indexCount,
                                            limits.vertexCount);
            const bool blocked = shadowHit.hit && shadowHit.distance < (lightDistance - 1.0e-3f);
            if (!blocked) {
                const float3 halfVec = normalize(lightDir + viewDir);
                const float NdotH = max(dot(normal, halfVec), 0.0f);
                const float VdotH = max(dot(viewDir, halfVec), 0.0f);
                const float alpha = max(material.roughness * material.roughness, 1.0e-3f);
                const float D = distributionGGX(NdotH, alpha);
                const float G = geometrySmith(NdotV, NdotL, material.roughness);
                const float3 F = fresnelSchlick(VdotH, mix(float3(0.04f), baseColour, material.metallic));
                const float3 specular = (D * G * F) / max(4.0f * NdotV * NdotL, 1.0e-4f);
                const float3 kd = (float3(1.0f) - F) * (1.0f - material.metallic);
                const float3 diffuse = kd * baseColour / 3.14159265f;
                lighting += currentThroughput * (diffuse + specular) * lightColor * NdotL;
            }
        }

        accum += clamp(lighting, 0.0f, 64.0f);

        const float reflectivity = clamp(material.reflectivity, 0.0f, 1.0f);
        const float metallic = clamp(material.metallic, 0.0f, 1.0f);
        const float ior = max(material.indexOfRefraction, 1.0f);
        const float specularWeight = clamp(reflectivity + metallic * 0.5f, 0.0f, 1.0f);

        float3 nextDir = float3(0.0f);
        bool hasNextRay = false;
        if (specularWeight > 0.0f) {
            nextDir = normalize(reflect(currentDir, normal));
            currentThroughput *= specularWeight;
            hasNextRay = true;
        } else if (ior > 1.0f) {
            const float eta = dot(currentDir, normal) < 0.0f ? (1.0f / ior) : ior;
            const float3 refractDir = refract(currentDir, normal, eta);
            if (all(isfinite(refractDir)) && length(refractDir) > 0.0f) {
                nextDir = normalize(refractDir);
                currentThroughput *= 0.8f;
                hasNextRay = true;
            }
        }

        if (!hasNextRay || all(currentThroughput < float3(1.0e-3f))) {
            break;
        }

        TraceHit bounceHit = traceScene(hitPos + normal * 1.0e-3f,
                                        nextDir,
                                        positions,
                                        indices,
                                        limits.primitiveCount,
                                        limits.indexCount,
                                        limits.vertexCount);
        if (!bounceHit.hit) {
            accum += currentThroughput * sampleSkyColor(nextDir);
            break;
        }

        currentDir = nextDir;
        currentHit = bounceHit;
    }

    return accum;
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
    const float invDistance = 1.0f / max(lightDistance, 1e-3f);
    lightDir = toSample * invDistance;
    const float3 normal = normalize(light.forward.xyz);
    lightColor = light.color.xyz * (invDistance * invDistance);
    lightColor *= clamp(dot(-lightDir, normal), 0.0f, 1.0f);
}

}  // namespace
#if RTR_HAS_RAYTRACING

kernel void rayKernel(constant RTRHardwareRayUniforms& uniforms [[buffer(0)]],
                      device RTRHardwareRay* rays [[buffer(1)]],
                      texture2d<unsigned int> randomTex [[texture(0)]],
                      texture2d<float, access::write> dstTex [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint width = uniforms.camera.width;
    const uint height = uniforms.camera.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint index = gid.y * width + gid.x;
    device RTRHardwareRay& ray = rays[index];

    uint jitterSeed = mixBits(gid.x * 73856093u ^ gid.y * 19349663u ^ (uniforms.camera.frameIndex * 1664525u));
    float2 jitter = float2(0.0f);
    if (randomTex.get_width() > 0 && randomTex.get_height() > 0) {
        const uint2 coord = uint2(gid.x % randomTex.get_width(), gid.y % randomTex.get_height());
        const uint4 noise = randomTex.read(coord);
        jitter.x = (static_cast<float>(noise.x & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
        jitter.y = (static_cast<float>(noise.y & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    } else {
        jitter.x = (static_cast<float>(jitterSeed & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
        jitterSeed = mixBits(jitterSeed ^ 0x85ebca6bu);
        jitter.y = (static_cast<float>(jitterSeed & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    }

    const float2 dims = float2(max(width, 1u), max(height, 1u));
    const float2 ndc = ((float2(gid) + 0.5f + jitter) / dims) * 2.0f - 1.0f;
    const float3 eye = uniforms.camera.eye.xyz;
    const float3 forward = uniforms.camera.forward.xyz;
    const float3 right = uniforms.camera.right.xyz;
    const float3 up = uniforms.camera.up.xyz;
    const float3 target = eye + forward + right * (ndc.x * uniforms.camera.imagePlaneHalfExtents.x) +
                          up * (ndc.y * uniforms.camera.imagePlaneHalfExtents.y);

    ray.origin = eye;
    ray.direction = normalize(target - eye);
    ray.mask = RTR_RAY_MASK_PRIMARY;
    ray.maxDistance = FLT_MAX;
    ray.color = float3(1.0f);

    dstTex.write(float4(0.0f), gid);
}

kernel void shadeKernel(constant RTRHardwareRayUniforms& uniforms [[buffer(0)]],
                        device RTRHardwareRay* rays [[buffer(1)]],
                        device RTRHardwareRay* shadowRays [[buffer(2)]],
                        const device MPSIntersectionData* intersections [[buffer(3)]],
                        const device packed_float3* positions [[buffer(4)]],
                        const device packed_float3* normals [[buffer(5)]],
                        const device uint* indices [[buffer(6)]],
                        const device packed_float3* colors [[buffer(7)]],
                        const device packed_float2* texcoords [[buffer(8)]],
                        const device uint* primitiveMaterials [[buffer(9)]],
                        const device RTRRayTracingMaterial* materials [[buffer(10)]],
                        constant RTRRayTracingTextureResource* textureInfos [[buffer(11)]],
                        const device float* texturePixels [[buffer(12)]],
                        constant MPSSceneLimits& limits [[buffer(13)]],
                        texture2d<unsigned int> randomTex [[texture(0)]],
                        texture2d<float, access::write> dstTex [[texture(1)]],
                        constant uint& bounce [[buffer(14)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint width = uniforms.camera.width;
    const uint height = uniforms.camera.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint index = gid.y * width + gid.x;
    const bool debugAlbedo = (uniforms.camera.flags & RTR_RAY_FLAG_DEBUG) != 0u;
    device RTRHardwareRay& ray = rays[index];
    device RTRHardwareRay& shadowRay = shadowRays[index];
    shadowRay.maxDistance = -1.0f;
    shadowRay.color = float3(0.0f);
    shadowRay.mask = RTR_RAY_MASK_SHADOW;

    if (ray.maxDistance < 0.0f) {
        return;
    }

    const MPSIntersectionData isect = intersections[index];
    const uint primitiveCount = (limits.primitiveCount > 0u) ? limits.primitiveCount : (limits.indexCount / 3u);
    const bool hasIntersection = isfinite(isect.distance) && isect.distance >= 0.0f &&
                                 isect.primitiveIndex != UINT_MAX && isect.primitiveIndex < primitiveCount;
    if (!hasIntersection) {
        ray.maxDistance = -1.0f;
        const float3 missColour = debugAlbedo ? float3(0.0f) : sampleSkyColor(normalize(ray.direction));
        dstTex.write(float4(clamp(missColour, 0.0f, 1.0f), 1.0f), gid);
        return;
    }

    const uint primitiveIndex = static_cast<uint>(isect.primitiveIndex);
    const uint base = primitiveIndex * 3u;
    if (limits.indexCount == 0u || base + 2u >= limits.indexCount) {
        ray.maxDistance = -1.0f;
        dstTex.write(float4(0.0f), gid);
        return;
    }

    const uint i0 = indices[base + 0];
    const uint i1 = indices[base + 1];
    const uint i2 = indices[base + 2];
    if (i0 >= limits.vertexCount || i1 >= limits.vertexCount || i2 >= limits.vertexCount) {
        ray.maxDistance = -1.0f;
        dstTex.write(float4(0.0f), gid);
        return;
    }

    const float2 bary = clamp(float2(isect.coordinates.x, isect.coordinates.y), 0.0f, 1.0f);
    const float3 baryFull = float3(clamp(1.0f - bary.x - bary.y, 0.0f, 1.0f), bary.x, bary.y);
    const float3 v0 = float3(positions[i0]);
    const float3 v1 = float3(positions[i1]);
    const float3 v2 = float3(positions[i2]);
    const float3 hitPos = v0 * baryFull.x + v1 * baryFull.y + v2 * baryFull.z;
    const float3 normal = computeNormal(i0, i1, i2, v0, v1, v2, normals, limits.normalCount, baryFull);
    const float3 rayDir = normalize(ray.direction);
    const float3 viewDir = normalize(-rayDir);
    const float3 offsetOrigin = hitPos + normal * 1.0e-3f;

    float2 uv = float2(0.0f);
    if (limits.texcoordCount > 0 && texcoords != nullptr) {
        const float2 uv0 = (i0 < limits.texcoordCount) ? float2(texcoords[i0]) : float2(0.0f);
        const float2 uv1 = (i1 < limits.texcoordCount) ? float2(texcoords[i1]) : float2(0.0f);
        const float2 uv2 = (i2 < limits.texcoordCount) ? float2(texcoords[i2]) : float2(0.0f);
        uv = uv0 * baryFull.x + uv1 * baryFull.y + uv2 * baryFull.z;
    }

    const RTRRayTracingMaterial material = loadMaterial(primitiveIndex,
                                                        primitiveMaterials,
                                                        materials,
                                                        limits.primitiveCount,
                                                        limits.materialCount);
    const float3 baseColour = clamp(sampleMaterialColor(material,
                                                        uv,
                                                        textureInfos,
                                                        limits.textureCount,
                                                        texturePixels),
                                    0.0f,
                                    1.0f);
    const float3 incomingThroughput = clamp(ray.color, 0.0f, 1.0f);

    float3 baseLighting = incomingThroughput * (baseColour * 0.08f) + incomingThroughput * material.emission;
    baseLighting += incomingThroughput * sampleSkyColor(rayDir) * 0.02f;

    if (!debugAlbedo) {
        const RTRHardwareAreaLight light = getAreaLight(uniforms);
        uint randomSeed = mixBits(gid.x * 73856093u ^ gid.y * 19349663u ^ uniforms.camera.frameIndex ^ bounce);
        const float2 lightSamples = float2(halton(randomSeed, 2u), halton(randomSeed ^ 0x9e3779b9u, 3u));
        float3 lightDir;
        float3 lightColor;
        float lightDistance;
        sampleAreaLight(light, lightSamples, hitPos, lightDir, lightColor, lightDistance);

        const float NdotL = max(dot(normal, lightDir), 0.0f);
        const float NdotV = max(dot(normal, viewDir), 0.0f);
        if (lightDistance > 0.0f && NdotL > 1.0e-4f && NdotV > 1.0e-4f && all(isfinite(lightDir))) {
            const float3 halfVec = normalize(lightDir + viewDir);
            const float NdotH = max(dot(normal, halfVec), 0.0f);
            const float VdotH = max(dot(viewDir, halfVec), 0.0f);
            const float alpha = max(material.roughness * material.roughness, 1.0e-3f);
            const float D = distributionGGX(NdotH, alpha);
            const float G = geometrySmith(NdotV, NdotL, material.roughness);
            const float3 F = fresnelSchlick(VdotH, mix(float3(0.04f), baseColour, material.metallic));
            const float3 specular = (D * G * F) / max(4.0f * NdotV * NdotL, 1.0e-4f);
            const float3 kd = (float3(1.0f) - F) * (1.0f - material.metallic);
            const float3 diffuse = kd * baseColour / 3.14159265f;
            const float3 directContribution = incomingThroughput * (diffuse + specular) * lightColor * NdotL;
            if (any(directContribution > float3(1.0e-4f))) {
                shadowRay.origin = packed_float3(offsetOrigin.x, offsetOrigin.y, offsetOrigin.z);
                shadowRay.direction = packed_float3(lightDir.x, lightDir.y, lightDir.z);
                shadowRay.maxDistance = max(lightDistance - 1.0e-3f, 0.0f);
                shadowRay.color = clamp(directContribution, 0.0f, 64.0f);
            }
        }
    }

    float3 secondaryLighting = float3(0.0f);
    if (!debugAlbedo && uniforms.maxBounces > 1u) {
        const float reflectivity = clamp(material.reflectivity, 0.0f, 1.0f);
        const float metallic = clamp(material.metallic, 0.0f, 1.0f);
        const float ior = max(material.indexOfRefraction, 1.0f);
        const float specularWeight = clamp(reflectivity + metallic * 0.5f, 0.0f, 1.0f);
        float3 nextDir = float3(0.0f);
        float3 secondaryThroughput = float3(0.0f);
        if (specularWeight > 0.0f) {
            nextDir = normalize(reflect(rayDir, normal));
            secondaryThroughput = incomingThroughput * specularWeight;
        } else if (ior > 1.0f) {
            const float eta = dot(rayDir, normal) < 0.0f ? (1.0f / ior) : ior;
            const float3 refractDir = refract(rayDir, normal, eta);
            if (all(isfinite(refractDir)) && dot(refractDir, refractDir) > 0.0f) {
                nextDir = normalize(refractDir);
                secondaryThroughput = incomingThroughput * 0.8f;
            }
        }
        if (secondaryThroughput.x > 1.0e-3f || secondaryThroughput.y > 1.0e-3f || secondaryThroughput.z > 1.0e-3f) {
            TraceHit secondaryHit = traceScene(offsetOrigin,
                                               nextDir,
                                               positions,
                                               indices,
                                               limits.primitiveCount,
                                               limits.indexCount,
                                               limits.vertexCount);
            if (secondaryHit.hit) {
                RTRHardwareAreaLight bounceLight = getAreaLight(uniforms);
                const uint bounceSeed = mixBits(gid.x * 73856093u ^ gid.y * 19349663u ^ bounce);
                secondaryLighting = accumulateSpecularBounces(secondaryHit,
                                                              nextDir,
                                                              secondaryThroughput,
                                                              uniforms.maxBounces - 1u,
                                                              positions,
                                                              normals,
                                                              indices,
                                                              colors,
                                                              texcoords,
                                                              materials,
                                                              primitiveMaterials,
                                                              textureInfos,
                                                              limits.textureCount,
                                                              texturePixels,
                                                              limits,
                                                              bounceLight,
                                                              bounceSeed);
            } else {
                secondaryLighting = secondaryThroughput * sampleSkyColor(nextDir);
            }
        }
    }

    const float3 outputColour = debugAlbedo ? clamp(baseColour, 0.0f, 1.0f)
                                             : clamp(baseLighting + secondaryLighting, 0.0f, 64.0f);
    dstTex.write(float4(outputColour, 1.0f), gid);
    ray.maxDistance = -1.0f;
}

kernel void shadowKernel(constant RTRHardwareRayUniforms& uniforms [[buffer(0)]],
                         device RTRHardwareRay* shadowRays [[buffer(1)]],
                         const device float* shadowIntersections [[buffer(2)]],
                         texture2d<float, access::read> srcTex [[texture(0)]],
                         texture2d<float, access::write> dstTex [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint width = uniforms.camera.width;
    const uint height = uniforms.camera.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint index = gid.y * width + gid.x;
    const device RTRHardwareRay& shadowRay = shadowRays[index];
    const float shadowHit = shadowIntersections[index];

    float3 color = srcTex.read(gid).xyz;
    const bool hasRequest = shadowRay.maxDistance >= 0.0f;
    const bool occluded = hasRequest && isfinite(shadowHit) && shadowHit > 1.0e-4f &&
                          (shadowHit + 1.0e-3f) < shadowRay.maxDistance;
    if (hasRequest && !occluded) {
        color += shadowRay.color;
    }

    dstTex.write(float4(color, 1.0f), gid);
}

kernel void accumulateKernel(constant RTRHardwareRayUniforms& uniforms [[buffer(0)]],
                             texture2d<float, access::read> renderTex [[texture(0)]],
                             texture2d<float, access::read> prevTex [[texture(1)]],
                             texture2d<float, access::write> accumTex [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    const uint width = uniforms.camera.width;
    const uint height = uniforms.camera.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    float3 color = renderTex.read(gid).xyz;
    if (uniforms.camera.frameIndex > 0u) {
        float3 prev = prevTex.read(gid).xyz * static_cast<float>(uniforms.camera.frameIndex);
        color = (color + prev) / static_cast<float>(uniforms.camera.frameIndex + 1u);
    }

    accumTex.write(float4(color, 1.0f), gid);
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
                           const device packed_float3* normals [[buffer(2)]],
                           const device uint* indices [[buffer(3)]],
                           const device packed_float3* colors [[buffer(4)]],
                           const device packed_float2* texcoords [[buffer(5)]],
                           const device RTRRayTracingMaterial* materials [[buffer(6)]],
                           const device uint* primitiveMaterials [[buffer(7)]],
                           constant RTRRayTracingTextureResource* textureInfos [[buffer(8)]],
                           const device float* texturePixels [[buffer(9)]],
                           device float4* outRadiance [[buffer(10)]],
                           constant MPSCameraUniforms& uniforms [[buffer(11)]],
                           constant MPSSceneLimits& limits [[buffer(12)]],
                           device float4* debugBuffer [[buffer(13)]],
                           const device MPSRayOriginMaskDirectionMaxDistance* rays [[buffer(14)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= uniforms.width * uniforms.height) {
        return;
    }

    const bool isDebugPixel = (gid == (225 * uniforms.width + 153));

    const MPSIntersectionData isect = intersections[gid];
    float4 colour = float4(0.08f, 0.08f, 0.12f, 1.0f);

    if (!(rays && limits.vertexCount > 0 && limits.indexCount > 0)) {
        outRadiance[gid] = colour;
        return;
    }

    float3 rayOrigin = float3(rays[gid].origin);
    float3 rayDir = normalize(float3(rays[gid].direction));

    if (!(isfinite(isect.distance) && isect.distance < FLT_MAX && isect.primitiveIndex != UINT_MAX &&
          isect.primitiveIndex < limits.primitiveCount)) {
        outRadiance[gid] = colour;
        return;
    }

    const uint maxBounces = 2u;
    float3 throughput = float3(1.0f);
    float3 accum = float3(0.0f);
    TraceHit currentHit{};
    currentHit.hit = true;
    currentHit.primitiveIndex = isect.primitiveIndex;
    currentHit.distance = isect.distance;
    currentHit.bary = clamp(float2(isect.coordinates.x, isect.coordinates.y), 0.0f, 1.0f);

    for (uint bounce = 0; bounce < maxBounces && currentHit.hit; ++bounce) {
        const float w = clamp(1.0f - currentHit.bary.x - currentHit.bary.y, 0.0f, 1.0f);
        const float3 baseColour = shadeHit(currentHit.primitiveIndex,
                                           currentHit.bary,
                                           rayDir,
                                           positions,
                                           normals,
                                           indices,
                                           colors,
                                           texcoords,
                                           materials,
                                           primitiveMaterials,
                                           textureInfos,
                                           limits.textureCount,
                                           texturePixels,
                                           limits);

        const RTRRayTracingMaterial material = loadMaterial(currentHit.primitiveIndex,
                                                            primitiveMaterials,
                                                            materials,
                                                            limits.primitiveCount,
                                                            limits.materialCount);
        accum += throughput * baseColour;

        // Compute normal for scattering
        const uint base = currentHit.primitiveIndex * 3u;
        const uint i0 = indices[base + 0];
        const uint i1 = indices[base + 1];
        const uint i2 = indices[base + 2];
        const float3 v0 = float3(positions[i0]);
        const float3 v1 = float3(positions[i1]);
        const float3 v2 = float3(positions[i2]);
        const float3 baryFull = float3(w, currentHit.bary.x, currentHit.bary.y);
        const float3 normal = computeNormal(i0, i1, i2, v0, v1, v2, normals, limits.normalCount, baryFull);
        const float3 hitPos = v0 * baryFull.x + v1 * baryFull.y + v2 * baryFull.z;

        // Determine reflection/refraction contributions
        const float reflectivity = clamp(material.reflectivity, 0.0f, 1.0f);
        const float metallic = clamp(material.metallic, 0.0f, 1.0f);
        const float ior = max(material.indexOfRefraction, 1.0f);
        const float specularWeight = clamp(reflectivity + metallic * 0.5f, 0.0f, 1.0f);

        float3 nextDir = float3(0.0f);
        float3 offsetOrigin = hitPos + normal * 1e-3f;
        bool hasNextRay = false;
        bool refractNext = false;

        if (specularWeight > 0.0f) {
            nextDir = normalize(reflect(rayDir, normal));
            throughput *= specularWeight;
            hasNextRay = true;
        } else if (ior > 1.0f) {
            const float eta = dot(rayDir, normal) < 0.0f ? (1.0f / ior) : ior;
            const float3 refractDir = refract(rayDir, normal, eta);
            if (all(isfinite(refractDir)) && length(refractDir) > 0.0f) {
                nextDir = normalize(refractDir);
                throughput *= 0.8f;
                hasNextRay = true;
                refractNext = true;
            }
        }

        if (!hasNextRay) {
            break;
        }

        TraceHit bounceHit = traceScene(offsetOrigin,
                                        nextDir,
                                        positions,
                                        indices,
                                        limits.primitiveCount,
                                        limits.indexCount,
                                        limits.vertexCount);
        if (!bounceHit.hit) {
            break;
        }

        rayOrigin = offsetOrigin;
        rayDir = nextDir;
        currentHit = bounceHit;
    }

    colour = float4(clamp(accum, 0.0f, 1.0f), 1.0f);

    if (isDebugPixel && debugBuffer != nullptr) {
        debugBuffer[0] = float4(rayDir, 0.0f);
        debugBuffer[1] = float4(accum, 0.0f);
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
