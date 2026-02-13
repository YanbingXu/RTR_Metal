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
// Must stay in sync with kMaxTrianglesPerGeometry in AccelerationStructureBuilder.mm.
constant uint kTrianglesPerGeometryChunk = 4096u;
constant uint kDirectSamplesPerBounce = 4u;
constant float kPi = 3.14159265f;

inline uint mixBits(uint value) {
    value ^= value >> 16u;
    value *= 0x7FEB352Du;
    value ^= value >> 15u;
    value *= 0x846CA68Bu;
    value ^= value >> 16u;
    return value;
}

simd_float3 sampleSkyColor(float3 /*direction*/) {
    // Path-traced baseline: no artificial environment term.
    return float3(0.0f);
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

}  // namespace

#if RTR_HAS_RAYTRACING
using namespace metal::raytracing;

struct HardwareHit {
    bool hit;
    uint geometryIndex;
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
        hit.geometryIndex = result.geometry_id;
        hit.primitiveIndex = result.primitive_id;
        hit.distance = result.distance;
        hit.bary = result.triangle_barycentric_coord;
        hit.instanceIndex = result.instance_id;
        hit.instanceUserId = result.instance_id;
    }
    return hit;
}

inline uint resolveGlobalPrimitive(HardwareHit hit,
                                   RTRRayTracingInstanceResource instanceInfo,
                                   RTRRayTracingMeshResource meshResource) {
    const uint meshPrimitiveCount = max(meshResource.indexCount / 3u, 1u);
    const uint chunkOffset = hit.geometryIndex * kTrianglesPerGeometryChunk;
    const uint localPrimitive = min(chunkOffset + hit.primitiveIndex, meshPrimitiveCount - 1u);
    uint globalPrimitive = instanceInfo.primitiveOffset + localPrimitive;
    if (instanceInfo.primitiveCount > 0u) {
        const uint primitiveEnd = instanceInfo.primitiveOffset + instanceInfo.primitiveCount - 1u;
        globalPrimitive = min(globalPrimitive, primitiveEnd);
    }
    return globalPrimitive;
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

struct SurfaceSample {
    uint valid;
    uint instanceOutOfRange;
    uint rawInstanceId;
    uint globalPrimitive;
    uint meshIndex;
    float3 position;
    float3 normal;
    float3 baseColor;
    RTRRayTracingMaterial material;
};

inline float randomUnit(thread uint& state) {
    state = mixBits(state * 1664525u + 1013904223u);
    return (static_cast<float>(state & 0x00FFFFFFu) + 0.5f) * (1.0f / 16777216.0f);
}

inline float maxComponent(float3 value) {
    return max(value.x, max(value.y, value.z));
}

inline void buildOrthonormalBasis(float3 normal, thread float3& tangent, thread float3& bitangent) {
    const float3 helper = (abs(normal.z) < 0.999f) ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(helper, normal));
    bitangent = cross(normal, tangent);
}

inline float3 sampleCosineHemisphere(float3 normal, thread uint& rngState) {
    const float u1 = randomUnit(rngState);
    const float u2 = randomUnit(rngState);
    const float radius = sqrt(u1);
    const float phi = 6.28318530718f * u2;
    const float x = radius * cos(phi);
    const float y = radius * sin(phi);
    const float z = sqrt(max(0.0f, 1.0f - u1));
    float3 tangent;
    float3 bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

inline float3 sampleRoughSpecular(float3 reflectedDir, float roughness, thread uint& rngState) {
    const float3 roughSample = sampleCosineHemisphere(reflectedDir, rngState);
    const float alpha = clamp(roughness * roughness, 0.0f, 1.0f);
    return normalize(mix(reflectedDir, roughSample, alpha));
}

inline float3 evaluateOpaqueBRDF(float3 normal,
                                 float3 viewDir,
                                 float3 lightDir,
                                 float3 baseColor,
                                 RTRRayTracingMaterial material) {
    const float NdotL = max(dot(normal, lightDir), 0.0f);
    const float NdotV = max(dot(normal, viewDir), 0.0f);
    if (NdotL <= 1.0e-5f || NdotV <= 1.0e-5f) {
        return float3(0.0f);
    }

    float3 halfVec = normalize(lightDir + viewDir);
    if (!all(isfinite(halfVec))) {
        halfVec = normal;
    }
    const float NdotH = max(dot(normal, halfVec), 0.0f);
    const float VdotH = max(dot(viewDir, halfVec), 0.0f);
    const float alpha = max(material.roughness * material.roughness, 1.0e-3f);
    const float D = distributionGGX(NdotH, alpha);
    const float G = geometrySmith(NdotV, NdotL, material.roughness);
    const float3 F0 = mix(float3(0.04f), baseColor, material.metallic);
    const float3 F = fresnelSchlick(VdotH, F0);
    const float3 specular = (D * G * F) / max(4.0f * NdotV * NdotL, 1.0e-4f);
    const float3 kd = (float3(1.0f) - F) * (1.0f - material.metallic);
    const float3 diffuse = kd * baseColor * (1.0f / kPi);
    return diffuse + specular;
}

inline float powerHeuristic(float pdfA, float pdfB) {
    const float a2 = pdfA * pdfA;
    const float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, 1.0e-6f);
}

inline float opaqueSpecProbability(RTRRayTracingMaterial material) {
    return clamp(max(material.reflectivity, material.metallic), 0.0f, 0.98f);
}

inline float opaqueScatterPdf(float3 incidentDir,
                              float3 normal,
                              float3 outDir,
                              RTRRayTracingMaterial material) {
    const float diffusePdf = max(dot(normal, outDir), 0.0f) * (1.0f / kPi);
    const float3 reflected = normalize(reflect(incidentDir, normal));
    const float specPdf = max(dot(reflected, outDir), 0.0f) * (1.0f / kPi);
    const float specProb = opaqueSpecProbability(material);
    return mix(diffusePdf, specPdf, specProb);
}

inline float3 sampleDirectLighting(constant RTRHardwareRayUniforms& uniforms,
                                   acceleration_structure<instancing> accelerationStructure,
                                   float3 incidentDir,
                                   float3 position,
                                   float3 normal,
                                   float3 viewDir,
                                   float3 baseColor,
                                   RTRRayTracingMaterial material,
                                   thread uint& rngState) {
    if (uniforms.lightCount == 0u) {
        return float3(0.0f);
    }

    const uint lightIndex = min(static_cast<uint>(randomUnit(rngState) * static_cast<float>(uniforms.lightCount)),
                                uniforms.lightCount - 1u);
    const RTRHardwareAreaLight light = getAreaLight(uniforms, lightIndex);
    const float2 mapped = float2(randomUnit(rngState), randomUnit(rngState)) * 2.0f - 1.0f;
    const float3 samplePoint = light.position.xyz + light.right.xyz * mapped.x + light.up.xyz * mapped.y;
    const float3 toLight = samplePoint - position;
    const float lightDistance = length(toLight);
    if (lightDistance <= 1.0e-4f) {
        return float3(0.0f);
    }
    const float3 lightDir = toLight / lightDistance;
    if (!all(isfinite(lightDir))) {
        return float3(0.0f);
    }

    const float NdotL = max(dot(normal, lightDir), 0.0f);
    if (NdotL <= 1.0e-5f) {
        return float3(0.0f);
    }

    const float3 lightNormal = normalize(light.forward.xyz);
    const float cosAtLight = max(dot(-lightDir, lightNormal), 0.0f);
    if (cosAtLight <= 1.0e-5f) {
        return float3(0.0f);
    }

    const float lightArea = max(4.0f * length(cross(light.right.xyz, light.up.xyz)), 1.0e-6f);
    const float pdfLight = (lightDistance * lightDistance) /
                           max(cosAtLight * lightArea * static_cast<float>(uniforms.lightCount), 1.0e-6f);
    const float pdfBsdf = opaqueScatterPdf(incidentDir, normal, lightDir, material);
    const float misWeight = powerHeuristic(pdfLight, pdfBsdf);

    if (isOccluded(accelerationStructure, position, lightDir, lightDistance)) {
        return float3(0.0f);
    }

    // `light.color` is authored as area-light intensity in this scene setup.
    // Convert to emitted radiance to match the solid-angle PDF estimator.
    const float3 emittedRadiance = light.color.xyz / lightArea;
    const float3 brdf = evaluateOpaqueBRDF(normal, viewDir, lightDir, baseColor, material);
    return brdf * emittedRadiance * NdotL * (misWeight / max(pdfLight, 1.0e-6f));
}

inline void sampleOpaqueScatter(float3 incidentDir,
                                float3 normal,
                                float3 baseColor,
                                RTRRayTracingMaterial material,
                                thread uint& rngState,
                                thread float3& outDirection,
                                thread float3& outWeight) {
    const float specProb = opaqueSpecProbability(material);
    const float choice = randomUnit(rngState);
    if (specProb > 1.0e-4f && choice < specProb) {
        const float3 reflected = normalize(reflect(incidentDir, normal));
        outDirection = sampleRoughSpecular(reflected, material.roughness, rngState);
        const float3 specColor = clamp(mix(float3(0.04f), baseColor, material.metallic), 0.0f, 1.0f);
        outWeight = specColor / max(specProb, 1.0e-4f);
    } else {
        outDirection = sampleCosineHemisphere(normal, rngState);
        const float diffuseProb = max(1.0f - specProb, 1.0e-4f);
        const float3 diffuseColor = clamp(baseColor * (1.0f - material.metallic), 0.0f, 1.0f);
        outWeight = diffuseColor / diffuseProb;
    }
}

inline void sampleRefractiveScatter(float3 incidentDir,
                                    float3 normal,
                                    float3 baseColor,
                                    RTRRayTracingMaterial material,
                                    thread bool& inRefractiveMedium,
                                    thread uint& rngState,
                                    thread float3& outDirection,
                                    thread float3& outWeight) {
    const float ior = clamp(material.indexOfRefraction, 1.01f, 3.0f);
    const float3 orientedNormal = (dot(incidentDir, normal) < 0.0f) ? normal : -normal;
    const float etaI = inRefractiveMedium ? ior : 1.0f;
    const float etaT = inRefractiveMedium ? 1.0f : ior;
    const float eta = etaI / etaT;
    const float cosIncident = clamp(dot(-incidentDir, orientedNormal), 0.0f, 1.0f);
    const float f0Base = (etaT - etaI) / (etaT + etaI);
    const float f0 = f0Base * f0Base;
    const float fresnel = clamp(f0 + (1.0f - f0) * pow(1.0f - cosIncident, 5.0f), 0.0f, 1.0f);

    const bool sampleReflection = randomUnit(rngState) < fresnel;
    if (sampleReflection) {
        outDirection = normalize(reflect(incidentDir, orientedNormal));
        outWeight = float3(1.0f);
        return;
    }

    float3 refracted = refract(incidentDir, orientedNormal, eta);
    if (length_squared(refracted) < 1.0e-6f || !all(isfinite(refracted))) {
        outDirection = normalize(reflect(incidentDir, orientedNormal));
        outWeight = float3(1.0f);
    } else {
        outDirection = normalize(refracted);
        outWeight = clamp(baseColor, 0.0f, 1.0f);
        inRefractiveMedium = !inRefractiveMedium;
    }
}

inline SurfaceSample sampleSurfaceHit(HardwareHit hit,
                                      const device packed_float3* positions,
                                      const device packed_float3* normals,
                                      const device uint* indices,
                                      const device packed_float3* colors,
                                      const device packed_float2* texcoords,
                                      const device RTRRayTracingMeshResource* meshes,
                                      const device RTRRayTracingMaterial* materials,
                                      constant RTRRayTracingTextureResource* textureInfos,
                                      const device float* texturePixels,
                                      const device RTRRayTracingInstanceResource* instances,
                                      constant MPSSceneLimits& limits) {
    SurfaceSample out{};
    out.valid = 0u;
    out.instanceOutOfRange = 0u;
    out.rawInstanceId = 0u;
    out.globalPrimitive = 0u;
    out.meshIndex = 0u;
    out.position = float3(0.0f);
    out.normal = float3(0.0f, 1.0f, 0.0f);
    out.baseColor = float3(0.0f);
    out.material = makeFallbackMaterial();

    if (!hit.hit) {
        return out;
    }

    const bool hasInstanceData = (instances != nullptr && limits.instanceCount > 0u);
    out.rawInstanceId = hit.instanceUserId;
    const uint safeInstance = hasInstanceData ? min(out.rawInstanceId, limits.instanceCount - 1u) : 0u;
    out.instanceOutOfRange = (!hasInstanceData || out.rawInstanceId >= limits.instanceCount) ? 1u : 0u;

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

    out.meshIndex = instanceInfo.meshIndex;
    out.globalPrimitive = resolveGlobalPrimitive(hit, instanceInfo, meshResource);
    const uint base = out.globalPrimitive * 3u;
    if (indices == nullptr || base + 2u >= limits.indexCount) {
        return out;
    }

    const uint i0 = indices[base + 0u];
    const uint i1 = indices[base + 1u];
    const uint i2 = indices[base + 2u];
    if (i0 >= limits.vertexCount || i1 >= limits.vertexCount || i2 >= limits.vertexCount) {
        return out;
    }

    const float w = clamp(1.0f - hit.bary.x - hit.bary.y, 0.0f, 1.0f);
    const float3 baryFull = float3(w, hit.bary.x, hit.bary.y);
    const float3 localV0 = float3(positions[i0]);
    const float3 localV1 = float3(positions[i1]);
    const float3 localV2 = float3(positions[i2]);
    const float3 localHit = localV0 * baryFull.x + localV1 * baryFull.y + localV2 * baryFull.z;
    out.position = transformPosition(instanceInfo.objectToWorld, localHit);

    float2 uv = float2(0.0f);
    if (texcoords != nullptr &&
        i0 < limits.texcoordCount &&
        i1 < limits.texcoordCount &&
        i2 < limits.texcoordCount) {
        const float2 t0 = float2(texcoords[i0]);
        const float2 t1 = float2(texcoords[i1]);
        const float2 t2 = float2(texcoords[i2]);
        uv = t0 * baryFull.x + t1 * baryFull.y + t2 * baryFull.z;
    }

    float3 vertexColor = float3(1.0f);
    if (colors != nullptr &&
        i0 < limits.colorCount &&
        i1 < limits.colorCount &&
        i2 < limits.colorCount) {
        const float3 c0 = float3(colors[i0]);
        const float3 c1 = float3(colors[i1]);
        const float3 c2 = float3(colors[i2]);
        vertexColor = clamp(c0 * baryFull.x + c1 * baryFull.y + c2 * baryFull.z, 0.0f, 1.0f);
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
    out.normal = transformNormal(instanceInfo.worldToObject, localNormal);
    out.material = loadMaterial(instanceInfo.materialIndex,
                                materials,
                                limits.materialCount,
                                meshResource.materialIndex);
    const float3 sampledColor = sampleMaterialColor(out.material,
                                                    uv,
                                                    textureInfos,
                                                    limits.textureCount,
                                                    texturePixels);
    out.baseColor = clamp(sampledColor * vertexColor, 0.0f, 1.0f);
    out.valid = 1u;
    return out;
}

inline float3 integratePath(constant RTRHardwareRayUniforms& uniforms,
                            acceleration_structure<instancing> accelerationStructure,
                            const device packed_float3* positions,
                            const device packed_float3* normals,
                            const device uint* indices,
                            const device packed_float3* colors,
                            const device packed_float2* texcoords,
                            const device RTRRayTracingMeshResource* meshes,
                            const device RTRRayTracingMaterial* materials,
                            constant RTRRayTracingTextureResource* textureInfos,
                            const device float* texturePixels,
                            const device RTRRayTracingInstanceResource* instances,
                            constant MPSSceneLimits& limits,
                            float3 rayOrigin,
                            float3 rayDirection,
                            thread uint& rngState,
                            thread uint& outHitDebugValue) {
    float3 radiance = float3(0.0f);
    float3 throughput = float3(1.0f);
    bool inRefractiveMedium = false;
    outHitDebugValue = 0u;
    const uint maxBounces = max(1u, uniforms.maxBounces);

    for (uint bounce = 0u; bounce < maxBounces; ++bounce) {
        const uint rayMask = (bounce == 0u) ? RTR_RAY_MASK_PRIMARY : RTR_RAY_MASK_SECONDARY;
        const HardwareHit hit = traceScene(accelerationStructure, rayOrigin, rayDirection, 1.0e-3f, FLT_MAX, rayMask);
        if (!hit.hit) {
            radiance += throughput * sampleSkyColor(rayDirection);
            break;
        }

        const SurfaceSample surface = sampleSurfaceHit(hit,
                                                       positions,
                                                       normals,
                                                       indices,
                                                       colors,
                                                       texcoords,
                                                       meshes,
                                                       materials,
                                                       textureInfos,
                                                       texturePixels,
                                                       instances,
                                                       limits);
        if (surface.valid == 0u) {
            break;
        }

        if (bounce == 0u) {
            outHitDebugValue = 1u;
        }

        radiance += throughput * surface.material.emission;

        // Keep geometric normal orientation for refractive media boundary tests.
        const float3 geometricNormal = surface.normal;
        float3 shadingNormal = geometricNormal;
        if (dot(rayDirection, shadingNormal) > 0.0f) {
            shadingNormal = -shadingNormal;
        }
        const float3 viewDir = normalize(-rayDirection);
        const bool refractiveSurface = (surface.material.materialFlags & RTR_MATERIAL_FLAG_REFRACTIVE) != 0u;

        float3 nextDirection = rayDirection;
        float3 scatterWeight = float3(1.0f);

        if (refractiveSurface) {
            sampleRefractiveScatter(rayDirection,
                                    geometricNormal,
                                    surface.baseColor,
                                    surface.material,
                                    inRefractiveMedium,
                                    rngState,
                                    nextDirection,
                                    scatterWeight);
        } else {
            float3 directLighting = float3(0.0f);
            for (uint lightSample = 0u; lightSample < kDirectSamplesPerBounce; ++lightSample) {
                directLighting += sampleDirectLighting(uniforms,
                                                       accelerationStructure,
                                                       rayDirection,
                                                       surface.position,
                                                       shadingNormal,
                                                       viewDir,
                                                       surface.baseColor,
                                                       surface.material,
                                                       rngState);
            }
            radiance += throughput * (directLighting / static_cast<float>(kDirectSamplesPerBounce));
            sampleOpaqueScatter(rayDirection,
                                shadingNormal,
                                surface.baseColor,
                                surface.material,
                                rngState,
                                nextDirection,
                                scatterWeight);
        }

        throughput *= scatterWeight;
        if (!all(isfinite(throughput)) || maxComponent(throughput) < 1.0e-4f) {
            break;
        }

        if (bounce >= 2u) {
            const float survival = clamp(maxComponent(throughput), 0.05f, 0.95f);
            if (randomUnit(rngState) > survival) {
                break;
            }
            throughput /= survival;
        }

        rayDirection = normalize(nextDirection);
        rayOrigin = surface.position + rayDirection * 1.0e-3f;
    }

    return max(radiance, float3(0.0f));
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
                      texture2d<float, access::read> accumulationHistoryTex [[texture(1)]],
                      texture2d<float, access::write> accumulationOutTex [[texture(2)]],
                      texture2d<float, access::write> dstTex [[texture(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint width = uniforms.camera.width;
    const uint height = uniforms.camera.height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint hashX = mixBits(uniforms.camera.sampleSeed ^ (gid.x * 1973u + gid.y * 9277u + 0x9E3779B9u));
    uint hashY = mixBits(uniforms.camera.sampleSeed ^ (gid.y * 26699u + gid.x * 3187u + 0x85EBCA6Bu));
    if (randomTex.get_width() > 0 && randomTex.get_height() > 0) {
        const uint2 coord = uint2(hashX % randomTex.get_width(), hashY % randomTex.get_height());
        const uint4 noise = randomTex.read(coord);
        hashX = mixBits(hashX ^ noise.x ^ (uniforms.camera.frameIndex * 2246822519u));
        hashY = mixBits(hashY ^ noise.y ^ (uniforms.camera.frameIndex * 3266489917u));
    } else {
        hashX = mixBits(hashX ^ (uniforms.camera.frameIndex * 2246822519u));
        hashY = mixBits(hashY ^ (uniforms.camera.frameIndex * 3266489917u));
    }

    float2 jitter;
    jitter.x = (static_cast<float>(hashX & 0x00FFFFFFu) + 0.5f) * (1.0f / 16777216.0f) - 0.5f;
    jitter.y = (static_cast<float>(hashY & 0x00FFFFFFu) + 0.5f) * (1.0f / 16777216.0f) - 0.5f;

    const float2 dims = float2(static_cast<float>(max(width, 1u)), static_cast<float>(max(height, 1u)));
    float2 ndc = ((float2(gid) + 0.5f + jitter) / dims) * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    const float3 eye = uniforms.camera.eye.xyz;
    const float3 forward = uniforms.camera.forward.xyz;
    const float3 right = uniforms.camera.right.xyz;
    const float3 up = uniforms.camera.up.xyz;
    const float3 target = eye + forward + right * (ndc.x * uniforms.camera.imagePlaneHalfExtents.x) +
                          up * (ndc.y * uniforms.camera.imagePlaneHalfExtents.y);
    const float3 rayDirection = normalize(target - eye);
    const float3 rayOrigin = eye;

    const bool debugAlbedo = (uniforms.camera.flags & RTR_RAY_FLAG_DEBUG) != 0u;
    const bool debugInstanceColors = (uniforms.camera.flags & RTR_RAY_FLAG_INSTANCE_COLOR) != 0u;
    const bool debugInstanceTrace = (uniforms.camera.flags & RTR_RAY_FLAG_INSTANCE_TRACE) != 0u;
    const bool debugPrimitiveTrace = (uniforms.camera.flags & RTR_RAY_FLAG_PRIMITIVE_TRACE) != 0u;

    if (is_null_instance_acceleration_structure(accelerationStructure)) {
        dstTex.write(float4(1.0f, 0.0f, 1.0f, 1.0f), gid);
        writeHitDebug(gid, width, height, hitDebug, 3u);
        return;
    }

    const HardwareHit firstHit = traceScene(accelerationStructure, rayOrigin, rayDirection, 1.0e-3f, FLT_MAX);
    const SurfaceSample firstSurface = sampleSurfaceHit(firstHit,
                                                        positions,
                                                        normals,
                                                        indices,
                                                        colors,
                                                        texcoords,
                                                        meshes,
                                                        materials,
                                                        textureInfos,
                                                        texturePixels,
                                                        instances,
                                                        limits);

    float3 color = sampleSkyColor(rayDirection);
    uint hitDebugValue = 0u;

    if (debugPrimitiveTrace) {
        if (firstSurface.valid != 0u) {
            hitDebugValue = firstSurface.globalPrimitive + 1u;
            const uint p = hitDebugValue;
            color = float3(static_cast<float>((p * 13u) & 0xFFu) / 255.0f,
                           static_cast<float>((p * 29u) & 0xFFu) / 255.0f,
                           static_cast<float>((p * 53u) & 0xFFu) / 255.0f);
        }
    } else if (debugInstanceTrace) {
        if (firstSurface.valid != 0u) {
            const uint meshBits = min(firstSurface.meshIndex, 0xFFFFu);
            const uint instanceBits = min(firstSurface.rawInstanceId, 0xFFFFu);
            hitDebugValue = ((instanceBits + 1u) << 16) | (meshBits + 1u);
            if (firstSurface.instanceOutOfRange != 0u) {
                hitDebugValue |= 0x80000000u;
            }
            color = (firstSurface.instanceOutOfRange != 0u) ? float3(1.0f, 0.0f, 0.0f)
                                                            : debugInstanceColor(firstSurface.meshIndex);
        }
    } else if (debugInstanceColors) {
        if (firstSurface.valid != 0u) {
            hitDebugValue = 1u;
            color = debugInstanceColor(firstSurface.meshIndex);
        }
    } else if (debugAlbedo) {
        if (firstSurface.valid != 0u) {
            hitDebugValue = 1u;
            color = firstSurface.baseColor;
        }
    } else {
        uint rngState = mixBits(hashX ^ (hashY * 747796405u) ^ (uniforms.camera.frameIndex * 2891336453u));
        color = integratePath(uniforms,
                              accelerationStructure,
                              positions,
                              normals,
                              indices,
                              colors,
                              texcoords,
                              meshes,
                              materials,
                              textureInfos,
                              texturePixels,
                              instances,
                              limits,
                              rayOrigin,
                              rayDirection,
                              rngState,
                              hitDebugValue);
    }

    writeHitDebug(gid, width, height, hitDebug, hitDebugValue);

    const bool allowAccumulation = !debugAlbedo && !debugInstanceColors && !debugInstanceTrace && !debugPrimitiveTrace;
    if (allowAccumulation &&
        uniforms.camera.frameIndex > 0u &&
        accumulationHistoryTex.get_width() > gid.x &&
        accumulationHistoryTex.get_height() > gid.y) {
        const float4 previous = accumulationHistoryTex.read(gid);
        const float blend = 1.0f / static_cast<float>(uniforms.camera.frameIndex + 1u);
        color = mix(previous.xyz, color, blend);
    }
    if (allowAccumulation &&
        accumulationOutTex.get_width() > gid.x &&
        accumulationOutTex.get_height() > gid.y) {
        accumulationOutTex.write(float4(color, 1.0f), gid);
    }

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
