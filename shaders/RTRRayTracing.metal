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

constant uint kHaltonPrimes[] = {
    2,   3,   5,   7,   11,  13,  17,  19,
    23,  29,  31,  37,  41,  43,  47,  53,
    59,  61,  67,  71,  73,  79,  83,  89
};

constant uint kMaxRayBounces = 3;
constant uint kInvalidBufferOffset = 0xFFFFFFFFu;

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

inline float halton(uint index, uint dimension) {
    const uint primeCount = static_cast<uint>(sizeof(kHaltonPrimes) / sizeof(uint));
    const uint primeIndex = dimension % max(primeCount, 1u);
    const uint base = kHaltonPrimes[primeIndex];
    float invBase = 1.0f / static_cast<float>(base);
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

inline uint randomSeedForPixel(uint2 gid,
                               texture2d<float, access::read> randomTex,
                               uint randomWidth,
                               uint randomHeight) {
    uint seed = mixBits(gid.x * 73856093u ^ gid.y * 19349663u);
    if (randomWidth > 0u && randomHeight > 0u &&
        randomTex.get_width() > 0 && randomTex.get_height() > 0) {
        const uint2 coord = uint2(gid.x % randomWidth, gid.y % randomHeight);
        const float4 noise = randomTex.read(coord);
        const uint nx = static_cast<uint>(clamp(noise.x, 0.0f, 1.0f) * 65535.0f);
        const uint ny = static_cast<uint>(clamp(noise.y, 0.0f, 1.0f) * 65535.0f);
        const uint nz = static_cast<uint>(clamp(noise.z, 0.0f, 1.0f) * 65535.0f);
        seed ^= mixBits((nx & 0xFFFFu) | ((ny & 0xFFFFu) << 16));
        seed ^= mixBits(nz);
    }
    return seed;
}

inline __attribute__((unused)) float2 pseudoRandom(uint2 gid, uint frameIndex) {
    const uint base = mixBits(gid.x * 73856093u ^ gid.y * 19349663u ^ ((frameIndex + 1u) * 83492791u));
    const uint hashX = mixBits(base ^ 0x9e3779b9u);
    const uint hashY = mixBits(base ^ 0x7f4a7c15u);
    const float rx = (static_cast<float>(hashX & 0xFFFFFFu) + 0.5f) / 16777216.0f;
    const float ry = (static_cast<float>(hashY & 0xFFFFFFu) + 0.5f) / 16777216.0f;
    return float2(rx, ry);
}

inline float3 sampleCosineWeightedHemisphere(float2 u) {
    const float phi = 2.0f * 3.14159265f * u.x;
    float cosPhi;
    const float sinPhi = sincos(phi, cosPhi);
    const float cosTheta = sqrt(u.y);
    const float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));
    return float3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
}

inline float3 alignHemisphereWithNormal(float3 sample, float3 normal) {
    const float3 up = normal;
    const float3 right = normalize(cross(up, float3(0.0072f, 1.0f, 0.0034f)));
    const float3 forward = cross(right, up);
    return sample.x * right + sample.y * up + sample.z * forward;
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

struct AreaLight {
    float3 position;
    float3 right;
    float3 forward;
    float3 normal;
    float3 color;
};

inline AreaLight makeCornellAreaLight() {
    AreaLight light;
    light.position = float3(0.0f, 0.92f, -1.05f);
    light.right = float3(0.25f, 0.0f, 0.0f);
    light.forward = float3(0.0f, 0.0f, -0.18f);
    light.normal = float3(0.0f, -1.0f, 0.0f);
    light.color = float3(18.0f, 17.5f, 17.0f);
    return light;
}

inline void sampleAreaLight(AreaLight light,
                            float2 u,
                            float3 position,
                            thread float3& lightDir,
                            thread float3& lightColor,
                            thread float& lightDistance) {
    const float2 mapped = u * 2.0f - 1.0f;
    const float3 samplePoint = light.position + light.right * mapped.x + light.forward * mapped.y;
    const float3 toSample = samplePoint - position;
    lightDistance = length(toSample);
    const float invDistance = 1.0f / max(lightDistance, 1e-3f);
    lightDir = toSample * invDistance;
    lightColor = light.color * (invDistance * invDistance);
    lightColor *= clamp(dot(-lightDir, light.normal), 0.0f, 1.0f);
}

inline float3 sampleSkyColor(float3 direction) {
    const float t = clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
    const float3 skyTop = float3(0.45f, 0.55f, 0.85f);
    const float3 skyBottom = float3(0.1f, 0.12f, 0.2f);
    return mix(skyBottom, skyTop, t);
}

inline __attribute__((unused)) float distributionGGX(float nDotH, float alpha) {
    const float a2 = alpha * alpha;
    const float denom = (nDotH * nDotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(3.14159265f * denom * denom, 1e-4f);
}

inline float geometrySchlickGGX(float nDotV, float k) {
    return nDotV / (nDotV * (1.0f - k) + k);
}

inline __attribute__((unused)) float geometrySmith(float nDotV, float nDotL, float roughness) {
    const float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    return geometrySchlickGGX(nDotV, k) * geometrySchlickGGX(nDotL, k);
}

inline __attribute__((unused)) float3 fresnelSchlick(float cosTheta, float3 F0) {
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

struct SurfaceSample {
    float3 worldPosition;
    float3 worldNormal;
    float3 baseAlbedo;
    float3 emission;
    RTRRayTracingMaterial material;
};

inline bool gatherSurfaceSample(instance_acceleration_structure scene,
                                ray traceRay,
                                device const RTRRayTracingResourceHeader* resourceHeader,
                                device const RTRRayTracingMeshResource* meshResources,
                                device const uchar* fallbackVertexBytes,
                                device const uint* fallbackIndices,
                                device const RTRRayTracingInstanceResource* instances,
                                device const RTRRayTracingMaterial* materials,
                                constant RTRRayTracingTextureResource* textureInfos,
                                uint textureCount,
                                device const float* texturePixels,
                                thread SurfaceSample& sampleOut) {
    if (is_null_instance_acceleration_structure(scene) || resourceHeader == nullptr || meshResources == nullptr ||
        instances == nullptr) {
        return false;
    }

    intersection_params params;
    params.assume_geometry_type(geometry_type::triangle);
    params.force_opacity(forced_opacity::opaque);
    intersection_query<triangle_data, instancing> query;
    query.reset(traceRay, scene, ~0u, params);
    while (query.next()) {}

    if (query.get_committed_intersection_type() != intersection_type::triangle) {
        return false;
    }

    const uint instanceCount = resourceHeader->instanceCount;
    const uint geometryCount = resourceHeader->geometryCount;
    if (instanceCount == 0u || geometryCount == 0u) {
        return false;
    }

    uint instanceIndex = min(query.get_committed_user_instance_id(), instanceCount - 1u);
    uint meshIndex = 0u;
    RTRRayTracingInstanceResource instance = {};
    if (instances != nullptr) {
        instance = instances[instanceIndex];
        meshIndex = min(instance.meshIndex, geometryCount - 1u);
    }

    RTRRayTracingMeshResource mesh = meshResources[meshIndex];
    const bool hasGPUAddresses = mesh.vertexBufferAddress != 0 && mesh.indexBufferAddress != 0;
    const bool hasFallbackVertices = mesh.fallbackVertexOffset != kInvalidBufferOffset && fallbackVertexBytes != nullptr;
    const bool hasFallbackIndices = mesh.fallbackIndexOffset != kInvalidBufferOffset && fallbackIndices != nullptr;

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

    if (!vertexBytes || !indexData || mesh.vertexStride == 0 || mesh.vertexCount < 3 || mesh.indexCount < 3) {
        return false;
    }

    const uint primitiveID = query.get_committed_primitive_id();
    const uint base = min(primitiveID * 3u, mesh.indexCount - 3u);
    const uint maxVertex = mesh.vertexCount - 1u;
    const uint i0 = min(indexData[base + 0], maxVertex);
    const uint i1 = min(indexData[base + 1], maxVertex);
    const uint i2 = min(indexData[base + 2], maxVertex);

    const RTRVertexSample v0 = loadVertexSample(vertexBytes, mesh.vertexStride, i0);
    const RTRVertexSample v1 = loadVertexSample(vertexBytes, mesh.vertexStride, i1);
    const RTRVertexSample v2 = loadVertexSample(vertexBytes, mesh.vertexStride, i2);

    const float2 bary = query.get_committed_triangle_barycentric_coord();
    const float w = clamp(1.0f - bary.x - bary.y, 0.0f, 1.0f);
    const float u = clamp(bary.x, 0.0f, 1.0f);
    const float v = clamp(bary.y, 0.0f, 1.0f);

    float3 objectNormal = normalize(v0.normal * w + v1.normal * u + v2.normal * v);
    if (!all(isfinite(objectNormal)) || length_squared(objectNormal) < 1e-4f) {
        const float3 e1 = v1.position - v0.position;
        const float3 e2 = v2.position - v0.position;
        objectNormal = normalize(cross(e1, e2));
    }

    float2 interpolatedUV = v0.texcoord * w + v1.texcoord * u + v2.texcoord * v;

    const float distance = query.get_committed_distance();
    const float3 worldPosition = traceRay.origin + traceRay.direction * distance;
    float3 worldNormal = objectNormal;
    if (instances != nullptr) {
        const float3x3 objectToWorld3x3(instance.objectToWorld[0].xyz,
                                        instance.objectToWorld[1].xyz,
                                        instance.objectToWorld[2].xyz);
        worldNormal = normalize(objectToWorld3x3 * objectNormal);
    }

    RTRRayTracingMaterial materialProps = {};
    materialProps.albedo = float3(0.75f);
    materialProps.roughness = 0.5f;
    materialProps.metallic = 0.0f;
    materialProps.reflectivity = 0.0f;
    materialProps.indexOfRefraction = 1.5f;

    const uint materialCount = resourceHeader->materialCount;
    if (materials != nullptr && materialCount > 0u) {
        uint materialIndex = mesh.materialIndex;
        if (instance.materialIndex < materialCount) {
            materialIndex = instance.materialIndex;
        }
        materialIndex = min(materialIndex, materialCount - 1u);
        materialProps = materials[materialIndex];
    }

    float3 baseAlbedo = clamp(materialProps.albedo, 0.0f, 1.0f);
    if (materialProps.textureIndex != RTR_INVALID_TEXTURE_INDEX && textureInfos != nullptr && texturePixels != nullptr) {
        const float4 texSample = sampleTexture(materialProps.textureIndex, textureInfos, textureCount, texturePixels,
                                              interpolatedUV);
        baseAlbedo = clamp(texSample.xyz, 0.0f, 1.0f);
    }

    sampleOut.worldPosition = worldPosition;
    sampleOut.worldNormal = worldNormal;
    sampleOut.baseAlbedo = baseAlbedo;
    sampleOut.emission = materialProps.emission;
    sampleOut.material = materialProps;
    return true;
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
    const uint textureCount = (resourceHeader != nullptr) ? resourceHeader->textureCount : 0u;
    const uint randomWidth = (resourceHeader != nullptr) ? resourceHeader->randomTextureWidth : 0u;
    const uint randomHeight = (resourceHeader != nullptr) ? resourceHeader->randomTextureHeight : 0u;
    const uint noiseWidth = (randomWidth > 0u) ? randomWidth : randomTex.get_width();
    const uint noiseHeight = (randomHeight > 0u) ? randomHeight : randomTex.get_height();
    const uint haltonSeed = randomSeedForPixel(gid, randomTex, noiseWidth, noiseHeight);
    const uint haltonIndex = haltonSeed + uniforms.frameIndex + 1u;
    const float2 jitter = float2(halton(haltonIndex, 0u), halton(haltonIndex, 1u));

    const float2 pixel = float2(gid) + jitter;
    const float2 ndc = (pixel / dims - 0.5f) * 2.0f;

    const float3 eye = uniforms.eye.xyz;
    const float3 forward = uniforms.forward.xyz;
    const float3 right = uniforms.right.xyz;
    const float3 up = uniforms.up.xyz;
    const float3 target = eye + forward +
                          right * (ndc.x * uniforms.imagePlaneHalfExtents.x) +
                          up * (ndc.y * uniforms.imagePlaneHalfExtents.y);
    const float3 direction = normalize(target - eye);

    const float rayMin = 0.001f;
    const float rayMax = 1.0e6f;
    ray currentRay(eye, direction, rayMin, rayMax);

    float3 radiance = float3(0.0f);
    float3 throughput = float3(1.0f);
    const AreaLight areaLight = makeCornellAreaLight();

    for (uint bounce = 0u; bounce < kMaxRayBounces; ++bounce) {
        SurfaceSample surface = {};
        const bool hit = gatherSurfaceSample(scene,
                                             currentRay,
                                             resourceHeader,
                                             meshResources,
                                             fallbackVertexBytes,
                                             fallbackIndices,
                                             instances,
                                             materials,
                                             textureInfos,
                                             textureCount,
                                             texturePixels,
                                             surface);
        if (!hit) {
            radiance += throughput * sampleSkyColor(currentRay.direction);
            break;
        }

        if (any(surface.emission > float3(0.0f))) {
            radiance += throughput * surface.emission;
            break;
        }

        const float diffuseWeight = clamp(1.0f - surface.material.reflectivity, 0.0f, 1.0f) *
                                    (1.0f - clamp(surface.material.metallic, 0.0f, 1.0f));
        const uint dimBase = 2u + bounce * 5u;
        const float3 bounceOrigin = surface.worldPosition + surface.worldNormal * 0.003f;

        if (diffuseWeight > 0.0f) {
            const float2 lightSample = float2(halton(haltonIndex, dimBase + 0u),
                                              halton(haltonIndex, dimBase + 1u));
            float3 lightDir;
            float3 lightColor;
            float lightDistance;
            sampleAreaLight(areaLight, lightSample, surface.worldPosition, lightDir, lightColor, lightDistance);
            const bool occluded = traceShadowRay(scene, bounceOrigin, lightDir, lightDistance - 0.01f);
            if (!occluded) {
                const float nDotL = clamp(dot(surface.worldNormal, lightDir), 0.0f, 1.0f);
                radiance += throughput * surface.baseAlbedo * diffuseWeight * lightColor * nDotL;
            }
        }

        float reflectionChance = clamp(surface.material.reflectivity, 0.0f, 1.0f);
        float refractionChance = (surface.material.indexOfRefraction > 1.01f)
                                     ? clamp(1.0f - surface.material.roughness, 0.0f, 1.0f) * 0.5f
                                     : 0.0f;
        float diffuseChance = max(1.0f - reflectionChance - refractionChance, 0.0f);
        float chanceSum = reflectionChance + refractionChance + diffuseChance;
        if (chanceSum <= 0.0f) {
            diffuseChance = 1.0f;
            chanceSum = 1.0f;
        }
        reflectionChance /= chanceSum;
        refractionChance /= chanceSum;
        diffuseChance = 1.0f - reflectionChance - refractionChance;

        const float lobeSample = halton(haltonIndex, dimBase + 4u);
        if (lobeSample < reflectionChance) {
            const float3 reflectDir = normalize(reflect(currentRay.direction, surface.worldNormal));
            throughput *= mix(float3(1.0f), surface.baseAlbedo, surface.material.metallic);
            currentRay = ray(bounceOrigin, reflectDir, rayMin, rayMax);
        } else if (lobeSample < reflectionChance + refractionChance) {
            float3 normal = surface.worldNormal;
            float eta = clamp(surface.material.indexOfRefraction, 1.01f, 2.5f);
            if (dot(currentRay.direction, surface.worldNormal) > 0.0f) {
                normal = -surface.worldNormal;
                eta = 1.0f / eta;
            }
            float3 refractDir = refract(currentRay.direction, normal, eta);
            if (!all(isfinite(refractDir)) || dot(refractDir, refractDir) < 1e-6f) {
                refractDir = reflect(currentRay.direction, surface.worldNormal);
            }
            throughput *= mix(float3(1.0f), surface.baseAlbedo, 0.2f);
            currentRay = ray(bounceOrigin, normalize(refractDir), rayMin, rayMax);
        } else {
            const float2 hemiSample = float2(halton(haltonIndex, dimBase + 2u),
                                             halton(haltonIndex, dimBase + 3u));
            float3 diffuseDir = sampleCosineWeightedHemisphere(hemiSample);
            diffuseDir = alignHemisphereWithNormal(diffuseDir, surface.worldNormal);
            throughput *= surface.baseAlbedo;
            currentRay = ray(bounceOrigin, normalize(diffuseDir), rayMin, rayMax);
        }

        if (all(throughput < float3(1e-3f))) {
            break;
        }
    }

    float3 colour = accumulateColor(accumulation, gid, uniforms, radiance, accumulationEnabled);
    float3 mappedColour = colour / (colour + float3(1.0f));
    mappedColour = pow(clamp(mappedColour, 0.0f, 1.0f), float3(1.0f / 2.2f));
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
