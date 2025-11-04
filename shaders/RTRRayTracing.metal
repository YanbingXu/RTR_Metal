#include <metal_stdlib>
#include "MPSUniforms.metal"
#include "RTRMetalEngine/Rendering/RayTracingShaderTypes.h"

#if defined(MTL_ENABLE_RAYTRACING) && MTL_ENABLE_RAYTRACING && __has_include(<metal_raytracing>)
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

inline float3 loadVertexPosition(const device uchar* base, uint stride, uint index) {
    return *reinterpret_cast<const device float3*>(base + stride * index);
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

kernel void rtGradientKernel(constant RTRRayTracingUniforms& uniforms [[buffer(0)]],
                             device const RTRRayTracingResourceHeader* resourceHeader [[buffer(1)]],
                             device const RTRRayTracingMeshResource* meshResources [[buffer(2)]],
                             device const uchar* fallbackVertexBytes [[buffer(4)]],
                             device const uint* fallbackIndices [[buffer(5)]],
                             device const RTRRayTracingInstanceResource* instances [[buffer(6)]],
                             device const RTRRayTracingMaterial* materials [[buffer(7)]],
                             texture2d<float, access::write> output [[texture(0)]],
                             texture2d<float, access::write> accumulation [[texture(1)]],
                             texture2d<float, access::read> randomTex [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
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
    output.write(float4(colour, 1.0f), gid);

    if (accumulation.get_width() == uniforms.width && accumulation.get_height() == uniforms.height) {
        accumulation.write(float4(colour, 1.0f), gid);
    }
}

#if RTR_HAS_RAYTRACING
kernel void rtHardwareKernel(instance_acceleration_structure scene [[buffer(3)]],
                             constant RTRRayTracingUniforms& uniforms [[buffer(0)]],
                             device const RTRRayTracingResourceHeader* resourceHeader [[buffer(1)]],
                             device const RTRRayTracingMeshResource* meshResources [[buffer(2)]],
                             device const uchar* fallbackVertexBytes [[buffer(4)]],
                             device const uint* fallbackIndices [[buffer(5)]],
                             device const RTRRayTracingInstanceResource* instances [[buffer(6)]],
                             device const RTRRayTracingMaterial* materials [[buffer(7)]],
                             texture2d<float, access::write> output [[texture(0)]],
                             texture2d<float, access::write> accumulation [[texture(1)]],
                             texture2d<float, access::read> randomTex [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const float2 dims = float2(max(uniforms.width, 1u), max(uniforms.height, 1u));
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

    constexpr uint kInvalidSlot = 0xFFFFFFFFu;

    if (!is_null_instance_acceleration_structure(scene)) {
        intersection_query<triangle_data, instancing> query(primaryRay, scene);
        if (query.next()) {
            query.commit_triangle_intersection();

            const uint primitiveID = query.get_committed_primitive_id();
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
            bool haveInstanceData = false;
            if (meshResources != nullptr && resourceHeader != nullptr && resourceHeader->geometryCount > 0 && instances) {
                instance = instances[instanceIndex];
                haveInstanceData = true;
                const uint meshIndex = min(instance.meshIndex, resourceHeader->geometryCount - 1u);
                mesh = meshResources[meshIndex];
                
                const bool hasGPUAddresses = mesh.vertexBufferAddress != 0 && mesh.indexBufferAddress != 0;
                const bool useFallbackVertices = mesh.fallbackVertexSlot != kInvalidSlot && fallbackVertexBytes;
                const bool useFallbackIndices = mesh.fallbackIndexSlot != kInvalidSlot && fallbackIndices;

                const device uchar* vertexBytes = nullptr;
                const device uint* indexData = nullptr;
                if (hasGPUAddresses) {
                    vertexBytes = reinterpret_cast<const device uchar*>(mesh.vertexBufferAddress);
                    indexData = reinterpret_cast<const device uint*>(mesh.indexBufferAddress);
                } else {
                    if (useFallbackVertices && mesh.fallbackVertexSlot == 0) {
                        vertexBytes = fallbackVertexBytes;
                    }
                    if (useFallbackIndices && mesh.fallbackIndexSlot == 0) {
                        indexData = fallbackIndices;
                    }
                }

                if (vertexBytes && indexData && mesh.vertexStride > 0 && mesh.indexCount >= 3 && mesh.vertexCount >= 3) {
                    const uint base = min(primitiveID * 3u, mesh.indexCount - 3u);
                    const uint maxVertex = mesh.vertexCount - 1u;
                    const uint i0 = min(indexData[base + 0], maxVertex);
                    const uint i1 = min(indexData[base + 1], maxVertex);
                    const uint i2 = min(indexData[base + 2], maxVertex);

                    const float3 p0 = loadVertexPosition(vertexBytes, mesh.vertexStride, i0);
                    const float3 p1 = loadVertexPosition(vertexBytes, mesh.vertexStride, i1);
                    const float3 p2 = loadVertexPosition(vertexBytes, mesh.vertexStride, i2);
                    const float3 e1 = p1 - p0;
                    const float3 e2 = p2 - p0;
                    const float3 maybeNormal = normalize(cross(e1, e2));
                    if (all(isfinite(maybeNormal))) {
                        normal = maybeNormal;
                    }
                }

                const uint materialIndex = instance.materialIndex;
                if (materials && resourceHeader->materialCount > 0) {
                    const RTRRayTracingMaterial material = materials[min(materialIndex, resourceHeader->materialCount - 1u)];
                    albedo = material.albedo;
                    emission = material.emission;
                }
            }

            const float distance = query.get_committed_distance();
            const float attenuation = clamp(exp(-distance * 0.2f), 0.15f, 1.0f);
            float3 worldNormal = normal;
            if (haveInstanceData) {
                const float3x3 objectToWorld3x3 = float3x3(instance.objectToWorld[0].xyz,
                                                           instance.objectToWorld[1].xyz,
                                                           instance.objectToWorld[2].xyz);
                worldNormal = normalize(objectToWorld3x3 * normal);
            }

            float3 baseAlbedo = clamp(albedo, 0.0f, 1.0f);
            float3 worldPosition = primaryRay.origin + primaryRay.direction * distance;
            const float3 lightPosition = float3(0.0f, 0.95f, -1.0f);
            const float3 lightColor = float3(15.0f, 14.0f, 13.0f);
            const float3 toLight = lightPosition - worldPosition;
            const float lightDistanceSq = max(dot(toLight, toLight), 1e-3f);
            const float3 lightDir = normalize(toLight);
            const float nDotL = clamp(dot(worldNormal, lightDir), 0.0f, 1.0f);
            const float3 direct = (lightColor * nDotL) / lightDistanceSq;

            const float3 ambient = baseAlbedo * 0.05f;
            const float3 shading = baseAlbedo * direct * attenuation;
            colour = clamp(ambient + shading + emission, 0.0f, 1.0f);
        } else {
            const float t = direction.y * 0.5f + 0.5f;
            const float3 skyTop = float3(0.45f, 0.55f, 0.85f);
            const float3 skyBottom = float3(0.1f, 0.12f, 0.2f);
            colour = mix(skyBottom, skyTop, t);
        }
    }

    const uint noiseW = (resourceHeader != nullptr && resourceHeader->randomTextureWidth > 0)
                            ? resourceHeader->randomTextureWidth
                            : randomTex.get_width();
    const uint noiseH = (resourceHeader != nullptr && resourceHeader->randomTextureHeight > 0)
                            ? resourceHeader->randomTextureHeight
                            : randomTex.get_height();

    if (noiseW > 0 && noiseH > 0) {
        const uint2 noiseCoord = uint2(gid.x % noiseW, gid.y % noiseH);
        const float noise = randomTex.read(noiseCoord).x;
        colour *= clamp(0.8f + 0.2f * noise, 0.75f, 1.1f);
    }

    colour = clamp(colour, 0.0f, 1.0f);
    output.write(float4(colour, 1.0f), gid);

    if (accumulation.get_width() == uniforms.width && accumulation.get_height() == uniforms.height) {
        accumulation.write(float4(colour, 1.0f), gid);
    }
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
