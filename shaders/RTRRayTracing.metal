#include <metal_stdlib>
#include "MPSUniforms.metal"
#include "RTRMetalEngine/Rendering/RayTracingShaderTypes.h"
using namespace metal;

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
    if (sampling.samplesPerPixel == 1 && sampleIndex == 0 && sampling.baseSeed == 0) {
        return float2(0.0f);
    }
    const uint base = mixBits(gid.x ^ (gid.y << 16) ^ (sampling.baseSeed * 0x9E3779B9U) ^ sampleIndex);
    const uint hashX = mixBits(base ^ 0x68bc21ebu);
    const uint hashY = mixBits(base ^ 0x02e5be93u);
    const float jitterX = (static_cast<float>(hashX & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    const float jitterY = (static_cast<float>(hashY & 0xFFFFu) + 0.5f) / 65536.0f - 0.5f;
    return float2(jitterX, jitterY);
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
                             device const uint* resourceBuffer [[buffer(1)]],
                             texture2d<float, access::write> output [[texture(0)]],
                             texture2d<float, access::write> accumulation [[texture(1)]],
                             texture2d<float, access::read> randomTex [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const float2 dims = float2(max(uniforms.width, 1u), max(uniforms.height, 1u));
    float2 uv = float2(gid) / dims;

    uint resourceCount = resourceBuffer ? resourceBuffer[0] : 0;
    float base = resourceCount > 0 ? 0.7f : 0.4f;

    const uint noiseWidth = randomTex.get_width();
    const uint noiseHeight = randomTex.get_height();
    float noise = 0.0f;
    if (noiseWidth > 0 && noiseHeight > 0) {
        const uint2 noiseCoord = uint2(gid.x % noiseWidth, gid.y % noiseHeight);
        noise = randomTex.read(noiseCoord).x;
    }

    float3 colour = float3(uv * (0.5f + 0.5f * base), 0.3f + 0.6f * sin((float)uniforms.frameIndex * 0.1f + noise));
    output.write(float4(colour, 1.0f), gid);

    if (accumulation.get_width() == uniforms.width && accumulation.get_height() == uniforms.height) {
        accumulation.write(float4(colour, 1.0f), gid);
    }
}

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
