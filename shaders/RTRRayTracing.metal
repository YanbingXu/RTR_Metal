#include <metal_stdlib>
#include "MPSUniforms.metal"
using namespace metal;

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

kernel void mpsRayKernel(device MPSRayOriginMaskDirectionMaxDistance* rays [[buffer(0)]],
                         constant MPSCameraUniforms& uniforms [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const float2 pixel = float2(gid);
    const float2 ndc = (pixel / float2(uniforms.width, uniforms.height) - 0.5f) * 2.0f;
    const float3 target = uniforms.eye + uniforms.forward + uniforms.right * (ndc.x * uniforms.imagePlaneHalfExtents.x) +
                          uniforms.up * (ndc.y * uniforms.imagePlaneHalfExtents.y);
    const float3 direction = normalize(target - uniforms.eye);

    const uint index = gid.y * uniforms.width + gid.x;
    rays[index].origin = uniforms.eye;
    rays[index].direction = direction;
    rays[index].mask = 0xFFFFFFFF;
    rays[index].maxDistance = FLT_MAX;
}

kernel void mpsShadeKernel(const device MPSIntersectionData* intersections [[buffer(0)]],
                           const device float3* positions [[buffer(1)]],
                           const device uint* indices [[buffer(2)]],
                           const device float3* colors [[buffer(3)]],
                           device float4* outRadiance [[buffer(4)]],
                           constant MPSCameraUniforms& uniforms [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= uniforms.width * uniforms.height) {
        return;
    }

    const MPSIntersectionData isect = intersections[gid];
    float4 colour = float4(0.08f, 0.08f, 0.12f, 1.0f);

    if (isect.primitiveIndex != UINT_MAX) {
        const uint primitiveIndex = isect.primitiveIndex;
        const uint base = primitiveIndex * 3;
        float3 hitColour = {1.0f, 1.0f, 1.0f};
        if ((base + 2) < uniforms.width * uniforms.height) {
            const uint i0 = indices[base + 0];
            const uint i1 = indices[base + 1];
            const uint i2 = indices[base + 2];
            const float3 v0 = positions[i0];
            const float3 v1 = positions[i1];
            const float3 v2 = positions[i2];
            const float3 e1 = v1 - v0;
            const float3 e2 = v2 - v0;
            const float3 normal = normalize(cross(e1, e2));

            float intensity = max(0.0f, dot(normal, normalize(float3(0.2f, 0.8f, 0.6f))));
            intensity = intensity * 0.8f + 0.2f;

            const float u = isect.coordinates.x;
            const float v = isect.coordinates.y;
            const float w = 1.0f - u - v;

            const float3 c0 = colors[i0];
            const float3 c1 = colors[i1];
            const float3 c2 = colors[i2];
            hitColour = (c0 * w + c1 * u + c2 * v) * intensity;
        }
        colour = float4(clamp(hitColour, 0.0f, 1.0f), 1.0f);
    }

    outRadiance[gid] = colour;
}
