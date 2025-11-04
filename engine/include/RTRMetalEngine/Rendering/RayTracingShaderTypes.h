#pragma once

#include <simd/simd.h>
#ifndef __METAL_VERSION__
#include <cstdint>
#endif

#ifdef __METAL_VERSION__

typedef struct RTRRayTracingUniforms {
    float4 eye;
    float4 forward;
    float4 right;
    float4 up;
    float2 imagePlaneHalfExtents;
    uint width;
    uint height;
    uint frameIndex;
    uint padding;
} RTRRayTracingUniforms;

#else

namespace rtr::rendering {

struct RayTracingUniforms {
    simd_float4 eye{0.0F, 0.0F, 2.0F, 1.0F};
    simd_float4 forward{0.0F, 0.0F, -1.0F, 0.0F};
    simd_float4 right{1.0F, 0.0F, 0.0F, 0.0F};
    simd_float4 up{0.0F, 1.0F, 0.0F, 0.0F};
    simd_float2 imagePlaneHalfExtents{1.0F, 1.0F};
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t frameIndex = 0;
    uint32_t padding = 0;
};

}  // namespace rtr::rendering

#endif
