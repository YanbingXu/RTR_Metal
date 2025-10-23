#pragma once

#include <cstdint>

#include <simd/simd.h>

namespace rtr::rendering {

struct MPSCameraUniforms {
    simd_float3 eye{0.0f, 0.0f, 0.0f};
    float padding0 = 0.0f;
    simd_float3 forward{0.0f, 0.0f, -1.0f};
    float padding1 = 0.0f;
    simd_float3 right{1.0f, 0.0f, 0.0f};
    float padding2 = 0.0f;
    simd_float3 up{0.0f, 1.0f, 0.0f};
    float padding3 = 0.0f;
    simd_float2 imagePlaneHalfExtents{1.0f, 1.0f};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

static_assert(sizeof(MPSCameraUniforms) % 16 == 0, "MPSCameraUniforms must be 16-byte aligned");

struct MPSIntersectionData {
    float distance = 0.0f;
    std::uint32_t primitiveIndex = 0;
    simd_float2 barycentric{0.0f, 0.0f};
    float padding = 0.0f;
};

}  // namespace rtr::rendering
