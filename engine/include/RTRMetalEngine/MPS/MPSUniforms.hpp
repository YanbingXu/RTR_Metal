#pragma once

#include <cstdint>

#include <simd/simd.h>

namespace rtr::rendering {

struct alignas(16) MPSCameraUniforms {
    simd_float4 eye{0.0f, 0.0f, 0.0f, 1.0f};
    simd_float4 forward{0.0f, 0.0f, -1.0f, 0.0f};
    simd_float4 right{1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 up{0.0f, 1.0f, 0.0f, 0.0f};
    simd_float2 imagePlaneHalfExtents{1.0f, 1.0f};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t padding[2] = {0, 0};
};

struct alignas(16) MPSSamplingUniforms {
    std::uint32_t sampleIndex = 0;
    std::uint32_t samplesPerPixel = 0;
    std::uint32_t baseSeed = 0;
    std::uint32_t padding = 0;
};

struct alignas(16) MPSIntersectionData {
    float distance = 0.0f;
    std::uint32_t primitiveIndex = 0;
    simd_float2 barycentric{0.0f, 0.0f};
};

struct MPSSceneLimits {
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t colorCount = 0;
    std::uint32_t primitiveCount = 0;
    std::uint32_t normalCount = 0;
    std::uint32_t texcoordCount = 0;
    std::uint32_t materialCount = 0;
    std::uint32_t textureCount = 0;
};

struct alignas(16) MPSAccumulationUniforms {
    std::uint32_t frameIndex = 0;
    std::uint32_t reset = 0;
    std::uint32_t padding[2] = {0, 0};
};

static_assert(sizeof(MPSCameraUniforms) % 16 == 0, "MPSCameraUniforms must be 16-byte aligned");
static_assert(sizeof(MPSIntersectionData) == sizeof(float) * 4, "MPSIntersectionData must match MPS layout");
static_assert(sizeof(MPSSceneLimits) % 16 == 0, "MPSSceneLimits must be 16-byte aligned");
static_assert(sizeof(MPSAccumulationUniforms) % 16 == 0, "MPSAccumulationUniforms must be 16-byte aligned");
static_assert(sizeof(MPSSamplingUniforms) % 16 == 0, "MPSSamplingUniforms must be 16-byte aligned");

}  // namespace rtr::rendering
