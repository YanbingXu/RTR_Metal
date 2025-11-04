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

typedef struct RTRRayTracingResourceHeader {
    uint geometryCount;
    uint materialCount;
    uint randomTextureWidth;
    uint randomTextureHeight;
} RTRRayTracingResourceHeader;

typedef struct RTRRayTracingMeshResource {
    ulong vertexBufferAddress;
    ulong indexBufferAddress;
    uint vertexCount;
    uint indexCount;
    uint vertexStride;
    uint materialIndex;
    uint fallbackVertexSlot;
    uint fallbackIndexSlot;
} RTRRayTracingMeshResource;

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

struct alignas(16) RayTracingResourceHeader {
    std::uint32_t geometryCount = 0;
    std::uint32_t materialCount = 0;
    std::uint32_t randomTextureWidth = 0;
    std::uint32_t randomTextureHeight = 0;
};

struct alignas(16) RayTracingMeshResource {
    std::uint64_t vertexBufferAddress = 0ULL;
    std::uint64_t indexBufferAddress = 0ULL;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t vertexStride = 0;
    std::uint32_t materialIndex = 0;
    std::uint32_t fallbackVertexSlot = 0;
    std::uint32_t fallbackIndexSlot = 0;
};

static_assert(sizeof(RayTracingResourceHeader) % 16 == 0, "RayTracingResourceHeader must be 16-byte aligned");
static_assert(sizeof(RayTracingMeshResource) % 16 == 0, "RayTracingMeshResource must be 16-byte aligned");

}  // namespace rtr::rendering

#endif
