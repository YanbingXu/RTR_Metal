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
    uint instanceCount;
    uint materialCount;
    uint randomTextureWidth;
    uint randomTextureHeight;
    uint padding0;
    uint padding1;
    uint padding2;
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

typedef struct RTRRayTracingInstanceResource {
    float4x4 objectToWorld;
    float4x4 worldToObject;
    uint meshIndex;
    uint materialIndex;
    uint padding0;
    uint padding1;
} RTRRayTracingInstanceResource;

typedef struct RTRRayTracingMaterial {
    float3 albedo;
    float roughness;
    float3 emission;
    float metallic;
    float reflectivity;
    float indexOfRefraction;
    float padding[2];
} RTRRayTracingMaterial;

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
    std::uint32_t instanceCount = 0;
    std::uint32_t materialCount = 0;
    std::uint32_t randomTextureWidth = 0;
    std::uint32_t randomTextureHeight = 0;
    std::uint32_t padding0 = 0;
    std::uint32_t padding1 = 0;
    std::uint32_t padding2 = 0;
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

struct alignas(16) RayTracingInstanceResource {
    simd_float4x4 objectToWorld = matrix_identity_float4x4;
    simd_float4x4 worldToObject = matrix_identity_float4x4;
    std::uint32_t meshIndex = 0;
    std::uint32_t materialIndex = 0;
    std::uint32_t padding0 = 0;
    std::uint32_t padding1 = 0;
};

struct alignas(16) RayTracingMaterialResource {
    simd_float3 albedo{1.0F, 1.0F, 1.0F};
    float roughness = 0.5F;
    simd_float3 emission{0.0F, 0.0F, 0.0F};
    float metallic = 0.0F;
    float reflectivity = 0.0F;
    float indexOfRefraction = 1.5F;
    float padding[2] = {0.0F, 0.0F};
};

static_assert(sizeof(RayTracingResourceHeader) % 16 == 0, "RayTracingResourceHeader must be 16-byte aligned");
static_assert(sizeof(RayTracingMeshResource) % 16 == 0, "RayTracingMeshResource must be 16-byte aligned");
static_assert(sizeof(RayTracingInstanceResource) % 16 == 0, "RayTracingInstanceResource must be 16-byte aligned");
static_assert(sizeof(RayTracingMaterialResource) % 16 == 0, "RayTracingMaterialResource must be 16-byte aligned");

}  // namespace rtr::rendering

#endif
