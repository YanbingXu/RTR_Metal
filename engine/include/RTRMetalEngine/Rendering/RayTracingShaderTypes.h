#pragma once

#include <simd/simd.h>
#ifndef __METAL_VERSION__
#include <cstdint>
#include <limits>
#endif

#define RTR_RAY_FLAG_DEBUG 0x1u
#define RTR_RAY_FLAG_ACCUMULATE 0x2u

#define RTR_RAY_MASK_PRIMARY 0x3u
#define RTR_RAY_MASK_SHADOW 0x1u
#define RTR_RAY_MASK_SECONDARY 0x3u

#define RTR_TRIANGLE_MASK_GEOMETRY 0x1u
#define RTR_TRIANGLE_MASK_LIGHT 0x2u

#define RTR_MAX_AREA_LIGHTS 4u
#define RTR_INVALID_MATERIAL_INDEX 0xFFFFFFFFu

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
    uint flags;
    uint samplesPerPixel;
    uint sampleSeed;
} RTRRayTracingUniforms;

typedef struct RTRRayTracingResourceHeader {
    uint geometryCount;
    uint instanceCount;
    uint materialCount;
    uint textureCount;
    uint randomTextureWidth;
    uint randomTextureHeight;
    uint padding0;
    uint padding1;
} RTRRayTracingResourceHeader;

typedef struct RTRRayTracingMeshResource {
    ulong vertexBufferAddress;
    ulong indexBufferAddress;
    uint vertexCount;
    uint indexCount;
    uint vertexStride;
    uint materialIndex;
    uint padding0;
    uint padding1;
} RTRRayTracingMeshResource;

typedef struct RTRRayTracingInstanceResource {
    float4x4 objectToWorld;
    float4x4 worldToObject;
    uint meshIndex;
    uint materialIndex;
    uint padding0;
    uint padding1;
} RTRRayTracingInstanceResource;

#define RTR_INVALID_TEXTURE_INDEX 0xFFFFFFFFu

typedef struct RTRRayTracingTextureResource {
    uint width;
    uint height;
    uint rowPitch;
    uint dataOffset;
} RTRRayTracingTextureResource;

typedef struct RTRRayTracingMaterial {
    float3 albedo;
    float roughness;
    float3 emission;
    float metallic;
    float reflectivity;
    float indexOfRefraction;
    uint textureIndex;
    uint materialFlags;
} RTRRayTracingMaterial;

typedef struct RTRHardwareAreaLight {
    float4 position;
    float4 right;
    float4 up;
    float4 forward;
    float4 color;
} RTRHardwareAreaLight;

typedef struct RTRHardwareRayUniforms {
    RTRRayTracingUniforms camera;
    uint lightCount;
    uint maxBounces;
    uint padding0;
    uint padding1;
    RTRHardwareAreaLight lights[RTR_MAX_AREA_LIGHTS];
} RTRHardwareRayUniforms;

typedef struct RTRHardwareRay {
    packed_float3 origin;
    uint mask;
    packed_float3 direction;
    float maxDistance;
    float3 color;
    float padding0;
} RTRHardwareRay;

#else

namespace rtr::rendering {

inline constexpr std::uint32_t kInvalidTextureIndex = std::numeric_limits<std::uint32_t>::max();

struct RayTracingUniforms {
    simd_float4 eye{0.0F, 0.0F, 2.0F, 1.0F};
    simd_float4 forward{0.0F, 0.0F, -1.0F, 0.0F};
    simd_float4 right{1.0F, 0.0F, 0.0F, 0.0F};
    simd_float4 up{0.0F, 1.0F, 0.0F, 0.0F};
    simd_float2 imagePlaneHalfExtents{1.0F, 1.0F};
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t frameIndex = 0;
    uint32_t flags = 0;
    uint32_t samplesPerPixel = 1;
    uint32_t sampleSeed = 0;
};

struct alignas(16) RayTracingResourceHeader {
    std::uint32_t geometryCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t materialCount = 0;
    std::uint32_t textureCount = 0;
    std::uint32_t randomTextureWidth = 0;
    std::uint32_t randomTextureHeight = 0;
    std::uint32_t padding0 = 0;
    std::uint32_t padding1 = 0;
};

struct alignas(16) RayTracingMeshResource {
    std::uint64_t vertexBufferAddress = 0ULL;
    std::uint64_t indexBufferAddress = 0ULL;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t vertexStride = 0;
    std::uint32_t materialIndex = 0;
    std::uint32_t padding0 = 0;
    std::uint32_t padding1 = 0;
};

struct alignas(16) RayTracingInstanceResource {
    simd_float4x4 objectToWorld = matrix_identity_float4x4;
    simd_float4x4 worldToObject = matrix_identity_float4x4;
    std::uint32_t meshIndex = 0;
    std::uint32_t materialIndex = 0;
    std::uint32_t padding0 = 0;
    std::uint32_t padding1 = 0;
};

struct alignas(16) RayTracingTextureResource {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t rowPitch = 0;
    std::uint32_t dataOffset = 0;
};

struct alignas(16) RayTracingMaterialResource {
    simd_float3 albedo{1.0F, 1.0F, 1.0F};
    float roughness = 0.5F;
    simd_float3 emission{0.0F, 0.0F, 0.0F};
    float metallic = 0.0F;
    float reflectivity = 0.0F;
    float indexOfRefraction = 1.5F;
    std::uint32_t textureIndex = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t materialFlags = 0;
};

struct alignas(16) HardwareAreaLight {
    simd_float4 position{0.0F, 0.0F, 0.0F, 1.0F};
    simd_float4 right{1.0F, 0.0F, 0.0F, 0.0F};
    simd_float4 up{0.0F, 1.0F, 0.0F, 0.0F};
    simd_float4 forward{0.0F, 0.0F, -1.0F, 0.0F};
    simd_float4 color{1.0F, 1.0F, 1.0F, 0.0F};
};

struct alignas(16) HardwareRayUniforms {
    RayTracingUniforms camera;
    std::uint32_t lightCount = 0;
    std::uint32_t maxBounces = 1;
    std::uint32_t padding0 = 0;
    std::uint32_t padding1 = 0;
    HardwareAreaLight lights[RTR_MAX_AREA_LIGHTS] = {};
};

struct alignas(16) HardwareRay {
    float origin[3] = {0.0F, 0.0F, 0.0F};
    std::uint32_t mask = RTR_RAY_MASK_PRIMARY;
    float direction[3] = {0.0F, 0.0F, -1.0F};
    float maxDistance = 0.0F;
    simd_float3 color{1.0F, 1.0F, 1.0F};
    float padding0 = 0.0F;
};

static_assert(sizeof(RayTracingResourceHeader) % 16 == 0, "RayTracingResourceHeader must be 16-byte aligned");
static_assert(sizeof(RayTracingMeshResource) % 16 == 0, "RayTracingMeshResource must be 16-byte aligned");
static_assert(sizeof(RayTracingInstanceResource) % 16 == 0, "RayTracingInstanceResource must be 16-byte aligned");
static_assert(sizeof(RayTracingMaterialResource) % 16 == 0, "RayTracingMaterialResource must be 16-byte aligned");
static_assert(sizeof(RayTracingTextureResource) % 16 == 0, "RayTracingTextureResource must be 16-byte aligned");
static_assert(sizeof(HardwareAreaLight) % 16 == 0, "HardwareAreaLight must be 16-byte aligned");
static_assert(sizeof(HardwareRayUniforms) % 16 == 0, "HardwareRayUniforms must be 16-byte aligned");
static_assert(sizeof(HardwareRay) % 16 == 0, "HardwareRay must be 16-byte aligned");

}  // namespace rtr::rendering

#endif
