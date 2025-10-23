#ifndef MPS_UNIFORMS_METAL
#define MPS_UNIFORMS_METAL

#include <metal_stdlib>
using namespace metal;

struct MPSCameraUniforms {
    float3 eye;
    float padding0;
    float3 forward;
    float padding1;
    float3 right;
    float padding2;
    float3 up;
    float padding3;
    float2 imagePlaneHalfExtents;
    uint width;
    uint height;
};

struct MPSRayOriginMaskDirectionMaxDistance {
    float3 origin;
    uint mask;
    float3 direction;
    float maxDistance;
};

struct MPSMaterial {
    float3 albedo;
    float roughness;
    float3 emission;
    float metallic;
};

struct MPSIntersectionData {
    float distance;
    uint primitiveIndex;
    float2 coordinates;
    float padding;
};

#endif
