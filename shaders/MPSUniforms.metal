#ifndef MPS_UNIFORMS_METAL
#define MPS_UNIFORMS_METAL

#include <metal_stdlib>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
using namespace metal;

typedef MPSIntersectionDistancePrimitiveIndexCoordinates MPSIntersectionData;

struct MPSCameraUniforms {
    float4 eye;
    float4 forward;
    float4 right;
    float4 up;
    float2 imagePlaneHalfExtents;
    uint width;
    uint height;
    uint padding0;
    uint padding1;
};

struct MPSMaterial {
    float3 albedo;
    float roughness;
    float3 emission;
    float metallic;
};

struct MPSSceneLimits {
    uint vertexCount;
    uint indexCount;
    uint colorCount;
    uint primitiveCount;
    uint normalCount;
    uint texcoordCount;
    uint materialCount;
    uint textureCount;
    uint instanceCount;
    uint meshCount;
    uint padding0;
    uint padding1;
};

struct MPSAccumulationUniforms {
    uint frameIndex;
    uint reset;
    uint padding0;
    uint padding1;
};

struct MPSSamplingUniforms {
    uint sampleIndex;
    uint samplesPerPixel;
    uint baseSeed;
    uint padding;
};

#endif
