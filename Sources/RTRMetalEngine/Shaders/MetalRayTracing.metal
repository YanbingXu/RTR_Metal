#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;

struct CameraUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 inverseViewMatrix;
    float4x4 inverseProjectionMatrix;
    float3 cameraPosition;
    float padding;
};

struct DirectionalLightUniform {
    float3 direction;
    float intensity;
    float3 color;
    float padding;
};

struct FrameUniforms {
    CameraUniforms camera;
    DirectionalLightUniform light;
    float2 resolution;
    float2 paddingResolution;
    uint frameIndex;
    uint maxBounces;
    uint sampleCount;
    uint padding;
};

struct MaterialGPU {
    float3 albedo;
    float padding0;
    float3 emission;
    float padding1;
    float roughness;
    float metallic;
    float2 padding2;
};

struct GeometryInfoGPU {
    uint64_t vertexBufferAddress;
    uint64_t normalBufferAddress;
    uint64_t indexBufferAddress;
    uint materialIndex;
    uint primitiveCount;
    uint padding;
};

struct InstanceInfoGPU {
    float4x4 transform;
    float3x3 normalMatrix;
    uint materialIndex;
    float3 padding;
};

constant uint kRayMask = 0xFF;
constant float kPi = 3.14159265358979323846;

float3 sampleCameraDirection(uint2 pixel, uint2 resolution, constant CameraUniforms &camera, uint frameIndex) {
    float2 ndc = ((float2(pixel) + 0.5) / float2(resolution)) * 2.0 - 1.0;
    ndc.y *= -1.0;
    float4 clip = float4(ndc, 0.0, 1.0);
    float4 view = camera.inverseProjectionMatrix * clip;
    view.z = -1.0;
    view.w = 0.0;
    float3 world = (camera.inverseViewMatrix * view).xyz;
    return normalize(world);
}

float3 shadeDirectional(float3 normal,
                        float3 viewDir,
                        MaterialGPU material,
                        float3 lightDir,
                        float3 lightColor,
                        float intensity) {
    float NdotL = saturate(dot(normal, -lightDir));
    float3 diffuse = material.albedo * (1.0f / kPi);
    float3 specular = float3(0.0);
    float3 halfVec = normalize(-lightDir + viewDir);
    float NdotH = saturate(dot(normal, halfVec));
    float roughness = max(material.roughness, 0.05);
    float specPower = pow(max(NdotH, 0.0), max(1.0, 2.0 / (roughness * roughness)));
    specular = mix(float3(0.04), material.albedo, material.metallic) * specPower;
    return (diffuse + specular) * NdotL * intensity * lightColor + material.emission;
}

[[kernel]]
void rayGenMain(uint2 threadPosition [[thread_position_in_grid]],
                constant FrameUniforms &uniforms [[buffer(0)]],
                const device MaterialGPU *materials [[buffer(1)]],
                const device GeometryInfoGPU *geometryInfo [[buffer(2)]],
                const device InstanceInfoGPU *instanceInfo [[buffer(3)]],
                texture2d<float, access::write> renderTarget [[texture(0)]]) {
    float2 resolution = uniforms.resolution;
    if (threadPosition.x >= uint(resolution.x) || threadPosition.y >= uint(resolution.y)) {
        return;
    }

    float3 resultColor = float3(1.0, 0.0, 0.0);
    renderTarget.write(float4(resultColor, 1.0), threadPosition);
}
