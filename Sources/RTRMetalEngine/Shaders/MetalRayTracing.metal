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
    uint2 resolution;
    uint2 paddingResolution;
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

struct RayPayload {
    float3 radiance;
    float3 attenuation;
    uint depth;
    uint active;
};

struct RayAttributes {
    float2 barycentrics;
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

RayPayload makeInitialPayload() {
    RayPayload payload;
    payload.radiance = float3(0.0);
    payload.attenuation = float3(1.0);
    payload.depth = 0;
    payload.active = 1;
    return payload;
}

float3 shadeDirectional(float3 normal,
                        float3 viewDir,
                        constant MaterialGPU &material,
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
                texture2d<float, access::write> renderTarget [[texture(0)]],
                metal::raytracing::acceleration_structure<metal::raytracing::instance> scene [[acceleration_structure(0)]]) {
    uint2 resolution = uniforms.resolution;
    if (threadPosition.x >= resolution.x || threadPosition.y >= resolution.y) {
        return;
    }

    RayPayload payload = makeInitialPayload();
    float3 origin = uniforms.camera.cameraPosition;
    float3 direction = sampleCameraDirection(threadPosition, resolution, uniforms.camera, uniforms.frameIndex);
    metal::raytracing::ray ray(origin, direction, 0.001, 10000.0);

    metal::raytracing::trace_ray(scene,
                                 ray,
                                 payload,
                                 kRayMask,
                                 metal::raytracing::ray_flags::none,
                                 /*max_level*/ uniforms.maxBounces,
                                 /*closest_hit_function*/ 0,
                                 /*any_hit_function*/ 0,
                                 /*miss_function*/ 0);

    renderTarget.write(float4(payload.radiance, 1.0), threadPosition);
}

[[visible]]
void missShader(thread RayPayload &payload [[payload]]) {
    if (payload.active == 0) {
        return;
    }
    payload.radiance += float3(0.1, 0.1, 0.2);
    payload.active = 0;
}

[[visible]]
void closestHitShader(thread RayPayload &payload [[payload]],
                      constant FrameUniforms &uniforms [[buffer(0)]],
                      const device MaterialGPU *materials [[buffer(1)]],
                      const device GeometryInfoGPU *geometryInfo [[buffer(2)]],
                      const device InstanceInfoGPU *instanceInfo [[buffer(3)]],
                      metal::raytracing::intersection_data<metal::raytracing::triangle_data> intersection [[intersection_data]],
                      uint primitiveIndex [[primitive_id]],
                      uint instanceIndex [[instance_id]]) {
    if (payload.active == 0) {
        return;
    }

    const GeometryInfoGPU geom = geometryInfo[instanceIndex];
    const InstanceInfoGPU inst = instanceInfo[instanceIndex];

    device const float3 *positions = reinterpret_cast<device const float3 *>(geom.vertexBufferAddress);
    device const float3 *normals = reinterpret_cast<device const float3 *>(geom.normalBufferAddress);
    device const uint *indices = reinterpret_cast<device const uint *>(geom.indexBufferAddress);

    uint baseIndex = primitiveIndex * 3;
    uint i0 = indices[baseIndex + 0];
    uint i1 = indices[baseIndex + 1];
    uint i2 = indices[baseIndex + 2];

    float2 bary = intersection.barycentric_coord;
    float3 localPosition = positions[i0] * (1.0 - bary.x - bary.y) + positions[i1] * bary.x + positions[i2] * bary.y;
    float3 localNormal = normals[i0] * (1.0 - bary.x - bary.y) + normals[i1] * bary.x + normals[i2] * bary.y;

    float4 worldPosition4 = inst.transform * float4(localPosition, 1.0);
    float3 worldPosition = worldPosition4.xyz / worldPosition4.w;
    float3 worldNormal = normalize(inst.normalMatrix * localNormal);

    const MaterialGPU material = *(materials + inst.materialIndex);

    float3 lightDir = normalize(uniforms.light.direction);
    float3 lightColor = uniforms.light.color;
    float intensity = uniforms.light.intensity;
    float3 viewDir = normalize(uniforms.camera.cameraPosition - worldPosition);

    float3 color = shadeDirectional(worldNormal, viewDir, material, lightDir, lightColor, intensity);
    payload.radiance += color * payload.attenuation;
    payload.active = 0;
}
