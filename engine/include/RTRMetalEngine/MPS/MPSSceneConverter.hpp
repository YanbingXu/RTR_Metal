#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <simd/simd.h>

namespace rtr::scene {
class Scene;
}

namespace rtr::rendering {

struct MPSMaterialProperties {
    vector_float3 albedo;
    float roughness;
    vector_float3 emission;
    float metallic;
    float reflectivity;
    float indexOfRefraction;
};

struct MPSInstanceRange {
    std::uint32_t vertexOffset = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexOffset = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t materialIndex = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t padding = 0;
    simd_float4x4 transform = matrix_identity_float4x4;
    simd_float4x4 inverseTransform = matrix_identity_float4x4;
};

struct MPSSceneData {
    std::vector<vector_float3> positions;
    std::vector<vector_float3> normals;
    std::vector<vector_float2> texcoords;
    std::vector<vector_float3> colors;
    std::vector<uint32_t> indices;
    std::vector<MPSMaterialProperties> materials;
    std::vector<uint32_t> primitiveMaterials;
    std::vector<MPSInstanceRange> instanceRanges;
};

MPSSceneData buildSceneData(const scene::Scene& scene,
                           vector_float3 defaultColor = {0.9f, 0.9f, 0.9f});

}  // namespace rtr::rendering
