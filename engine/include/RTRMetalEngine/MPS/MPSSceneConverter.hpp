#pragma once

#include <cstdint>
#include <vector>

#include <simd/simd.h>

#include <simd/simd.h>

#include <vector>
#include <cstdint>

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

struct MPSSceneData {
    std::vector<vector_float3> positions;
    std::vector<vector_float3> colors;
    std::vector<uint32_t> indices;
    std::vector<MPSMaterialProperties> materials;
    std::vector<uint32_t> primitiveMaterials;
};

MPSSceneData buildSceneData(const scene::Scene& scene,
                           vector_float3 defaultColor = {0.9f, 0.9f, 0.9f});

}  // namespace rtr::rendering
