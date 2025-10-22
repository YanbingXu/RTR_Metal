#pragma once

#include <cstdint>
#include <vector>

#include <simd/simd.h>

namespace rtr::scene {
class Scene;
}

namespace rtr::rendering {

struct MPSSceneData {
    std::vector<vector_float3> positions;
    std::vector<vector_float3> colors;
    std::vector<uint32_t> indices;
};

MPSSceneData buildSceneData(const scene::Scene& scene,
                           vector_float3 defaultColor = {0.9f, 0.9f, 0.9f});

}  // namespace rtr::rendering
