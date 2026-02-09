#pragma once

#include <simd/simd.h>
#include <string>

namespace rtr::scene {

struct Material {
    simd_float3 albedo{1.0F, 1.0F, 1.0F};
    float roughness = 0.5F;
    simd_float3 emission{0.0F, 0.0F, 0.0F};
    float metallic = 0.0F;
    float reflectivity = 0.0F;
    float indexOfRefraction = 1.0F;
    float padding[1] = {0.0F};
    std::string albedoTexturePath;
};

}  // namespace rtr::scene
