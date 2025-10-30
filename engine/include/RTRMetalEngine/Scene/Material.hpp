#pragma once

#include <simd/simd.h>

namespace rtr::scene {

struct Material {
    simd_float3 albedo{1.0F, 1.0F, 1.0F};
    float roughness = 0.5F;
    float metallic = 0.0F;
    float reflectivity = 0.0F;
    float indexOfRefraction = 1.5F;
    float padding[1] = {0.0F};
};

}  // namespace rtr::scene
