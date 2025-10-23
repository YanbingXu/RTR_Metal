#pragma once

#include <simd/simd.h>

namespace rtr::rendering {

struct MPSMaterial {
    simd_float3 albedo{1.0f, 1.0f, 1.0f};
    float roughness = 0.5f;
    simd_float3 emission{0.0f, 0.0f, 0.0f};
    float metallic = 0.0f;
};

static_assert(sizeof(MPSMaterial) % 16 == 0, "MPSMaterial must be 16-byte aligned");

}  // namespace rtr::rendering

