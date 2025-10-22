#pragma once

#include <simd/simd.h>

namespace rtr::scene {

struct Vertex {
    simd_float3 position{0.0F, 0.0F, 0.0F};
    simd_float3 normal{0.0F, 1.0F, 0.0F};
    simd_float2 texcoord{0.0F, 0.0F};
};

}  // namespace rtr::scene
