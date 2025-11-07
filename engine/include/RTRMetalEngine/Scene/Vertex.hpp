#pragma once

#include <cstddef>

#include <simd/simd.h>

namespace rtr::scene {

struct Vertex {
    simd_float3 position{0.0F, 0.0F, 0.0F};
    simd_float3 normal{0.0F, 1.0F, 0.0F};
    simd_float2 texcoord{0.0F, 0.0F};
};

static_assert(offsetof(Vertex, position) == 0, "Vertex.position must be at offset 0");
static_assert(offsetof(Vertex, normal) == sizeof(simd_float3), "Vertex.normal layout mismatch");
static_assert(offsetof(Vertex, texcoord) == sizeof(simd_float3) * 2, "Vertex.texcoord layout mismatch");

}  // namespace rtr::scene
