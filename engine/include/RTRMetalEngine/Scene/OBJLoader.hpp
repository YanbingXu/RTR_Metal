#pragma once

#include <filesystem>
#include <vector>

#include <simd/simd.h>

namespace rtr::scene {

bool loadOBJMesh(const std::filesystem::path& path,
                 std::vector<simd_float3>& outPositions,
                 std::vector<std::uint32_t>& outIndices);

}  // namespace rtr::scene
