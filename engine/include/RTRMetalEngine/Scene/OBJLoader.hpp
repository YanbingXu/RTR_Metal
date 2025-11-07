#pragma once

#include <filesystem>
#include <vector>

#include <simd/simd.h>

#include "RTRMetalEngine/Scene/Vertex.hpp"

namespace rtr::scene {

bool loadOBJMesh(const std::filesystem::path& path,
                 std::vector<Vertex>& outVertices,
                 std::vector<std::uint32_t>& outIndices);

}  // namespace rtr::scene
