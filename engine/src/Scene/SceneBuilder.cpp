#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace rtr::scene {

MeshHandle SceneBuilder::addTriangleMesh(std::span<const simd_float3> positions,
                                         std::span<const std::uint32_t> indices) {
    if (positions.empty()) {
        throw std::invalid_argument("Positions array cannot be empty");
    }
    if (indices.empty() || indices.size() % 3 != 0) {
        throw std::invalid_argument("Indices must contain complete triangles");
    }

    std::vector<Vertex> vertices;
    vertices.reserve(positions.size());
    for (const simd_float3& position : positions) {
        Vertex vertex{};
        vertex.position = position;
        vertices.push_back(vertex);
    }

    return scene_.addMesh(Mesh{std::move(vertices), std::vector<std::uint32_t>(indices.begin(), indices.end())});
}

MaterialHandle SceneBuilder::addDefaultMaterial() { return scene_.addMaterial(Material{}); }

}  // namespace rtr::scene
