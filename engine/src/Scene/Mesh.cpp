#include "RTRMetalEngine/Scene/Mesh.hpp"

#include <utility>

namespace rtr::scene {

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<std::uint32_t> indices)
    : vertices_(std::move(vertices)), indices_(std::move(indices)) {
    computeBounds();
}

void Mesh::computeBounds() noexcept {
    bounds_ = core::math::BoundingBox::makeEmpty();
    if (vertices_.empty()) {
        return;
    }
    for (const Vertex& vertex : vertices_) {
        bounds_.expand(vertex.position);
    }
}

}  // namespace rtr::scene
