#pragma once

#include <cstdint>
#include <vector>

#include <simd/simd.h>

#include "RTRMetalEngine/Core/Math.hpp"
#include "RTRMetalEngine/Scene/Vertex.hpp"

namespace rtr::scene {

class Mesh {
public:
    Mesh() = default;
    Mesh(std::vector<Vertex> vertices, std::vector<std::uint32_t> indices);

    [[nodiscard]] const std::vector<Vertex>& vertices() const noexcept { return vertices_; }
    [[nodiscard]] const std::vector<std::uint32_t>& indices() const noexcept { return indices_; }
    [[nodiscard]] const core::math::BoundingBox& bounds() const noexcept { return bounds_; }

private:
    void computeBounds() noexcept;

    std::vector<Vertex> vertices_;
    std::vector<std::uint32_t> indices_;
    core::math::BoundingBox bounds_ = core::math::BoundingBox::makeEmpty();
};

}  // namespace rtr::scene
