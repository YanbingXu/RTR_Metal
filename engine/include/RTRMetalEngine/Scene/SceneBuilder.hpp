#pragma once

#include <span>
#include <vector>

#include <simd/simd.h>

#include "RTRMetalEngine/Scene/Mesh.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/Vertex.hpp"

namespace rtr::scene {

class SceneBuilder {
public:
    explicit SceneBuilder(Scene& scene) : scene_(scene) {}

    MeshHandle addTriangleMesh(std::span<const simd_float3> positions, std::span<const std::uint32_t> indices);
    MeshHandle addTriangleMesh(std::span<const Vertex> vertices, std::span<const std::uint32_t> indices);
    MaterialHandle addDefaultMaterial();

private:
    Scene& scene_;
};

}  // namespace rtr::scene
