#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <simd/simd.h>

#include "RTRMetalEngine/Core/Math.hpp"
#include "RTRMetalEngine/Scene/Material.hpp"
#include "RTRMetalEngine/Scene/Mesh.hpp"

namespace rtr::scene {

struct MeshHandle {
    std::size_t index = static_cast<std::size_t>(-1);
    [[nodiscard]] bool isValid() const noexcept { return index != static_cast<std::size_t>(-1); }
};

struct MaterialHandle {
    std::size_t index = static_cast<std::size_t>(-1);
    [[nodiscard]] bool isValid() const noexcept { return index != static_cast<std::size_t>(-1); }
};

struct MeshInstance {
    MeshHandle mesh;
    MaterialHandle material;
    simd_float4x4 transform{matrix_identity_float4x4};
};

class Scene {
public:
    MeshHandle addMesh(Mesh mesh);
    MaterialHandle addMaterial(Material material);
    std::size_t addInstance(MeshHandle mesh, MaterialHandle material, simd_float4x4 transform);

    [[nodiscard]] const std::vector<Mesh>& meshes() const noexcept { return meshes_; }
    [[nodiscard]] const std::vector<Material>& materials() const noexcept { return materials_; }
    [[nodiscard]] const std::vector<MeshInstance>& instances() const noexcept { return instances_; }

    [[nodiscard]] core::math::BoundingBox computeSceneBounds() const noexcept;

private:
    std::vector<Mesh> meshes_;
    std::vector<Material> materials_;
    std::vector<MeshInstance> instances_;
};

}  // namespace rtr::scene
