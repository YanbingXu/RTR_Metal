#include "RTRMetalEngine/Scene/Scene.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace rtr::scene {

MeshHandle Scene::addMesh(Mesh mesh) {
    meshes_.push_back(std::move(mesh));
    return MeshHandle{meshes_.size() - 1};
}

MaterialHandle Scene::addMaterial(Material material) {
    materials_.push_back(std::move(material));
    return MaterialHandle{materials_.size() - 1};
}

std::size_t Scene::addInstance(MeshHandle mesh, MaterialHandle material, simd_float4x4 transform) {
    if (!mesh.isValid() || mesh.index >= meshes_.size()) {
        throw std::out_of_range("Mesh handle out of range");
    }
    if (!material.isValid() || material.index >= materials_.size()) {
        throw std::out_of_range("Material handle out of range");
    }

    instances_.push_back(MeshInstance{mesh, material, transform});
    return instances_.size() - 1;
}

core::math::BoundingBox Scene::computeSceneBounds() const noexcept {
    auto bounds = core::math::BoundingBox::makeEmpty();
    for (const MeshInstance& instance : instances_) {
        if (!instance.mesh.isValid() || instance.mesh.index >= meshes_.size()) {
            continue;
        }
        const Mesh& mesh = meshes_[instance.mesh.index];
        if (mesh.vertices().empty()) {
            continue;
        }
        const core::math::BoundingBox& meshBounds = mesh.bounds();

        const std::array<simd_float3, 8> corners = {
            simd_make_float3(meshBounds.min.x, meshBounds.min.y, meshBounds.min.z),
            simd_make_float3(meshBounds.max.x, meshBounds.min.y, meshBounds.min.z),
            simd_make_float3(meshBounds.min.x, meshBounds.max.y, meshBounds.min.z),
            simd_make_float3(meshBounds.min.x, meshBounds.min.y, meshBounds.max.z),
            simd_make_float3(meshBounds.max.x, meshBounds.max.y, meshBounds.min.z),
            simd_make_float3(meshBounds.max.x, meshBounds.min.y, meshBounds.max.z),
            simd_make_float3(meshBounds.min.x, meshBounds.max.y, meshBounds.max.z),
            simd_make_float3(meshBounds.max.x, meshBounds.max.y, meshBounds.max.z)};

        for (const simd_float3& corner : corners) {
            const simd_float4 transformed = simd_mul(instance.transform, simd_make_float4(corner, 1.0F));
            bounds.expand(simd_make_float3(transformed.x, transformed.y, transformed.z));
        }
    }
    return bounds;
}

}  // namespace rtr::scene
