#include "RTRMetalEngine/Scene/DemoScenes.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Scene/Material.hpp"
#include "RTRMetalEngine/Scene/OBJLoader.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <vector>

namespace rtr::scene {

namespace {

void addFloor(SceneBuilder& builder, Scene& scene) {
    const std::array<simd_float3, 6> floorVertices = {
        simd_make_float3(-2.0F, -0.3F, -2.0F),
        simd_make_float3(2.0F, -0.3F, -2.0F),
        simd_make_float3(2.0F, -0.3F, 2.0F),
        simd_make_float3(-2.0F, -0.3F, -2.0F),
        simd_make_float3(2.0F, -0.3F, 2.0F),
        simd_make_float3(-2.0F, -0.3F, 2.0F),
    };
    const std::array<std::uint32_t, 6> floorIndices = {0, 1, 2, 3, 4, 5};

    auto mesh = builder.addTriangleMesh(floorVertices, floorIndices);
    Material floorMaterial{};
    floorMaterial.albedo = {0.6F, 0.6F, 0.6F};
    floorMaterial.roughness = 0.8F;
    auto materialHandle = scene.addMaterial(floorMaterial);
    scene.addInstance(mesh, materialHandle, matrix_identity_float4x4);
}

void addBackdrop(SceneBuilder& builder, Scene& scene) {
    // simple back wall for reflections
    const std::array<simd_float3, 6> wallVertices = {
        simd_make_float3(-2.0F, -0.3F, -2.0F),
        simd_make_float3(2.0F, -0.3F, -2.0F),
        simd_make_float3(2.0F, 2.0F, -2.0F),
        simd_make_float3(-2.0F, -0.3F, -2.0F),
        simd_make_float3(2.0F, 2.0F, -2.0F),
        simd_make_float3(-2.0F, 2.0F, -2.0F),
    };
    const std::array<std::uint32_t, 6> wallIndices = {0, 1, 2, 3, 4, 5};

    auto mesh = builder.addTriangleMesh(wallVertices, wallIndices);
    Material wallMaterial{};
    wallMaterial.albedo = {0.75F, 0.8F, 0.85F};
    wallMaterial.roughness = 0.5F;
    auto materialHandle = scene.addMaterial(wallMaterial);
    scene.addInstance(mesh, materialHandle, matrix_identity_float4x4);
}

void normaliseMesh(std::vector<simd_float3>& positions) {
    if (positions.empty()) {
        return;
    }

    simd_float3 minPoint = positions.front();
    simd_float3 maxPoint = positions.front();
    for (const auto& p : positions) {
        minPoint = simd_min(minPoint, p);
        maxPoint = simd_max(maxPoint, p);
    }

    const simd_float3 centre = (minPoint + maxPoint) * 0.5F;
    const simd_float3 extent = maxPoint - minPoint;
    const float largest = std::max({extent.x, extent.y, extent.z, 1e-3F});
    const float scale = 1.0F / largest;

    for (auto& p : positions) {
        p = (p - centre) * scale;
    }
}

bool addOBJInstance(SceneBuilder& builder,
                    const std::filesystem::path& assetPath,
                    const simd_float4x4& transform,
                    const Material& material,
                    Scene& scene) {
    std::vector<simd_float3> positions;
    std::vector<std::uint32_t> indices;
    if (!loadOBJMesh(assetPath, positions, indices)) {
        rtr::core::Logger::warn("DemoScenes", "Failed to load %s", assetPath.string().c_str());
        return false;
    }

    normaliseMesh(positions);

    auto mesh = builder.addTriangleMesh(std::span<const simd_float3>(positions.data(), positions.size()),
                                        std::span<const std::uint32_t>(indices.data(), indices.size()));
    auto matHandle = scene.addMaterial(material);
    scene.addInstance(mesh, matHandle, transform);
    return true;
}

Scene createDemoScene(const std::filesystem::path& assetRoot,
                      const Material& marioMaterial,
                      float yOffset) {
    Scene scene;
    SceneBuilder builder(scene);

    addFloor(builder, scene);
    addBackdrop(builder, scene);

    const auto marioPath = assetRoot / "mario.obj";
    simd_float4x4 transform = matrix_identity_float4x4;
    transform.columns[3] = simd_make_float4(0.0F, yOffset, 0.0F, 1.0F);

    if (!addOBJInstance(builder, marioPath, transform, marioMaterial, scene)) {
        rtr::core::Logger::warn("DemoScenes", "Falling back to simple prism for demo scene");
        const std::array<simd_float3, 3> prismVertices = {
            simd_make_float3(-0.3F, -0.1F + yOffset, 0.2F),
            simd_make_float3(0.3F, -0.1F + yOffset, -0.2F),
            simd_make_float3(0.0F, 0.6F + yOffset, 0.25F),
        };
        const std::array<std::uint32_t, 3> prismIndices = {0U, 1U, 2U};
        auto mesh = builder.addTriangleMesh(prismVertices, prismIndices);
        auto matHandle = scene.addMaterial(marioMaterial);
        scene.addInstance(mesh, matHandle, matrix_identity_float4x4);
    }

    return scene;
}

}  // namespace

Scene createReflectiveDemoScene(const std::filesystem::path& assetRoot) {
    Material material{};
    material.albedo = {0.95F, 0.95F, 0.95F};
    material.roughness = 0.05F;
    material.metallic = 1.0F;
    return createDemoScene(assetRoot, material, 0.2F);
}

Scene createGlassDemoScene(const std::filesystem::path& assetRoot) {
    Material material{};
    material.albedo = {0.9F, 0.95F, 1.0F};
    material.roughness = 0.02F;
    material.metallic = 0.0F;
    return createDemoScene(assetRoot, material, 0.25F);
}

}  // namespace rtr::scene
