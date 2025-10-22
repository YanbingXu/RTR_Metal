#include <gtest/gtest.h>

#include <array>

#include <simd/simd.h>

#include "RTRMetalEngine/MPS/MPSSceneConverter.hpp"
#include "RTRMetalEngine/Scene/Material.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

namespace {

using rtr::rendering::buildSceneData;

void expectVectorNear(const vector_float3& lhs, const vector_float3& rhs, float tolerance = 1e-5f) {
    EXPECT_NEAR(lhs.x, rhs.x, tolerance);
    EXPECT_NEAR(lhs.y, rhs.y, tolerance);
    EXPECT_NEAR(lhs.z, rhs.z, tolerance);
}

}  // namespace

TEST(MPSSceneConverterTests, ConvertsSingleInstance) {
    rtr::scene::Scene scene;
    rtr::scene::SceneBuilder builder(scene);

    const std::array<simd_float3, 3> positions = {
        simd_make_float3(-0.25f, -0.25f, 0.0f),
        simd_make_float3(0.25f, -0.25f, 0.0f),
        simd_make_float3(0.0f, 0.35f, 0.0f),
    };
    const std::array<std::uint32_t, 3> indices = {0, 1, 2};

    auto meshHandle = builder.addTriangleMesh(positions, indices);

    rtr::scene::Material customMaterial{};
    customMaterial.albedo = {0.2f, 0.4f, 0.6f};
    auto materialHandle = scene.addMaterial(customMaterial);
    scene.addInstance(meshHandle, materialHandle, matrix_identity_float4x4);

    const auto sceneData = buildSceneData(scene);
    ASSERT_EQ(sceneData.positions.size(), 3U);
    ASSERT_EQ(sceneData.colors.size(), 3U);
    ASSERT_EQ(sceneData.indices.size(), 3U);

    expectVectorNear(sceneData.positions[0], simd_make_float3(-0.25f, -0.25f, 0.0f));
    expectVectorNear(sceneData.positions[1], simd_make_float3(0.25f, -0.25f, 0.0f));
    expectVectorNear(sceneData.positions[2], simd_make_float3(0.0f, 0.35f, 0.0f));

    expectVectorNear(sceneData.colors[0], simd_make_float3(0.2f, 0.08f, 0.12f));
    expectVectorNear(sceneData.colors[1], simd_make_float3(0.04f, 0.4f, 0.12f));
    expectVectorNear(sceneData.colors[2], simd_make_float3(0.04f, 0.08f, 0.6f));

    EXPECT_EQ(sceneData.indices[0], 0U);
    EXPECT_EQ(sceneData.indices[1], 1U);
    EXPECT_EQ(sceneData.indices[2], 2U);
}

TEST(MPSSceneConverterTests, FallsBackToMeshesWhenNoInstancesAndAppliesDefaultColor) {
    rtr::scene::Scene scene;
    rtr::scene::SceneBuilder builder(scene);

    const std::array<simd_float3, 3> positions = {
        simd_make_float3(0.0f, 0.0f, 0.0f),
        simd_make_float3(1.0f, 0.0f, 0.0f),
        simd_make_float3(0.0f, 1.0f, 0.0f),
    };
    const std::array<std::uint32_t, 3> indices = {0, 1, 2};

    builder.addTriangleMesh(positions, indices);

    const vector_float3 defaultColor = {0.8f, 0.7f, 0.6f};
    const auto sceneData = buildSceneData(scene, defaultColor);

    ASSERT_EQ(sceneData.positions.size(), 3U);
    ASSERT_EQ(sceneData.colors.size(), 3U);
    ASSERT_EQ(sceneData.indices.size(), 3U);

    expectVectorNear(sceneData.positions[0], simd_make_float3(0.0f, 0.0f, 0.0f));
    expectVectorNear(sceneData.positions[1], simd_make_float3(1.0f, 0.0f, 0.0f));
    expectVectorNear(sceneData.positions[2], simd_make_float3(0.0f, 1.0f, 0.0f));

    expectVectorNear(sceneData.colors[0], simd_make_float3(0.8f, 0.14f, 0.12f));
    expectVectorNear(sceneData.colors[1], simd_make_float3(0.16f, 0.7f, 0.12f));
    expectVectorNear(sceneData.colors[2], simd_make_float3(0.16f, 0.14f, 0.6f));
}
