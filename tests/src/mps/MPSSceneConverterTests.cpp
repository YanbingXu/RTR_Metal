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
    ASSERT_EQ(sceneData.instanceRanges.size(), 1U);
    EXPECT_EQ(sceneData.instanceRanges[0].vertexCount, 3U);
    EXPECT_EQ(sceneData.instanceRanges[0].indexCount, 3U);

    expectVectorNear(sceneData.positions[0], simd_make_float3(-0.25f, -0.25f, 0.0f));
    expectVectorNear(sceneData.positions[1], simd_make_float3(0.25f, -0.25f, 0.0f));
    expectVectorNear(sceneData.positions[2], simd_make_float3(0.0f, 0.35f, 0.0f));

    expectVectorNear(sceneData.colors[0], customMaterial.albedo);
    expectVectorNear(sceneData.colors[1], customMaterial.albedo);
    expectVectorNear(sceneData.colors[2], customMaterial.albedo);

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
    ASSERT_EQ(sceneData.instanceRanges.size(), 1U);

    expectVectorNear(sceneData.positions[0], simd_make_float3(0.0f, 0.0f, 0.0f));
    expectVectorNear(sceneData.positions[1], simd_make_float3(1.0f, 0.0f, 0.0f));
    expectVectorNear(sceneData.positions[2], simd_make_float3(0.0f, 1.0f, 0.0f));

    expectVectorNear(sceneData.colors[0], defaultColor);
    expectVectorNear(sceneData.colors[1], defaultColor);
    expectVectorNear(sceneData.colors[2], defaultColor);
}

TEST(MPSSceneConverterTests, MultipleInstancesAppendWithOffsetIndices) {
    rtr::scene::Scene scene;
    rtr::scene::SceneBuilder builder(scene);

    const std::array<simd_float3, 3> triA = {
        simd_make_float3(-0.2f, 0.0f, 0.0f),
        simd_make_float3(0.0f, 0.3f, 0.0f),
        simd_make_float3(0.2f, 0.0f, 0.0f),
    };
    const std::array<std::uint32_t, 3> triIndices = {0U, 1U, 2U};
    auto meshA = builder.addTriangleMesh(triA, triIndices);

    const std::array<simd_float3, 3> triB = {
        simd_make_float3(-0.1f, -0.5f, 0.0f),
        simd_make_float3(0.1f, -0.5f, 0.0f),
        simd_make_float3(0.0f, -0.2f, 0.0f),
    };
    auto meshB = builder.addTriangleMesh(triB, triIndices);

    auto matA = builder.addDefaultMaterial();
    scene.addInstance(meshA, matA, matrix_identity_float4x4);
    scene.addInstance(meshB, matA, matrix_identity_float4x4);

    const auto sceneData = buildSceneData(scene);
    ASSERT_EQ(sceneData.positions.size(), 6U);
    ASSERT_EQ(sceneData.indices.size(), 6U);
    ASSERT_EQ(sceneData.instanceRanges.size(), 2U);
    EXPECT_EQ(sceneData.instanceRanges[0].indexCount, 3U);
    EXPECT_EQ(sceneData.instanceRanges[1].indexCount, 3U);

    EXPECT_EQ(sceneData.indices[0], 0U);
    EXPECT_EQ(sceneData.indices[1], 1U);
    EXPECT_EQ(sceneData.indices[2], 2U);
    EXPECT_EQ(sceneData.indices[3], 3U);
    EXPECT_EQ(sceneData.indices[4], 4U);
    EXPECT_EQ(sceneData.indices[5], 5U);

    expectVectorNear(sceneData.positions[0], triA[0]);
    expectVectorNear(sceneData.positions[3], triB[0]);
}
