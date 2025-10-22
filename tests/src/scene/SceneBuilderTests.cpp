#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>

#include <simd/simd.h>

#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

TEST(SceneBuilder, AddsMeshAndMaterial) {
    rtr::scene::Scene scene;
    rtr::scene::SceneBuilder builder(scene);

    const std::array<simd_float3, 3> positions = {
        simd_make_float3(0.0F, 0.0F, 0.0F),
        simd_make_float3(1.0F, 0.0F, 0.0F),
        simd_make_float3(0.0F, 1.0F, 0.0F)};
    const std::array<std::uint32_t, 3> indices = {0, 1, 2};

    const auto meshHandle = builder.addTriangleMesh(positions, indices);
    const auto materialHandle = builder.addDefaultMaterial();

    EXPECT_TRUE(meshHandle.isValid());
    EXPECT_TRUE(materialHandle.isValid());
    EXPECT_EQ(scene.meshes().size(), 1U);
    EXPECT_EQ(scene.materials().size(), 1U);
}

TEST(SceneBuilder, RejectsIncompleteTriangles) {
    rtr::scene::Scene scene;
    rtr::scene::SceneBuilder builder(scene);

    const std::array<simd_float3, 3> positions = {
        simd_make_float3(0.0F, 0.0F, 0.0F),
        simd_make_float3(1.0F, 0.0F, 0.0F),
        simd_make_float3(0.0F, 1.0F, 0.0F)};
    const std::array<std::uint32_t, 2> indices = {0, 1};

    EXPECT_THROW(builder.addTriangleMesh(positions, indices), std::invalid_argument);
}
