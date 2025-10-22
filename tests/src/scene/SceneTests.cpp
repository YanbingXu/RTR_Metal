#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "RTRMetalEngine/Core/Math.hpp"
#include "RTRMetalEngine/Scene/Mesh.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/Vertex.hpp"

namespace {

rtr::scene::Mesh makeUnitCube() {
    using rtr::scene::Vertex;
    std::vector<Vertex> vertices = {
        {{-0.5F, -0.5F, -0.5F}},
        {{0.5F, -0.5F, -0.5F}},
        {{-0.5F, 0.5F, -0.5F}},
        {{-0.5F, -0.5F, 0.5F}},
        {{0.5F, 0.5F, -0.5F}},
        {{0.5F, -0.5F, 0.5F}},
        {{-0.5F, 0.5F, 0.5F}},
        {{0.5F, 0.5F, 0.5F}},
    };

    std::vector<std::uint32_t> indices = {0, 1, 2, 3, 4, 5};
    return rtr::scene::Mesh{std::move(vertices), std::move(indices)};
}

}  // namespace

TEST(SceneMesh, ComputesBoundingBoxFromVertices) {
    const rtr::scene::Mesh mesh = makeUnitCube();
    const auto bounds = mesh.bounds();

    EXPECT_FLOAT_EQ(bounds.min.x, -0.5F);
    EXPECT_FLOAT_EQ(bounds.max.x, 0.5F);
    EXPECT_FLOAT_EQ(bounds.min.y, -0.5F);
    EXPECT_FLOAT_EQ(bounds.max.y, 0.5F);
    EXPECT_FLOAT_EQ(bounds.min.z, -0.5F);
    EXPECT_FLOAT_EQ(bounds.max.z, 0.5F);
}

TEST(SceneGraph, ComputesSceneBoundsWithTransforms) {
    rtr::scene::Scene scene;
    const auto meshHandle = scene.addMesh(makeUnitCube());
    const auto materialHandle = scene.addMaterial({});

    simd_float4x4 translate = matrix_identity_float4x4;
    translate.columns[3] = simd_make_float4(2.0F, 0.0F, 0.0F, 1.0F);

    scene.addInstance(meshHandle, materialHandle, translate);

    const auto bounds = scene.computeSceneBounds();
    EXPECT_NEAR(bounds.min.x, 1.5F, 1e-5F);
    EXPECT_NEAR(bounds.max.x, 2.5F, 1e-5F);
}
