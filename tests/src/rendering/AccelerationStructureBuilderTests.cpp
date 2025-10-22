#include <gtest/gtest.h>

#include "RTRMetalEngine/Rendering/AccelerationStructureBuilder.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

#include <array>
#include <cstdint>

#include <simd/simd.h>

namespace {

rtr::scene::Scene createSingleTriangleScene() {
    rtr::scene::Scene scene;
    rtr::scene::SceneBuilder builder(scene);

    const std::array<simd_float3, 3> positions = {
        simd_make_float3(0.0F, 0.0F, 0.0F),
        simd_make_float3(1.0F, 0.0F, 0.0F),
        simd_make_float3(0.0F, 1.0F, 0.0F)};
    const std::array<std::uint32_t, 3> indices = {0, 1, 2};

    auto meshHandle = builder.addTriangleMesh(positions, indices);
    EXPECT_TRUE(meshHandle.isValid());

    auto materialHandle = builder.addDefaultMaterial();
    EXPECT_TRUE(materialHandle.isValid());

    scene.addInstance(meshHandle, materialHandle, matrix_identity_float4x4);
    return scene;
}

}  // namespace

TEST(AccelerationStructureBuilder, QueriesSizesWhenSupported) {
    rtr::rendering::MetalContext context;
    rtr::rendering::BufferAllocator allocator(context);
    rtr::rendering::GeometryStore store(allocator);
    rtr::rendering::AccelerationStructureBuilder builder(context);

    auto scene = createSingleTriangleScene();
    const auto& mesh = scene.meshes().front();

    const auto upload = store.uploadMesh(mesh, "triangle");

    if (!context.isValid() || !builder.isRayTracingSupported() || !upload.has_value()) {
        GTEST_SKIP() << "Ray tracing or GPU upload unavailable on this device";
    }

    const auto& meshBuffers = store.uploadedMeshes()[*upload];
    const auto sizes = builder.queryBottomLevelSizes(meshBuffers, "triangle");

    ASSERT_TRUE(sizes.has_value());
    EXPECT_GT(sizes->accelerationStructureSize, 0U);
    EXPECT_GT(sizes->buildScratchSize, 0U);
}

TEST(AccelerationStructureBuilder, HandlesUnsupportedDevicesGracefully) {
    rtr::rendering::MetalContext context;
    rtr::rendering::AccelerationStructureBuilder builder(context);

    if (context.isValid() && builder.isRayTracingSupported()) {
        GTEST_SKIP() << "Ray tracing available; cannot validate unsupported path";
    }

    rtr::rendering::MeshBuffers buffers{};
    const auto result = builder.queryBottomLevelSizes(buffers, "empty");
    EXPECT_FALSE(result.has_value());
}
