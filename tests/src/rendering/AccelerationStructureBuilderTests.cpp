#include <gtest/gtest.h>

#include "RTRMetalEngine/Rendering/AccelerationStructureBuilder.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

#include <array>
#include <cstdint>
#include <utility>

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

TEST(AccelerationStructureBuilder, QueriesTopLevelSizesWhenSupported) {
    rtr::rendering::MetalContext context;
    rtr::rendering::BufferAllocator allocator(context);
    rtr::rendering::GeometryStore store(allocator);
    rtr::rendering::AccelerationStructureBuilder builder(context);

    auto scene = createSingleTriangleScene();
    const auto& mesh = scene.meshes().front();

    const auto upload = store.uploadMesh(mesh, "triangle");
    void* queueHandle = context.rawCommandQueue();

    if (!context.isValid() || !builder.isRayTracingSupported() || !upload.has_value() || queueHandle == nullptr) {
        GTEST_SKIP() << "Ray tracing path unavailable; skipping TLAS sizing test";
    }

    const auto& meshBuffers = store.uploadedMeshes()[*upload];
    auto blas = builder.buildBottomLevel(meshBuffers, "triangle", queueHandle);
    if (!blas.has_value()) {
        GTEST_SKIP() << "BLAS build unavailable; skipping TLAS sizing test";
    }

    rtr::rendering::InstanceBuildInput input{};
    input.structure = &*blas;

    std::array<rtr::rendering::InstanceBuildInput, 1> instances = {input};

    const auto sizes = builder.queryTopLevelSizes(instances, "triangle_scene");
    ASSERT_TRUE(sizes.has_value());
    EXPECT_GT(sizes->accelerationStructureSize, 0U);
    EXPECT_GT(sizes->buildScratchSize, 0U);
    EXPECT_GT(sizes->instanceDescriptorBufferSize, 0U);
}

TEST(AccelerationStructureBuilder, BuildsTopLevelWhenSupported) {
    rtr::rendering::MetalContext context;
    rtr::rendering::BufferAllocator allocator(context);
    rtr::rendering::GeometryStore store(allocator);
    rtr::rendering::AccelerationStructureBuilder builder(context);

    auto scene = createSingleTriangleScene();
    const auto& mesh = scene.meshes().front();

    const auto upload = store.uploadMesh(mesh, "triangle");
    void* queueHandle = context.rawCommandQueue();

    if (!context.isValid() || !builder.isRayTracingSupported() || !upload.has_value() || queueHandle == nullptr) {
        GTEST_SKIP() << "Ray tracing path unavailable; skipping TLAS build test";
    }

    const auto& meshBuffers = store.uploadedMeshes()[*upload];
    auto blas = builder.buildBottomLevel(meshBuffers, "triangle", queueHandle);
    if (!blas.has_value()) {
        GTEST_SKIP() << "BLAS build unavailable; skipping TLAS build test";
    }

    rtr::rendering::AccelerationStructure bottomLevel = std::move(*blas);

    rtr::rendering::InstanceBuildInput input{};
    input.structure = &bottomLevel;

    std::array<rtr::rendering::InstanceBuildInput, 1> instances = {input};

    auto tlas = builder.buildTopLevel(instances, "triangle_scene", queueHandle);
    ASSERT_TRUE(tlas.has_value());
    EXPECT_TRUE(tlas->isValid());
    EXPECT_GT(tlas->sizeInBytes(), 0U);
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

TEST(AccelerationStructureBuilder, HandlesUnsupportedDevicesForTopLevel) {
    rtr::rendering::MetalContext context;
    rtr::rendering::AccelerationStructureBuilder builder(context);

    if (context.isValid() && builder.isRayTracingSupported()) {
        GTEST_SKIP() << "Ray tracing available; cannot validate unsupported TLAS path";
    }

    rtr::rendering::InstanceBuildInput input{};
    std::array<rtr::rendering::InstanceBuildInput, 1> instances = {input};

    const auto sizes = builder.queryTopLevelSizes(instances, "empty_scene");
    EXPECT_FALSE(sizes.has_value());

    const auto tl = builder.buildTopLevel(instances, "empty_scene", nullptr);
    EXPECT_FALSE(tl.has_value());
}
