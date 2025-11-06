#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include <simd/simd.h>

#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"
#include "RTRMetalEngine/Scene/Mesh.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

TEST(GeometryStore, HandlesContextAvailabilityGracefully) {
    rtr::rendering::MetalContext context;
    rtr::rendering::BufferAllocator allocator(context);
    rtr::rendering::GeometryStore store(allocator);

    rtr::scene::Scene scene;
    rtr::scene::SceneBuilder builder(scene);

    const std::array<simd_float3, 3> positions = {
        simd_make_float3(0.0F, 0.0F, 0.0F),
        simd_make_float3(1.0F, 0.0F, 0.0F),
        simd_make_float3(0.0F, 1.0F, 0.0F)};
    const std::array<std::uint32_t, 3> indices = {0, 1, 2};

    const auto meshHandle = builder.addTriangleMesh(positions, indices);

    ASSERT_TRUE(meshHandle.isValid());
    const auto& mesh = scene.meshes()[meshHandle.index];

    const auto result = store.uploadMesh(mesh, "triangle");

    if (context.isValid()) {
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(store.uploadedMeshes().size(), 1U);
        const auto& buffers = store.uploadedMeshes().front();
        EXPECT_TRUE(buffers.gpuVertexBuffer.isValid());
        EXPECT_TRUE(buffers.gpuIndexBuffer.isValid());
        EXPECT_TRUE(buffers.cpuVertexBuffer.isValid());
        EXPECT_TRUE(buffers.cpuIndexBuffer.isValid());
        EXPECT_EQ(buffers.indexCount, indices.size());
        EXPECT_EQ(buffers.vertexStride, sizeof(rtr::scene::Vertex));
    } else {
        EXPECT_FALSE(result.has_value());
        EXPECT_TRUE(store.uploadedMeshes().empty());
    }
}
