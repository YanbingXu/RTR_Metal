#include "RTRMetalEngine/Scene/CornellBox.hpp"

#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

#include <array>
#include <simd/simd.h>

namespace rtr::scene {

namespace {

MeshHandle addQuad(SceneBuilder& builder,
                   const simd_float3& p0,
                   const simd_float3& p1,
                   const simd_float3& p2,
                   const simd_float3& p3) {
    const std::array<simd_float3, 4> positions{p0, p1, p2, p3};
    const std::array<std::uint32_t, 6> indices{0, 1, 2, 0, 2, 3};
    return builder.addTriangleMesh(std::span<const simd_float3>(positions.data(), positions.size()),
                                   std::span<const std::uint32_t>(indices.data(), indices.size()));
}

MeshHandle addAxisAlignedBox(SceneBuilder& builder,
                             const simd_float3& minCorner,
                             const simd_float3& maxCorner) {
    const std::array<simd_float3, 8> positions = {
        simd_make_float3(minCorner.x, minCorner.y, minCorner.z),
        simd_make_float3(maxCorner.x, minCorner.y, minCorner.z),
        simd_make_float3(maxCorner.x, maxCorner.y, minCorner.z),
        simd_make_float3(minCorner.x, maxCorner.y, minCorner.z),
        simd_make_float3(minCorner.x, minCorner.y, maxCorner.z),
        simd_make_float3(maxCorner.x, minCorner.y, maxCorner.z),
        simd_make_float3(maxCorner.x, maxCorner.y, maxCorner.z),
        simd_make_float3(minCorner.x, maxCorner.y, maxCorner.z),
    };

    const std::array<std::uint32_t, 36> indices = {
        0, 1, 2, 0, 2, 3,
        5, 4, 7, 5, 7, 6,
        4, 0, 3, 4, 3, 7,
        1, 5, 6, 1, 6, 2,
        4, 5, 1, 4, 1, 0,
        3, 2, 6, 3, 6, 7,
    };

    return builder.addTriangleMesh(std::span<const simd_float3>(positions.data(), positions.size()),
                                   std::span<const std::uint32_t>(indices.data(), indices.size()));
}

void addCeilingLight(SceneBuilder& builder, Scene& scene) {
    const float lightHalfWidth = 0.25f;
    const float lightHalfDepth = 0.18f;
    const float ceilingY = 1.0f;
    const float lightZ = -1.0f;

    const simd_float3 p0 = simd_make_float3(-lightHalfWidth, ceilingY - 0.001f, -lightHalfDepth - lightZ);
    const simd_float3 p1 = simd_make_float3(lightHalfWidth, ceilingY - 0.001f, -lightHalfDepth - lightZ);
    const simd_float3 p2 = simd_make_float3(lightHalfWidth, ceilingY - 0.001f, lightHalfDepth - lightZ);
    const simd_float3 p3 = simd_make_float3(-lightHalfWidth, ceilingY - 0.001f, lightHalfDepth - lightZ);

    const std::array<simd_float3, 4> positions = {p0, p1, p2, p3};
    const std::array<std::uint32_t, 6> indices = {0, 1, 2, 0, 2, 3};

    auto mesh = builder.addTriangleMesh(positions, indices);

    Material lightMaterial{};
    lightMaterial.albedo = {1.0f, 0.98f, 0.92f};
    lightMaterial.emission = {18.0f, 17.5f, 17.0f};
    lightMaterial.roughness = 0.2f;

    auto lightHandle = scene.addMaterial(lightMaterial);
    scene.addInstance(mesh, lightHandle, matrix_identity_float4x4);
}

}  // namespace

Scene createCornellBoxScene() {
    Scene scene;
    SceneBuilder builder(scene);

    constexpr float roomHalfWidth = 1.0f;
    constexpr float roomHalfHeight = 1.0f;
    constexpr float roomDepth = 1.8f;

    // Floor
    auto floorMesh = addQuad(builder,
                              simd_make_float3(-roomHalfWidth, -roomHalfHeight, 0.0f),
                              simd_make_float3(roomHalfWidth, -roomHalfHeight, 0.0f),
                              simd_make_float3(roomHalfWidth, -roomHalfHeight, -roomDepth),
                              simd_make_float3(-roomHalfWidth, -roomHalfHeight, -roomDepth));
    Material floorMaterial{};
    floorMaterial.albedo = {0.725f, 0.71f, 0.68f};
    auto floorMatHandle = scene.addMaterial(floorMaterial);
    scene.addInstance(floorMesh, floorMatHandle, matrix_identity_float4x4);

    // Ceiling
    auto ceilingMesh = addQuad(builder,
                                simd_make_float3(-roomHalfWidth, roomHalfHeight, -roomDepth),
                                simd_make_float3(roomHalfWidth, roomHalfHeight, -roomDepth),
                                simd_make_float3(roomHalfWidth, roomHalfHeight, 0.0f),
                                simd_make_float3(-roomHalfWidth, roomHalfHeight, 0.0f));
    Material ceilingMaterial{};
    ceilingMaterial.albedo = {0.78f, 0.78f, 0.78f};
    auto ceilingMatHandle = scene.addMaterial(ceilingMaterial);
    scene.addInstance(ceilingMesh, ceilingMatHandle, matrix_identity_float4x4);

    // Back wall
    auto backMesh = addQuad(builder,
                             simd_make_float3(-roomHalfWidth, -roomHalfHeight, -roomDepth),
                             simd_make_float3(roomHalfWidth, -roomHalfHeight, -roomDepth),
                             simd_make_float3(roomHalfWidth, roomHalfHeight, -roomDepth),
                             simd_make_float3(-roomHalfWidth, roomHalfHeight, -roomDepth));
    Material backMaterial{};
    backMaterial.albedo = {0.725f, 0.71f, 0.68f};
    auto backMatHandle = scene.addMaterial(backMaterial);
    scene.addInstance(backMesh, backMatHandle, matrix_identity_float4x4);

    // Left wall (red)
    auto leftMesh = addQuad(builder,
                             simd_make_float3(-roomHalfWidth, -roomHalfHeight, 0.0f),
                             simd_make_float3(-roomHalfWidth, -roomHalfHeight, -roomDepth),
                             simd_make_float3(-roomHalfWidth, roomHalfHeight, -roomDepth),
                             simd_make_float3(-roomHalfWidth, roomHalfHeight, 0.0f));
    Material leftMaterial{};
    leftMaterial.albedo = {0.63f, 0.065f, 0.05f};
    leftMaterial.roughness = 0.45f;
    auto leftMatHandle = scene.addMaterial(leftMaterial);
    scene.addInstance(leftMesh, leftMatHandle, matrix_identity_float4x4);

    // Right wall (green)
    auto rightMesh = addQuad(builder,
                              simd_make_float3(roomHalfWidth, -roomHalfHeight, -roomDepth),
                              simd_make_float3(roomHalfWidth, -roomHalfHeight, 0.0f),
                              simd_make_float3(roomHalfWidth, roomHalfHeight, 0.0f),
                              simd_make_float3(roomHalfWidth, roomHalfHeight, -roomDepth));
    Material rightMaterial{};
    rightMaterial.albedo = {0.14f, 0.45f, 0.091f};
    rightMaterial.roughness = 0.45f;
    auto rightMatHandle = scene.addMaterial(rightMaterial);
    scene.addInstance(rightMesh, rightMatHandle, matrix_identity_float4x4);

    // Short block (axis-aligned for now)
    const simd_float3 shortMin = simd_make_float3(-0.6f, -roomHalfHeight, -1.3f);
    const simd_float3 shortMax = simd_make_float3(-0.1f, -roomHalfHeight + 0.6f, -0.7f);
    auto boxMesh = addAxisAlignedBox(builder, shortMin, shortMax);
    Material blockMaterial{};
    blockMaterial.albedo = {0.73f, 0.73f, 0.73f};
    blockMaterial.roughness = 0.35f;
    auto blockMatHandle = scene.addMaterial(blockMaterial);
    scene.addInstance(boxMesh, blockMatHandle, matrix_identity_float4x4);

    // Tall block (axis-aligned)
    const simd_float3 tallMin = simd_make_float3(0.2f, -roomHalfHeight, -1.6f);
    const simd_float3 tallMax = simd_make_float3(0.6f, -roomHalfHeight + 0.8f, -1.0f);
    auto tallBoxMesh = addAxisAlignedBox(builder, tallMin, tallMax);
    scene.addInstance(tallBoxMesh, blockMatHandle, matrix_identity_float4x4);

    addCeilingLight(builder, scene);

    return scene;
}

}  // namespace rtr::scene
