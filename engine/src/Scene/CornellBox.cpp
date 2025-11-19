#include "RTRMetalEngine/Scene/CornellBox.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Scene/Material.hpp"
#include "RTRMetalEngine/Scene/OBJLoader.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <simd/simd.h>
#include <vector>

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

MeshHandle addTexturedBox(SceneBuilder& builder,
                          const simd_float3& minCorner,
                          const simd_float3& maxCorner) {
    const simd_float3 p000 = simd_make_float3(minCorner.x, minCorner.y, minCorner.z);
    const simd_float3 p100 = simd_make_float3(maxCorner.x, minCorner.y, minCorner.z);
    const simd_float3 p110 = simd_make_float3(maxCorner.x, maxCorner.y, minCorner.z);
    const simd_float3 p010 = simd_make_float3(minCorner.x, maxCorner.y, minCorner.z);
    const simd_float3 p001 = simd_make_float3(minCorner.x, minCorner.y, maxCorner.z);
    const simd_float3 p101 = simd_make_float3(maxCorner.x, minCorner.y, maxCorner.z);
    const simd_float3 p111 = simd_make_float3(maxCorner.x, maxCorner.y, maxCorner.z);
    const simd_float3 p011 = simd_make_float3(minCorner.x, maxCorner.y, maxCorner.z);

    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    vertices.reserve(24);
    indices.reserve(36);

    const std::array<simd_float2, 4> faceUVs = {
        simd_make_float2(0.0f, 0.0f),
        simd_make_float2(1.0f, 0.0f),
        simd_make_float2(1.0f, 1.0f),
        simd_make_float2(0.0f, 1.0f),
    };

    auto emitFace = [&](const std::array<simd_float3, 4>& corners, const simd_float3& normal) {
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(vertices.size());
        for (std::size_t i = 0; i < corners.size(); ++i) {
            Vertex vertex{};
            vertex.position = corners[i];
            vertex.normal = normal;
            vertex.texcoord = faceUVs[i];
            vertices.push_back(vertex);
        }
        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 1);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 3);
    };

    emitFace({p000, p100, p110, p010}, simd_make_float3(0.0f, 0.0f, -1.0f));  // back
    emitFace({p101, p001, p011, p111}, simd_make_float3(0.0f, 0.0f, 1.0f));   // front
    emitFace({p001, p000, p010, p011}, simd_make_float3(-1.0f, 0.0f, 0.0f));  // left
    emitFace({p100, p101, p111, p110}, simd_make_float3(1.0f, 0.0f, 0.0f));   // right
    emitFace({p001, p101, p100, p000}, simd_make_float3(0.0f, -1.0f, 0.0f));  // bottom
    emitFace({p010, p110, p111, p011}, simd_make_float3(0.0f, 1.0f, 0.0f));   // top

    return builder.addTriangleMesh(std::span<const Vertex>(vertices.data(), vertices.size()),
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

MeshHandle addSphere(SceneBuilder& builder, float radius, std::uint32_t slices, std::uint32_t stacks) {
    std::vector<simd_float3> positions;
    std::vector<std::uint32_t> indices;
    positions.reserve((stacks + 1) * (slices + 1));

    for (std::uint32_t stack = 0; stack <= stacks; ++stack) {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * static_cast<float>(M_PI);
        const float y = std::cos(phi);
        const float ringRadius = std::sin(phi);
        for (std::uint32_t slice = 0; slice <= slices; ++slice) {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * static_cast<float>(M_PI) * 2.0f;
            const float x = ringRadius * std::cos(theta);
            const float z = ringRadius * std::sin(theta);
            positions.push_back(simd_make_float3(x * radius, y * radius, z * radius));
        }
    }

    for (std::uint32_t stack = 0; stack < stacks; ++stack) {
        for (std::uint32_t slice = 0; slice < slices; ++slice) {
            const std::uint32_t first = stack * (slices + 1) + slice;
            const std::uint32_t second = first + slices + 1;
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    return builder.addTriangleMesh(std::span<const simd_float3>(positions.data(), positions.size()),
                                   std::span<const std::uint32_t>(indices.data(), indices.size()));
}

simd_float4x4 makeTransform(const simd_float3& translation, float scale) {
    simd_float4x4 transform = matrix_identity_float4x4;
    transform.columns[0] = simd_make_float4(scale, 0.0f, 0.0f, 0.0f);
    transform.columns[1] = simd_make_float4(0.0f, scale, 0.0f, 0.0f);
    transform.columns[2] = simd_make_float4(0.0f, 0.0f, scale, 0.0f);
    transform.columns[3] = simd_make_float4(translation.x, translation.y, translation.z, 1.0f);
    return transform;
}

void addMario(SceneBuilder& builder,
              Scene& scene,
              const std::filesystem::path& assetRoot,
              float supportY) {
    if (assetRoot.empty()) {
        return;
    }
    const auto marioPath = assetRoot / "mario.obj";
    if (!std::filesystem::exists(marioPath)) {
        rtr::core::Logger::warn("CornellBox", "Mario asset missing at %s", marioPath.string().c_str());
        return;
    }

    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    if (!loadOBJMesh(marioPath, vertices, indices)) {
        rtr::core::Logger::warn("CornellBox", "Failed to load %s", marioPath.string().c_str());
        return;
    }

    simd_float3 minPoint = vertices.front().position;
    simd_float3 maxPoint = vertices.front().position;
    for (const auto& v : vertices) {
        minPoint = simd_min(minPoint, v.position);
        maxPoint = simd_max(maxPoint, v.position);
    }
    const simd_float3 centre = (minPoint + maxPoint) * 0.5f;
    const simd_float3 extent = maxPoint - minPoint;
    const float largest = std::max({extent.x, extent.y, extent.z, 1e-3f});
    const float scale = 0.62f / largest;
    for (auto& vertex : vertices) {
        vertex.position = (vertex.position - centre) * scale;
    }

    simd_float3 scaledMin = vertices.front().position;
    simd_float3 scaledMax = vertices.front().position;
    for (const auto& v : vertices) {
        scaledMin = simd_min(scaledMin, v.position);
        scaledMax = simd_max(scaledMax, v.position);
    }
    const float supportLift = (supportY - scaledMin.y) + 0.01f;

    Material marioMaterial{};
    marioMaterial.albedo = {0.9f, 0.35f, 0.3f};
    marioMaterial.roughness = 0.35f;
    marioMaterial.reflectivity = 0.2f;
    const auto marioTexture = assetRoot / "mario.png";
    if (std::filesystem::exists(marioTexture)) {
        marioMaterial.albedoTexturePath = marioTexture.string();
    } else {
        rtr::core::Logger::warn("CornellBox", "Mario texture missing at %s", marioTexture.string().c_str());
    }

    auto mesh = builder.addTriangleMesh(std::span<const Vertex>(vertices.data(), vertices.size()),
                                        std::span<const std::uint32_t>(indices.data(), indices.size()));
    auto matHandle = scene.addMaterial(marioMaterial);
    const simd_float3 marioTranslation = simd_make_float3(0.2f, supportLift, -0.95f);
    scene.addInstance(mesh, matHandle, makeTransform(marioTranslation, 1.0f));
}

void addFeatureGeometry(SceneBuilder& builder,
                        Scene& scene,
                        const std::filesystem::path& assetRoot,
                        float floorY,
                        float crateTopY) {
    Material mirror{};
    mirror.albedo = {0.95f, 0.95f, 0.95f};
    mirror.roughness = 0.05f;
    mirror.metallic = 1.0f;
    mirror.reflectivity = 0.95f;
    mirror.indexOfRefraction = 1.0f;
    constexpr float mirrorRadius = 0.3f;
    auto mirrorMesh = addSphere(builder, 1.0f, 32, 16);
    auto mirrorMat = scene.addMaterial(mirror);
    const simd_float3 mirrorPos = simd_make_float3(-0.35f, floorY + 0.35f, -0.5f);
    scene.addInstance(mirrorMesh, mirrorMat, makeTransform(mirrorPos, mirrorRadius));

    Material glass{};
    glass.albedo = {0.95f, 0.98f, 1.0f};
    glass.roughness = 0.02f;
    glass.metallic = 0.0f;
    glass.reflectivity = 0.05f;
    glass.indexOfRefraction = 1.5f;
    auto glassMesh = addSphere(builder, 1.0f, 32, 16);
    auto glassMat = scene.addMaterial(glass);
    constexpr float glassRadius = 0.26f;
    const simd_float3 glassPos = simd_make_float3(0.3f, crateTopY + glassRadius, -0.55f);
    scene.addInstance(glassMesh, glassMat, makeTransform(glassPos, glassRadius));

    addMario(builder, scene, assetRoot, crateTopY);
}

Scene createCornellBoxSceneInternal(const std::filesystem::path& assetRoot) {
    Scene scene;
    SceneBuilder builder(scene);

    constexpr float roomHalfWidth = 1.0f;
    constexpr float roomHalfHeight = 1.15f;
    constexpr float roomDepth = 1.8f;
    const float floorY = -roomHalfHeight;

    auto floorMesh = addQuad(builder,
                              simd_make_float3(-roomHalfWidth, floorY, 0.0f),
                              simd_make_float3(roomHalfWidth, floorY, 0.0f),
                              simd_make_float3(roomHalfWidth, floorY, -roomDepth),
                              simd_make_float3(-roomHalfWidth, floorY, -roomDepth));
    Material floorMaterial{};
    floorMaterial.albedo = {0.725f, 0.71f, 0.68f};
    floorMaterial.reflectivity = 0.0f;
    floorMaterial.metallic = 0.0f;
    floorMaterial.indexOfRefraction = 1.0f;
    auto floorMatHandle = scene.addMaterial(floorMaterial);
    scene.addInstance(floorMesh, floorMatHandle, matrix_identity_float4x4);

    auto ceilingMesh = addQuad(builder,
                                simd_make_float3(-roomHalfWidth, roomHalfHeight, -roomDepth),
                                simd_make_float3(roomHalfWidth, roomHalfHeight, -roomDepth),
                                simd_make_float3(roomHalfWidth, roomHalfHeight, 0.0f),
                                simd_make_float3(-roomHalfWidth, roomHalfHeight, 0.0f));
    Material ceilingMaterial{};
    ceilingMaterial.albedo = {0.78f, 0.78f, 0.78f};
    ceilingMaterial.reflectivity = 0.0f;
    ceilingMaterial.metallic = 0.0f;
    ceilingMaterial.indexOfRefraction = 1.0f;
    auto ceilingMatHandle = scene.addMaterial(ceilingMaterial);
    scene.addInstance(ceilingMesh, ceilingMatHandle, matrix_identity_float4x4);

    auto backMesh = addQuad(builder,
                             simd_make_float3(-roomHalfWidth, -roomHalfHeight, -roomDepth),
                             simd_make_float3(roomHalfWidth, -roomHalfHeight, -roomDepth),
                             simd_make_float3(roomHalfWidth, roomHalfHeight, -roomDepth),
                             simd_make_float3(-roomHalfWidth, roomHalfHeight, -roomDepth));
    Material backMaterial{};
    backMaterial.albedo = {0.725f, 0.71f, 0.68f};
    backMaterial.reflectivity = 0.0f;
    backMaterial.metallic = 0.0f;
    backMaterial.indexOfRefraction = 1.0f;
    auto backMatHandle = scene.addMaterial(backMaterial);
    scene.addInstance(backMesh, backMatHandle, matrix_identity_float4x4);

    auto leftMesh = addQuad(builder,
                             simd_make_float3(-roomHalfWidth, floorY, 0.0f),
                             simd_make_float3(-roomHalfWidth, floorY, -roomDepth),
                             simd_make_float3(-roomHalfWidth, roomHalfHeight, -roomDepth),
                             simd_make_float3(-roomHalfWidth, roomHalfHeight, 0.0f));
    Material leftMaterial{};
    leftMaterial.albedo = {0.63f, 0.065f, 0.05f};
    leftMaterial.roughness = 0.45f;
    leftMaterial.reflectivity = 0.0f;
    leftMaterial.metallic = 0.0f;
    leftMaterial.indexOfRefraction = 1.0f;
    auto leftMatHandle = scene.addMaterial(leftMaterial);
    scene.addInstance(leftMesh, leftMatHandle, matrix_identity_float4x4);

    auto rightMesh = addQuad(builder,
                              simd_make_float3(roomHalfWidth, floorY, -roomDepth),
                              simd_make_float3(roomHalfWidth, floorY, 0.0f),
                              simd_make_float3(roomHalfWidth, roomHalfHeight, 0.0f),
                              simd_make_float3(roomHalfWidth, roomHalfHeight, -roomDepth));
    Material rightMaterial{};
    rightMaterial.albedo = {0.14f, 0.45f, 0.091f};
    rightMaterial.roughness = 0.45f;
    rightMaterial.reflectivity = 0.0f;
    rightMaterial.metallic = 0.0f;
    rightMaterial.indexOfRefraction = 1.0f;
    auto rightMatHandle = scene.addMaterial(rightMaterial);
    scene.addInstance(rightMesh, rightMatHandle, matrix_identity_float4x4);

    const simd_float3 shortMin = simd_make_float3(-0.7f, floorY, -1.2f);
    const simd_float3 shortMax = simd_make_float3(-0.1f, floorY + 0.7f, -0.7f);
    auto boxMesh = addAxisAlignedBox(builder, shortMin, shortMax);
    Material blockMaterial{};
    blockMaterial.albedo = {0.73f, 0.73f, 0.73f};
    blockMaterial.roughness = 0.35f;
    blockMaterial.reflectivity = 0.0f;
    blockMaterial.metallic = 0.0f;
    blockMaterial.indexOfRefraction = 1.0f;
    auto blockMatHandle = scene.addMaterial(blockMaterial);
    scene.addInstance(boxMesh, blockMatHandle, matrix_identity_float4x4);

    const simd_float3 crateMin = simd_make_float3(0.05f, floorY, -0.9f);
    const simd_float3 crateMax = simd_make_float3(0.65f, floorY + 0.48f, -0.35f);

    auto crateMesh = addTexturedBox(builder, crateMin, crateMax);
    Material crateMaterial{};
    crateMaterial.albedo = {0.8f, 0.7f, 0.6f};
    crateMaterial.roughness = 0.6f;
    crateMaterial.reflectivity = 0.0f;
    crateMaterial.metallic = 0.0f;
    crateMaterial.indexOfRefraction = 1.0f;
    if (!assetRoot.empty()) {
        const auto crateTexture = assetRoot / "crate.jpg";
        if (std::filesystem::exists(crateTexture)) {
            crateMaterial.albedoTexturePath = crateTexture.string();
        } else {
            rtr::core::Logger::warn("CornellBox",
                                    "Crate texture missing at %s",
                                    crateTexture.string().c_str());
        }
    }
    auto crateMatHandle = scene.addMaterial(crateMaterial);
    scene.addInstance(crateMesh, crateMatHandle, matrix_identity_float4x4);
    const float crateTopY = crateMax.y;

    addCeilingLight(builder, scene);

    addFeatureGeometry(builder, scene, assetRoot, floorY, crateTopY);

    return scene;
}

}  // namespace

Scene createCornellBoxScene() { return createCornellBoxSceneInternal({}); }

Scene createCornellBoxScene(const std::filesystem::path& assetRoot) {
    if (assetRoot.empty()) {
        return createCornellBoxSceneInternal({});
    }
    return createCornellBoxSceneInternal(assetRoot);
}

}  // namespace rtr::scene
