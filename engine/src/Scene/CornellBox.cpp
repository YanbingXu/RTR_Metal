#include "RTRMetalEngine/Scene/CornellBox.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Scene/Material.hpp"
#include "RTRMetalEngine/Scene/OBJLoader.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <simd/simd.h>
#include <span>
#include <vector>

namespace rtr::scene {
namespace {

constexpr std::uint32_t FACE_MASK_NONE = 0;
constexpr std::uint32_t FACE_MASK_NEGATIVE_X = 1u << 0u;
constexpr std::uint32_t FACE_MASK_POSITIVE_X = 1u << 1u;
constexpr std::uint32_t FACE_MASK_NEGATIVE_Y = 1u << 2u;
constexpr std::uint32_t FACE_MASK_POSITIVE_Y = 1u << 3u;
constexpr std::uint32_t FACE_MASK_NEGATIVE_Z = 1u << 4u;
constexpr std::uint32_t FACE_MASK_POSITIVE_Z = 1u << 5u;
constexpr std::uint32_t FACE_MASK_ALL = (1u << 6u) - 1u;

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

simd_float4x4 makeTranslation(const simd_float3& translation) {
    simd_float4x4 result = matrix_identity_float4x4;
    result.columns[3] = simd_make_float4(translation.x, translation.y, translation.z, 1.0f);
    return result;
}

simd_float4x4 makeScale(const simd_float3& scale) {
    simd_float4x4 result = matrix_identity_float4x4;
    result.columns[0].x = scale.x;
    result.columns[1].y = scale.y;
    result.columns[2].z = scale.z;
    return result;
}

simd_float4x4 makeRotation(float radians, simd_float3 axis) {
    if (radians == 0.0f) {
        return matrix_identity_float4x4;
    }
    axis = simd_normalize(axis);
    const float ct = std::cos(radians);
    const float st = std::sin(radians);
    const float ci = 1.0f - ct;
    const float x = axis.x;
    const float y = axis.y;
    const float z = axis.z;

    simd_float4x4 result = matrix_identity_float4x4;
    result.columns[0] = simd_make_float4(ct + x * x * ci, x * y * ci + z * st, x * z * ci - y * st, 0.0f);
    result.columns[1] = simd_make_float4(y * x * ci - z * st, ct + y * y * ci, y * z * ci + x * st, 0.0f);
    result.columns[2] = simd_make_float4(z * x * ci + y * st, z * y * ci - x * st, ct + z * z * ci, 0.0f);
    return result;
}

simd_float4x4 composeTransform(const simd_float3& translation,
                               const simd_float3& scale,
                               float rotationRadians,
                               simd_float3 rotationAxis) {
    const simd_float4x4 translationMatrix = makeTranslation(translation);
    const simd_float4x4 rotationMatrix = makeRotation(rotationRadians, rotationAxis);
    const simd_float4x4 scaleMatrix = makeScale(scale);
    return simd_mul(translationMatrix, simd_mul(rotationMatrix, scaleMatrix));
}

MeshHandle addCubeWithTransform(SceneBuilder& builder,
                                const simd_float4x4& transform,
                                bool inwardNormals,
                                bool includeTexcoords,
                                std::uint32_t faceMask) {
    const simd_float3 baseVertices[] = {
        simd_make_float3(-0.5f, -0.5f, -0.5f),
        simd_make_float3(0.5f, -0.5f, -0.5f),
        simd_make_float3(0.5f, 0.5f, -0.5f),
        simd_make_float3(-0.5f, 0.5f, -0.5f),
        simd_make_float3(-0.5f, -0.5f, 0.5f),
        simd_make_float3(0.5f, -0.5f, 0.5f),
        simd_make_float3(0.5f, 0.5f, 0.5f),
        simd_make_float3(-0.5f, 0.5f, 0.5f),
    };

    struct Face {
        std::uint32_t mask;
        std::uint32_t indices[4];
        simd_float3 normal;
    };

    const Face faces[] = {
        {FACE_MASK_NEGATIVE_Z, {0, 1, 2, 3}, simd_make_float3(0.0f, 0.0f, -1.0f)},
        {FACE_MASK_POSITIVE_Z, {5, 4, 7, 6}, simd_make_float3(0.0f, 0.0f, 1.0f)},
        {FACE_MASK_NEGATIVE_X, {4, 0, 3, 7}, simd_make_float3(-1.0f, 0.0f, 0.0f)},
        {FACE_MASK_POSITIVE_X, {1, 5, 6, 2}, simd_make_float3(1.0f, 0.0f, 0.0f)},
        {FACE_MASK_NEGATIVE_Y, {4, 5, 1, 0}, simd_make_float3(0.0f, -1.0f, 0.0f)},
        {FACE_MASK_POSITIVE_Y, {3, 2, 6, 7}, simd_make_float3(0.0f, 1.0f, 0.0f)},
    };

    const simd_float4x4 normalMatrix = simd_transpose(simd_inverse(transform));
    const std::array<simd_float2, 4> faceUVs = {
        simd_make_float2(0.0f, 0.0f),
        simd_make_float2(1.0f, 0.0f),
        simd_make_float2(1.0f, 1.0f),
        simd_make_float2(0.0f, 1.0f),
    };

    auto transformPosition = [&](simd_float3 position) {
        const simd_float4 result = simd_mul(transform, simd_make_float4(position, 1.0f));
        return simd_make_float3(result.x, result.y, result.z);
    };
    auto transformNormal = [&](simd_float3 normal) {
        const simd_float4 result = simd_mul(normalMatrix, simd_make_float4(normal, 0.0f));
        return simd_normalize(simd_make_float3(result.x, result.y, result.z));
    };

    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    vertices.reserve(24);
    indices.reserve(36);

    for (const Face& face : faces) {
        if ((faceMask & face.mask) == 0u) {
            continue;
        }
        const simd_float3 faceNormal = inwardNormals ? -face.normal : face.normal;
        const simd_float3 transformedNormal = transformNormal(faceNormal);
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(vertices.size());
        for (std::size_t i = 0; i < 4; ++i) {
            const simd_float3 position = transformPosition(baseVertices[face.indices[i]]);
            Vertex vertex{};
            vertex.position = position;
            vertex.normal = transformedNormal;
            vertex.texcoord = includeTexcoords ? faceUVs[i] : simd_make_float2(0.0f, 0.0f);
            vertices.push_back(vertex);
        }
        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 1);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 3);
    }

    return builder.addTriangleMesh(std::span<const Vertex>(vertices.data(), vertices.size()),
                                   std::span<const std::uint32_t>(indices.data(), indices.size()));
}

MeshHandle addSphere(SceneBuilder& builder,
                     std::uint32_t /*slices*/,
                     std::uint32_t /*stacks*/,
                     const simd_float4x4& transform) {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    vertices.reserve(6);
    indices.reserve(8 * 3);

    const simd_float4x4 normalMatrix = simd_transpose(simd_inverse(transform));

    auto transformPosition = [&](simd_float3 position) {
        const simd_float4 result = simd_mul(transform, simd_make_float4(position, 1.0f));
        return simd_make_float3(result.x, result.y, result.z);
    };

    auto transformNormal = [&](simd_float3 normal) {
        const simd_float4 result = simd_mul(normalMatrix, simd_make_float4(normal, 0.0f));
        return simd_normalize(simd_make_float3(result.x, result.y, result.z));
    };

    auto pushVertex = [&](const simd_float3& localPos, float u, float v) {
        Vertex vertex{};
        vertex.position = transformPosition(localPos);
        vertex.normal = transformNormal(localPos);
        vertex.texcoord = simd_make_float2(u, 1.0f - v);
        vertices.push_back(vertex);
    };
    pushVertex(simd_make_float3(0.0f, 1.0f, 0.0f), 0.5f, 0.0f);   // top
    pushVertex(simd_make_float3(1.0f, 0.0f, 0.0f), 1.0f, 0.5f);   // +x
    pushVertex(simd_make_float3(0.0f, 0.0f, 1.0f), 0.75f, 0.5f);  // +z
    pushVertex(simd_make_float3(-1.0f, 0.0f, 0.0f), 0.5f, 0.5f);  // -x
    pushVertex(simd_make_float3(0.0f, 0.0f, -1.0f), 0.25f, 0.5f); // -z
    pushVertex(simd_make_float3(0.0f, -1.0f, 0.0f), 0.5f, 1.0f);  // bottom

    const std::uint32_t kTris[] = {
        0, 1, 2,
        0, 2, 3,
        0, 3, 4,
        0, 4, 1,
        5, 2, 1,
        5, 3, 2,
        5, 4, 3,
        5, 1, 4,
    };
    indices.assign(std::begin(kTris), std::end(kTris));

    return builder.addTriangleMesh(std::span<const Vertex>(vertices.data(), vertices.size()),
                                   std::span<const std::uint32_t>(indices.data(), indices.size()));
}

void addMario(SceneBuilder& builder,
              Scene& scene,
              const std::filesystem::path& assetRoot,
              float supportY,
              const simd_float3& translation,
              float rotationRadians) {
    const simd_float3 marioTranslation = simd_make_float3(translation.x, supportY + 0.18f, translation.z);
    const simd_float4x4 marioTransform = composeTransform(marioTranslation,
                                                          simd_make_float3(0.12f, 0.18f, 0.12f),
                                                          rotationRadians,
                                                          simd_make_float3(0.0f, 1.0f, 0.0f));

    Material marioMaterial{};
    marioMaterial.albedo = {1.0f, 1.0f, 1.0f};
    marioMaterial.roughness = 0.6f;
    marioMaterial.reflectivity = 0.0f;
    const auto marioTexture = assetRoot / "mario.png";
    if (std::filesystem::exists(marioTexture)) {
        marioMaterial.albedoTexturePath = marioTexture.string();
    } else {
        marioMaterial.albedo = {0.95f, 0.2f, 0.8f};
        rtr::core::Logger::warn("CornellBox",
                                "Mario texture missing at %s; using colored placeholder",
                                marioTexture.string().c_str());
    }
    auto mesh = addTexturedBox(builder,
                               simd_make_float3(-1.0f, -1.0f, -1.0f),
                               simd_make_float3(1.0f, 1.0f, 1.0f));
    auto matHandle = scene.addMaterial(marioMaterial);
    scene.addInstance(mesh, matHandle, marioTransform);
    rtr::core::Logger::info("CornellBox",
                            "Mario uses temporary placeholder mesh (OBJ path tracked separately)");
}

void addFeatureGeometry(SceneBuilder& builder,
                        Scene& scene,
                        const std::filesystem::path& assetRoot,
                        float /*crateTopY*/) {
    Material mirror{};
    mirror.albedo = {1.0f, 1.0f, 1.0f};
    mirror.roughness = 0.02f;
    mirror.metallic = 1.0f;
    mirror.reflectivity = 1.0f;
    mirror.indexOfRefraction = 1.0f;
    auto mirrorMat = scene.addMaterial(mirror);
    const simd_float4x4 mirrorTransform = composeTransform(simd_make_float3(-0.42f, 0.28f, 0.20f),
                                                          simd_make_float3(0.28f, 0.28f, 0.28f),
                                                          0.0f,
                                                          simd_make_float3(1.0f, 0.0f, 0.0f));
    auto mirrorMesh = addSphere(builder, 24, 12, matrix_identity_float4x4);
    scene.addInstance(mirrorMesh, mirrorMat, mirrorTransform);

    Material glass{};
    glass.albedo = {1.0f, 1.0f, 1.0f};
    glass.roughness = 0.0f;
    glass.reflectivity = 0.0f;
    glass.indexOfRefraction = 1.5f;
    auto glassMat = scene.addMaterial(glass);
    const simd_float4x4 glassTransform = composeTransform(simd_make_float3(0.36f, 0.22f, 0.32f),
                                                         simd_make_float3(0.22f, 0.22f, 0.22f),
                                                         0.0f,
                                                         simd_make_float3(1.0f, 0.0f, 0.0f));
    auto glassMesh = addSphere(builder, 24, 12, matrix_identity_float4x4);
    scene.addInstance(glassMesh, glassMat, glassTransform);

    addMario(builder,
             scene,
             assetRoot,
             0.0f,
             simd_make_float3(0.02f, 0.0f, 0.20f),
             0.01f);
}

}  // namespace

Scene createCornellBoxSceneInternal(const std::filesystem::path& assetRoot) {
    Scene scene;
    SceneBuilder builder(scene);

    const simd_float4x4 enclosureTransform = composeTransform(simd_make_float3(0.0f, 1.0f, 0.0f),
                                                              simd_make_float3(2.0f, 2.0f, 2.0f),
                                                              0.0f,
                                                              simd_make_float3(0.0f, 1.0f, 0.0f));

    Material enclosureMaterial{};
    enclosureMaterial.albedo = {0.725f, 0.71f, 0.68f};
    auto enclosureHandle = scene.addMaterial(enclosureMaterial);
    auto enclosureMesh = addCubeWithTransform(builder,
                                              enclosureTransform,
                                              true,
                                              false,
                                              FACE_MASK_NEGATIVE_Y | FACE_MASK_POSITIVE_Y | FACE_MASK_NEGATIVE_Z);
    scene.addInstance(enclosureMesh, enclosureHandle, matrix_identity_float4x4);

    Material leftMaterial{};
    leftMaterial.albedo = {0.63f, 0.065f, 0.05f};
    auto leftHandle = scene.addMaterial(leftMaterial);
    auto leftMesh = addCubeWithTransform(builder,
                                         enclosureTransform,
                                         true,
                                         false,
                                         FACE_MASK_NEGATIVE_X);
    scene.addInstance(leftMesh, leftHandle, matrix_identity_float4x4);

    Material rightMaterial{};
    rightMaterial.albedo = {0.14f, 0.45f, 0.091f};
    auto rightHandle = scene.addMaterial(rightMaterial);
    auto rightMesh = addCubeWithTransform(builder,
                                          enclosureTransform,
                                          true,
                                          false,
                                          FACE_MASK_POSITIVE_X);
    scene.addInstance(rightMesh, rightHandle, matrix_identity_float4x4);

    const simd_float4x4 lightTransform = composeTransform(simd_make_float3(0.0f, 1.0f, 0.0f),
                                                         simd_make_float3(0.5f, 1.98f, 0.5f),
                                                         0.0f,
                                                         simd_make_float3(0.0f, 1.0f, 0.0f));
    auto lightMesh = addCubeWithTransform(builder,
                                          lightTransform,
                                          true,
                                          false,
                                          FACE_MASK_POSITIVE_Y);
    Material lightMaterial{};
    lightMaterial.albedo = {1.0f, 1.0f, 1.0f};
    lightMaterial.emission = {8.0f, 8.0f, 8.0f};
    lightMaterial.roughness = 0.2f;
    auto lightHandle = scene.addMaterial(lightMaterial);
    scene.addInstance(lightMesh, lightHandle, matrix_identity_float4x4);

    Material shortBoxMaterial{};
    shortBoxMaterial.albedo = {0.725f, 0.71f, 0.68f};
    shortBoxMaterial.roughness = 0.4f;
    if (!assetRoot.empty()) {
        const auto crateTexture = assetRoot / "crate.jpg";
        if (std::filesystem::exists(crateTexture)) {
            shortBoxMaterial.albedoTexturePath = crateTexture.string();
        } else {
            rtr::core::Logger::warn("CornellBox",
                                    "Crate texture missing at %s",
                                    crateTexture.string().c_str());
        }
    }
    const simd_float3 shortTranslation = simd_make_float3(0.3275f, 0.3f, 0.0f);
    const simd_float3 shortScale = simd_make_float3(0.6f, 0.6f, 0.6f);
    const float shortRotation = -0.3f;
    const simd_float4x4 shortTransform = composeTransform(shortTranslation,
                                                          shortScale,
                                                          shortRotation,
                                                          simd_make_float3(0.0f, 1.0f, 0.0f));
    auto shortMesh = addCubeWithTransform(builder, shortTransform, false, true, FACE_MASK_ALL);
    auto shortHandle = scene.addMaterial(shortBoxMaterial);
    scene.addInstance(shortMesh, shortHandle, matrix_identity_float4x4);
    const float crateTopY = shortTranslation.y + shortScale.y * 0.5f;

    Material tallBoxMaterial{};
    tallBoxMaterial.albedo = {0.725f, 0.71f, 0.68f};
    const simd_float3 tallTranslation = simd_make_float3(-0.335f, 0.6f, -0.29f);
    const simd_float3 tallScale = simd_make_float3(0.6f, 1.2f, 0.6f);
    const float tallRotation = 0.3f;
    const simd_float4x4 tallTransform = composeTransform(tallTranslation,
                                                         tallScale,
                                                         tallRotation,
                                                         simd_make_float3(0.0f, 1.0f, 0.0f));
    auto tallMesh = addCubeWithTransform(builder, tallTransform, false, false, FACE_MASK_ALL);
    auto tallHandle = scene.addMaterial(tallBoxMaterial);
    scene.addInstance(tallMesh, tallHandle, matrix_identity_float4x4);

    addFeatureGeometry(builder, scene, assetRoot, crateTopY);

    return scene;
}
Scene createCornellBoxScene() { return createCornellBoxSceneInternal({}); }

Scene createCornellBoxScene(const std::filesystem::path& assetRoot) {
    if (assetRoot.empty()) {
        return createCornellBoxSceneInternal({});
    }
    return createCornellBoxSceneInternal(assetRoot);
}

}  // namespace rtr::scene
