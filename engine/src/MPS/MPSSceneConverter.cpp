#include "RTRMetalEngine/MPS/MPSSceneConverter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Scene/Material.hpp"
#include "RTRMetalEngine/Scene/Mesh.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"

namespace rtr::rendering {

namespace {

vector_float3 clampColor(vector_float3 value) {
    return simd_clamp(value, (vector_float3){0.0f, 0.0f, 0.0f}, (vector_float3){1.0f, 1.0f, 1.0f});
}

bool isMonochrome(vector_float3 value, float epsilon = 1e-3f) {
    return std::fabs(value.x - value.y) <= epsilon && std::fabs(value.x - value.z) <= epsilon;
}

bool appendMeshInstance(const scene::Mesh& mesh,
                        const scene::Material* material,
                        simd_float4x4 transform,
                        vector_float3 defaultColor,
                        MPSSceneData& outScene) {
    const auto& vertices = mesh.vertices();
    const auto& indices = mesh.indices();
    if (vertices.empty() || indices.empty()) {
        return false;
    }

    const std::size_t vertexCount = vertices.size();
    const bool indicesValid = std::all_of(indices.begin(), indices.end(), [vertexCount](std::uint32_t idx) {
        return static_cast<std::size_t>(idx) < vertexCount;
    });
    if (!indicesValid) {
        return false;
    }

    const std::size_t baseVertex = outScene.positions.size();
    if (baseVertex > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }

    const vector_float3 materialColor = material ? clampColor(material->albedo) : clampColor(defaultColor);
    constexpr vector_float3 palette[3] = {
        {1.0f, 0.2f, 0.2f},
        {0.2f, 1.0f, 0.2f},
        {0.2f, 0.2f, 1.0f},
    };

    const std::size_t positionStart = outScene.positions.size();
    const std::size_t colorStart = outScene.colors.size();
    const std::size_t indexStart = outScene.indices.size();

    outScene.positions.reserve(outScene.positions.size() + vertexCount);
    outScene.colors.reserve(outScene.colors.size() + vertexCount);
    for (std::size_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex) {
        const auto& vertex = vertices[vertexIndex];
        const simd_float4 position4 = simd_mul(transform, simd_make_float4(vertex.position, 1.0f));
        outScene.positions.push_back(simd_make_float3(position4.x, position4.y, position4.z));
        outScene.colors.push_back(materialColor);
    }

    outScene.indices.reserve(outScene.indices.size() + indices.size());
    for (std::uint32_t idx : indices) {
        const std::size_t transformedIndex = baseVertex + static_cast<std::size_t>(idx);
        if (transformedIndex > std::numeric_limits<std::uint32_t>::max()) {
            outScene.positions.resize(positionStart);
            outScene.colors.resize(colorStart);
            outScene.indices.resize(indexStart);
            return false;
        }
        outScene.indices.push_back(static_cast<std::uint32_t>(transformedIndex));
    }

    return true;
}

}  // namespace

MPSSceneData buildSceneData(const scene::Scene& scene, vector_float3 defaultColor) {
    MPSSceneData sceneData{};

    const auto& meshes = scene.meshes();
    const auto& materials = scene.materials();
    const auto& instances = scene.instances();

    bool appendedAny = false;
    for (std::size_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
        const auto& instance = instances[instanceIndex];
        if (!instance.mesh.isValid() || instance.mesh.index >= meshes.size()) {
            core::Logger::warn("MPSSceneConverter", "Instance %zu references invalid mesh", instanceIndex);
            continue;
        }

        const scene::Mesh& mesh = meshes[instance.mesh.index];
        const scene::Material* material = nullptr;
        if (instance.material.isValid() && instance.material.index < materials.size()) {
            material = &materials[instance.material.index];
        } else if (instance.material.isValid()) {
            core::Logger::warn("MPSSceneConverter", "Instance %zu references invalid material", instanceIndex);
        }

        if (appendMeshInstance(mesh, material, instance.transform, defaultColor, sceneData)) {
            appendedAny = true;
        } else {
            core::Logger::warn("MPSSceneConverter", "Skipped mesh instance %zu due to invalid geometry", instanceIndex);
        }
    }

    if (!appendedAny) {
        for (std::size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
            const scene::Mesh& mesh = meshes[meshIndex];
            if (appendMeshInstance(mesh, nullptr, matrix_identity_float4x4, defaultColor, sceneData)) {
                appendedAny = true;
            } else {
                core::Logger::warn("MPSSceneConverter", "Skipped mesh %zu due to invalid geometry", meshIndex);
            }
        }
    }

    return sceneData;
}

}  // namespace rtr::rendering
