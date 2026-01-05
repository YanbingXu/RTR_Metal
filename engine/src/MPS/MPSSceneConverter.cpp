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

bool appendMeshGeometry(const scene::Mesh& mesh,
                        vector_float3 vertexColor,
                        MPSSceneData& outScene,
                        MPSMeshRange& outRange) {
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

    const vector_float3 clampedColor = clampColor(vertexColor);

    const std::size_t positionStart = outScene.positions.size();
    const std::size_t normalStart = outScene.normals.size();
    const std::size_t texcoordStart = outScene.texcoords.size();
    const std::size_t colorStart = outScene.colors.size();
    const std::size_t indexStart = outScene.indices.size();

    outScene.positions.reserve(outScene.positions.size() + vertexCount);
    outScene.normals.reserve(outScene.normals.size() + vertexCount);
    outScene.texcoords.reserve(outScene.texcoords.size() + vertexCount);
    outScene.colors.reserve(outScene.colors.size() + vertexCount);
    for (const auto& vertex : vertices) {
        outScene.positions.push_back(vertex.position);
        outScene.normals.push_back(vertex.normal);
        outScene.texcoords.push_back(vertex.texcoord);
        outScene.colors.push_back(clampedColor);
    }

    outScene.indices.reserve(outScene.indices.size() + indices.size());
    for (std::uint32_t idx : indices) {
        const std::size_t transformedIndex = baseVertex + static_cast<std::size_t>(idx);
        if (transformedIndex > std::numeric_limits<std::uint32_t>::max()) {
            outScene.positions.resize(positionStart);
            outScene.normals.resize(normalStart);
            outScene.texcoords.resize(texcoordStart);
            outScene.colors.resize(colorStart);
            outScene.indices.resize(indexStart);
            return false;
        }
        outScene.indices.push_back(static_cast<std::uint32_t>(transformedIndex));
    }

    outRange.vertexOffset = static_cast<std::uint32_t>(positionStart);
    outRange.vertexCount = static_cast<std::uint32_t>(vertexCount);
    outRange.indexOffset = static_cast<std::uint32_t>(indexStart);
    outRange.indexCount = static_cast<std::uint32_t>(indices.size());
    return true;
}

}  // namespace

MPSSceneData buildSceneData(const scene::Scene& scene, vector_float3 defaultColor, bool enableDebugDump) {
    MPSSceneData sceneData{};
    const auto& meshes = scene.meshes();
    const auto& materials = scene.materials();
    const auto& instances = scene.instances();

    sceneData.instanceRanges.reserve(instances.size());
    sceneData.meshRanges.reserve(meshes.size());

    sceneData.materials.reserve(materials.size());
    for (const auto& mat : materials) {
        MPSMaterialProperties props{};
        props.albedo = mat.albedo;
        props.roughness = mat.roughness;
        props.emission = mat.emission;
        props.metallic = mat.metallic;
        props.reflectivity = mat.reflectivity;
        props.indexOfRefraction = mat.indexOfRefraction;
        sceneData.materials.push_back(props);
    }
    if (sceneData.materials.empty()) {
        MPSMaterialProperties fallback{};
        fallback.albedo = defaultColor;
        fallback.roughness = 0.5f;
        fallback.metallic = 0.0f;
        fallback.reflectivity = 0.0f;
        fallback.indexOfRefraction = 1.0f;
        sceneData.materials.push_back(fallback);
        core::Logger::warn("MPSSceneConverter", "Scene contained no materials; inserted fallback material");
    }

    std::vector<std::uint32_t> meshRangeLookup(meshes.size(), std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> meshMaterialLookup(meshes.size(), std::numeric_limits<std::uint32_t>::max());

    for (const auto& instance : instances) {
        if (!instance.mesh.isValid() || instance.mesh.index >= meshMaterialLookup.size()) {
            continue;
        }
        if (meshMaterialLookup[instance.mesh.index] != std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        if (instance.material.isValid() && instance.material.index < materials.size()) {
            meshMaterialLookup[instance.mesh.index] = static_cast<std::uint32_t>(instance.material.index);
        }
    }
    const vector_float3 neutralColor{1.0f, 1.0f, 1.0f};
    for (std::size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
        const scene::Mesh& mesh = meshes[meshIndex];
        const std::uint32_t meshMaterialIndex = meshMaterialLookup[meshIndex];
        vector_float3 vertexColor = defaultColor;
        if (meshMaterialIndex != std::numeric_limits<std::uint32_t>::max() &&
            meshMaterialIndex < materials.size()) {
            // Keep shared vertex colors neutral so per-instance materials remain accurate.
            vertexColor = neutralColor;
        }
        MPSMeshRange range{};
        if (appendMeshGeometry(mesh, vertexColor, sceneData, range)) {
            range.materialIndex = meshMaterialIndex;
            meshRangeLookup[meshIndex] = static_cast<std::uint32_t>(sceneData.meshRanges.size());
            sceneData.meshRanges.push_back(range);
            if (enableDebugDump) {
                core::Logger::info("MPSSceneConverter",
                                    "Mesh[%zu] -> range=%u verts=%u indices=%u material=%u",
                                    meshIndex,
                                    meshRangeLookup[meshIndex],
                                    range.vertexCount,
                                    range.indexCount,
                                    meshMaterialIndex);
            }
        } else {
            core::Logger::warn("MPSSceneConverter", "Skipped mesh %zu due to invalid geometry", meshIndex);
        }
    }

    auto computeInverse = [](const simd_float4x4& matrix) {
        simd_float4x4 inverse = simd_inverse(matrix);
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                if (!std::isfinite(inverse.columns[column][row])) {
                    return matrix_identity_float4x4;
                }
            }
        }
        return inverse;
    };

    for (std::size_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
        const auto& instance = instances[instanceIndex];
        if (!instance.mesh.isValid() || instance.mesh.index >= meshRangeLookup.size()) {
            core::Logger::warn("MPSSceneConverter", "Instance %zu references invalid mesh", instanceIndex);
            continue;
        }

        const std::uint32_t meshRangeIndex = meshRangeLookup[instance.mesh.index];
        if (meshRangeIndex == std::numeric_limits<std::uint32_t>::max()) {
            core::Logger::warn("MPSSceneConverter", "Mesh %zu missing range; instance %zu skipped", instance.mesh.index, instanceIndex);
            continue;
        }

        MPSInstanceRange range{};
        range.meshIndex = meshRangeIndex;
        if (instance.material.isValid() && instance.material.index < materials.size()) {
            range.materialIndex = static_cast<std::uint32_t>(instance.material.index);
        } else {
            range.materialIndex = 0u;
        }
        range.transform = instance.transform;
        range.inverseTransform = computeInverse(instance.transform);
        sceneData.instanceRanges.push_back(range);
        if (enableDebugDump) {
            const simd_float4& translation = range.transform.columns[3];
            core::Logger::info("MPSSceneConverter",
                                "Instance[%zu] meshRange=%u material=%u translation=(%.3f, %.3f, %.3f)",
                                instanceIndex,
                                range.meshIndex,
                                range.materialIndex,
                                translation.x,
                                translation.y,
                                translation.z);
        }
    }

    if (sceneData.instanceRanges.empty()) {
        for (std::size_t meshIndex = 0; meshIndex < meshRangeLookup.size(); ++meshIndex) {
            const std::uint32_t meshRangeIndex = meshRangeLookup[meshIndex];
            if (meshRangeIndex == std::numeric_limits<std::uint32_t>::max()) {
                continue;
            }
            MPSInstanceRange range{};
            range.meshIndex = meshRangeIndex;
            range.materialIndex = 0u;
            range.transform = matrix_identity_float4x4;
            range.inverseTransform = matrix_identity_float4x4;
            sceneData.instanceRanges.push_back(range);
        }
    }

    return sceneData;
}

}  // namespace rtr::rendering
