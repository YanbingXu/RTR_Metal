#include "RTRMetalEngine/Rendering/GeometryStore.hpp"

#include <cstdint>
#include <vector>

#include "RTRMetalEngine/Core/Logger.hpp"

namespace rtr::rendering {

GeometryStore::GeometryStore(BufferAllocator& allocator)
    : allocator_(allocator) {}

std::optional<std::size_t> GeometryStore::uploadMesh(const scene::Mesh& mesh, const std::string& label) {
    if (!allocator_.isDeviceAvailable()) {
        core::Logger::warn("GeometryStore", "Metal device unavailable; mesh '%s' skipped", label.c_str());
        return std::nullopt;
    }

    const auto& vertices = mesh.vertices();
    const auto& indices = mesh.indices();
    if (vertices.empty() || indices.empty()) {
        core::Logger::warn("GeometryStore", "Mesh '%s' has no geometry", label.c_str());
        return std::nullopt;
    }

    constexpr std::size_t kPackedPositionStride = sizeof(float) * 3;
    std::vector<float> packedPositions;
    packedPositions.reserve(vertices.size() * 3);
    for (const auto& vertex : vertices) {
        packedPositions.push_back(vertex.position.x);
        packedPositions.push_back(vertex.position.y);
        packedPositions.push_back(vertex.position.z);
    }
    const std::size_t packedLength = packedPositions.size() * sizeof(float);
    const std::size_t indexLength = indices.size() * sizeof(std::uint32_t);

    const std::string gpuVertexLabel = label + ".vb.gpu";
    const std::string gpuIndexLabel = label + ".ib.gpu";

    BufferHandle gpuVertexBuffer = allocator_.createBuffer(packedLength,
                                                          packedPositions.empty() ? nullptr : packedPositions.data(),
                                                          gpuVertexLabel.c_str());
    BufferHandle gpuIndexBuffer = allocator_.createBuffer(indexLength, indices.data(), gpuIndexLabel.c_str());

    if (!gpuVertexBuffer.isValid() || !gpuIndexBuffer.isValid()) {
        core::Logger::error("GeometryStore", "Failed to upload mesh '%s'", label.c_str());
        return std::nullopt;
    }

    meshes_.emplace_back(std::move(gpuVertexBuffer),
                         std::move(gpuIndexBuffer),
                         vertices.size(),
                         indices.size());
    return meshes_.size() - 1;
}

void GeometryStore::clear() {
    meshes_.clear();
}

}  // namespace rtr::rendering
