#include "RTRMetalEngine/Rendering/GeometryStore.hpp"

#include <cstdint>

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

    const std::size_t vertexLength = vertices.size() * sizeof(scene::Vertex);
    const std::size_t indexLength = indices.size() * sizeof(std::uint32_t);

    const std::string vertexLabel = label + ".vb";
    const std::string indexLabel = label + ".ib";

    BufferHandle vertexBuffer = allocator_.createBuffer(vertexLength, vertices.data(), vertexLabel.c_str());
    BufferHandle indexBuffer = allocator_.createBuffer(indexLength, indices.data(), indexLabel.c_str());

    if (!vertexBuffer.isValid() || !indexBuffer.isValid()) {
        core::Logger::error("GeometryStore", "Failed to upload mesh '%s'", label.c_str());
        return std::nullopt;
    }

    meshes_.emplace_back(std::move(vertexBuffer), std::move(indexBuffer), indices.size(), sizeof(scene::Vertex));
    return meshes_.size() - 1;
}

}  // namespace rtr::rendering
