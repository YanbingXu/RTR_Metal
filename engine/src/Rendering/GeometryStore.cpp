#include "RTRMetalEngine/Rendering/GeometryStore.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "RTRMetalEngine/Core/Logger.hpp"

namespace rtr::rendering {
namespace {

std::uint64_t fnv1a64(const void* data, std::size_t byteCount) {
    constexpr std::uint64_t kOffset = 1469598103934665603ull;
    constexpr std::uint64_t kPrime = 1099511628211ull;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = kOffset;
    for (std::size_t i = 0; i < byteCount; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= kPrime;
    }
    return hash;
}

}  // namespace

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

    std::vector<float> positions;
    positions.reserve(vertices.size() * 3);
    for (const auto& vertex : vertices) {
        positions.push_back(vertex.position.x);
        positions.push_back(vertex.position.y);
        positions.push_back(vertex.position.z);
    }
    const std::size_t vertexLength = positions.size() * sizeof(float);
    const std::size_t indexLength = indices.size() * sizeof(std::uint32_t);

    const std::string gpuVertexLabel = label + ".vb.gpu";
    const std::string gpuIndexLabel = label + ".ib.gpu";

    BufferHandle gpuVertexBuffer = allocator_.createBuffer(vertexLength,
                                                          positions.empty() ? nullptr : positions.data(),
                                                          gpuVertexLabel.c_str());
    BufferHandle gpuIndexBuffer = allocator_.createBuffer(indexLength, indices.data(), gpuIndexLabel.c_str());

    if (!gpuVertexBuffer.isValid() || !gpuIndexBuffer.isValid()) {
        core::Logger::error("GeometryStore", "Failed to upload mesh '%s'", label.c_str());
        return std::nullopt;
    }

    if (debugGeometryTrace_) {
        const std::uint64_t vertexHash = fnv1a64(positions.data(), vertexLength);
        const std::uint64_t indexHash = fnv1a64(indices.data(), indexLength);
        core::Logger::info("GeometryStore",
                           "Upload '%s': vertices=%zu (%zu bytes) indices=%zu (%zu bytes) gpuVB=%zu gpuIB=%zu vh=0x%016llX ih=0x%016llX",
                           label.c_str(),
                           vertices.size(),
                           vertexLength,
                           indices.size(),
                           indexLength,
                           gpuVertexBuffer.length(),
                           gpuIndexBuffer.length(),
                           static_cast<unsigned long long>(vertexHash),
                           static_cast<unsigned long long>(indexHash));
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
