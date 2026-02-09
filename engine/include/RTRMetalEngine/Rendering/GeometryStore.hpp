#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Scene/Mesh.hpp"

namespace rtr::rendering {

struct MeshBuffers {
    MeshBuffers() = default;
    MeshBuffers(BufferHandle&& gpuVb,
                BufferHandle&& gpuIb,
                std::size_t vtxCount,
                std::size_t idxCount)
        : gpuVertexBuffer(std::move(gpuVb)),
          gpuIndexBuffer(std::move(gpuIb)),
          vertexCount(vtxCount),
          indexCount(idxCount) {}

    BufferHandle gpuVertexBuffer;
    BufferHandle gpuIndexBuffer;
    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
};

class GeometryStore {
public:
    explicit GeometryStore(BufferAllocator& allocator);

    std::optional<std::size_t> uploadMesh(const scene::Mesh& mesh, const std::string& label);
    void clear();
    void setDebugGeometryTrace(bool enabled) noexcept { debugGeometryTrace_ = enabled; }

    [[nodiscard]] const std::vector<MeshBuffers>& uploadedMeshes() const noexcept { return meshes_; }

private:
    BufferAllocator& allocator_;
    std::vector<MeshBuffers> meshes_;
    bool debugGeometryTrace_ = false;
};

}  // namespace rtr::rendering
