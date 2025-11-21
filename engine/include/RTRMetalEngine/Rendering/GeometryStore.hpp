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
                BufferHandle&& cpuVb,
                BufferHandle&& cpuIb,
                std::size_t vtxCount,
                std::size_t idxCount,
                std::size_t stride)
        : gpuVertexBuffer(std::move(gpuVb)),
          gpuIndexBuffer(std::move(gpuIb)),
          cpuVertexBuffer(std::move(cpuVb)),
          cpuIndexBuffer(std::move(cpuIb)),
          vertexCount(vtxCount),
          indexCount(idxCount),
          vertexStride(stride) {}

    BufferHandle gpuVertexBuffer;
    BufferHandle gpuIndexBuffer;
    BufferHandle cpuVertexBuffer;
    BufferHandle cpuIndexBuffer;
    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
    std::size_t vertexStride = 0;
};

class GeometryStore {
public:
    explicit GeometryStore(BufferAllocator& allocator);

    std::optional<std::size_t> uploadMesh(const scene::Mesh& mesh, const std::string& label);
    void clear();

    [[nodiscard]] const std::vector<MeshBuffers>& uploadedMeshes() const noexcept { return meshes_; }

private:
    BufferAllocator& allocator_;
    std::vector<MeshBuffers> meshes_;
};

}  // namespace rtr::rendering
