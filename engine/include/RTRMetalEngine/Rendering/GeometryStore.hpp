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
    MeshBuffers(BufferHandle&& vb,
                BufferHandle&& ib,
                std::size_t vtxCount,
                std::size_t idxCount,
                std::size_t stride)
        : vertexBuffer(std::move(vb)),
          indexBuffer(std::move(ib)),
          vertexCount(vtxCount),
          indexCount(idxCount),
          vertexStride(stride) {}

    BufferHandle vertexBuffer;
    BufferHandle indexBuffer;
    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
    std::size_t vertexStride = 0;
};

class GeometryStore {
public:
    explicit GeometryStore(BufferAllocator& allocator);

    std::optional<std::size_t> uploadMesh(const scene::Mesh& mesh, const std::string& label);

    [[nodiscard]] const std::vector<MeshBuffers>& uploadedMeshes() const noexcept { return meshes_; }

private:
    BufferAllocator& allocator_;
    std::vector<MeshBuffers> meshes_;
};

}  // namespace rtr::rendering
