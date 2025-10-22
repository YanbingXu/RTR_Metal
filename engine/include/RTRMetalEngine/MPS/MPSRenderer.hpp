#pragma once

#include <memory>
#include <vector>

#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"

namespace rtr::scene {
class Scene;
}

namespace rtr::rendering {

class MetalContext;
class MPSRenderer {
public:
    explicit MPSRenderer(MetalContext& context);

    bool initialize();
    bool initialize(const scene::Scene& scene);
    bool renderFrame(const char* outputPath);

private:
    MetalContext& context_;
    BufferAllocator bufferAllocator_;
    GeometryStore geometryStore_;
    std::vector<std::size_t> uploadedMeshIndices_;
    MPSPathTracer pathTracer_;
};

}  // namespace rtr::rendering
