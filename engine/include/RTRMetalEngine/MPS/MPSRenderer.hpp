#pragma once

#include <memory>
#include <vector>

#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"

#include <simd/simd.h>

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
    MPSPathTracer pathTracer_;
    std::vector<vector_float3> cpuScenePositions_;
    std::vector<uint32_t> cpuSceneIndices_;
    std::vector<vector_float3> cpuSceneColors_;
};

}  // namespace rtr::rendering
