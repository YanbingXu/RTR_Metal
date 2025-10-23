#pragma once

#include <memory>
#include <vector>

#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"
#include "RTRMetalEngine/MPS/MPSUniforms.hpp"
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
    ~MPSRenderer();

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
    BufferHandle uniformBuffer_;
    MPSCameraUniforms cameraUniforms_{};
    bool gpuShadingEnabled_ = false;
    struct GPUState;
    std::unique_ptr<GPUState> gpuState_;

    void createUniformBuffer();
    void updateCameraUniforms(std::uint32_t width, std::uint32_t height);
    bool initializeGPUResources();
};

}  // namespace rtr::rendering
