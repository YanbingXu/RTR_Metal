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
    enum class ShadingMode {
        Auto,
        CpuOnly,
        GpuPreferred,
    };

    struct FrameComparison {
        std::vector<uint8_t> cpuPixels;
        std::vector<uint8_t> gpuPixels;
        double maxByteDifference = 0.0;
        double maxFloatDifference = 0.0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint64_t cpuPixelHash = 0;
        std::uint64_t gpuPixelHash = 0;
    };

    explicit MPSRenderer(MetalContext& context);
    ~MPSRenderer();

    bool initialize();
    bool initialize(const scene::Scene& scene);
    bool renderFrame(const char* outputPath);
    bool renderFrameComparison(const char* cpuOutputPath,
                               const char* gpuOutputPath,
                               FrameComparison* outComparison = nullptr);
    [[nodiscard]] bool usesGPUShading() const noexcept;
    void setShadingMode(ShadingMode mode) noexcept;
    void resetAccumulation() noexcept;

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
    ShadingMode shadingMode_ = ShadingMode::Auto;
    std::uint32_t gpuFrameIndex_ = 0;

    void createUniformBuffer();
    void updateCameraUniforms(std::uint32_t width, std::uint32_t height);
    bool initializeGPUResources();
    bool computeFrame(FrameComparison& comparison,
                      bool logDifferences,
                      bool enableCpuShading,
                      bool enableGpuShading,
                      bool accumulateGpu);
    static std::uint64_t computePixelHash(const std::vector<uint8_t>& data);
    static bool writePPM(const char* path, const std::vector<uint8_t>& data, std::uint32_t width, std::uint32_t height);
};

}  // namespace rtr::rendering
