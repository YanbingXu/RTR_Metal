#pragma once

#include <cstdint>
#include <limits>
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
    void setFrameDimensions(std::uint32_t width, std::uint32_t height) noexcept;
    [[nodiscard]] std::uint32_t frameWidth() const noexcept { return frameWidth_; }
    [[nodiscard]] std::uint32_t frameHeight() const noexcept { return frameHeight_; }
    void setAccumulationParameters(bool enabled, std::uint32_t maxFrames) noexcept;
    [[nodiscard]] bool accumulationEnabled() const noexcept { return accumulationEnabled_; }
    [[nodiscard]] std::uint32_t accumulationTargetFrames() const noexcept { return accumulationTargetFrames_; }
    [[nodiscard]] std::uint32_t accumulatedFrames() const noexcept;
    void setSamplingParameters(std::uint32_t samplesPerPixel, std::uint32_t seed) noexcept;
    [[nodiscard]] std::uint32_t samplesPerPixel() const noexcept { return samplesPerPixel_; }
    [[nodiscard]] std::uint32_t sampleSeed() const noexcept { return baseSeed_; }
    void resetAccumulation() noexcept;

private:
    struct MaterialProperties {
        vector_float3 albedo{1.0f, 1.0f, 1.0f};
        float roughness = 0.5f;
        vector_float3 emission{0.0f, 0.0f, 0.0f};
        float metallic = 0.0f;
        float reflectivity = 0.0f;
        float indexOfRefraction = 1.5f;
    };

    MetalContext& context_;
    BufferAllocator bufferAllocator_;
    GeometryStore geometryStore_;
    MPSPathTracer pathTracer_;
    std::vector<vector_float3> cpuScenePositions_;
    std::vector<uint32_t> cpuSceneIndices_;
    std::vector<vector_float3> cpuSceneColors_;
    std::vector<uint32_t> cpuScenePrimitiveMaterials_;
    std::vector<MaterialProperties> materials_;
    BufferHandle uniformBuffer_;
    BufferHandle rayBuffer_;
    BufferHandle intersectionBuffer_;
    MPSCameraUniforms cameraUniforms_{};
    bool gpuShadingEnabled_ = false;
    struct GPUState;
    std::unique_ptr<GPUState> gpuState_;
    ShadingMode shadingMode_ = ShadingMode::Auto;
    std::uint32_t gpuFrameIndex_ = 0;
    std::uint32_t frameWidth_ = 512;
    std::uint32_t frameHeight_ = 512;
    bool accumulationEnabled_ = true;
    std::uint32_t accumulationTargetFrames_ = 0;
    std::uint32_t samplesPerPixel_ = 1;
    std::uint32_t baseSeed_ = 0;

    void createUniformBuffer();
    void updateCameraUniforms();
    bool initializeGPUResources();
    bool ensureFrameBuffers();
    bool computeFrame(FrameComparison& comparison,
                      bool logDifferences,
                      bool enableCpuShading,
                      bool enableGpuShading,
                      bool accumulateGpu);
    bool uploadScene(const scene::Scene& scene);
    static std::uint64_t computePixelHash(const std::vector<uint8_t>& data);
    static bool writePPM(const char* path, const std::vector<uint8_t>& data, std::uint32_t width, std::uint32_t height);
};

}  // namespace rtr::rendering
