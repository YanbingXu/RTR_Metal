#pragma once

#include <memory>
#include <string>

namespace rtr::rendering {

class MetalContext;

enum class RayKernelStage {
    RayGeneration,
    Shade,
    Shadow,
    Accumulate,
};

class RayTracingPipeline {
public:
    RayTracingPipeline();
    ~RayTracingPipeline();

    RayTracingPipeline(const RayTracingPipeline&) = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;
    RayTracingPipeline(RayTracingPipeline&&) noexcept;
    RayTracingPipeline& operator=(RayTracingPipeline&&) noexcept;

    bool initialize(MetalContext& context, const std::string& shaderLibraryPath);
    bool isValid() const noexcept;
    void* rawPipelineState(RayKernelStage stage) const noexcept;
    bool hasHardwareKernels() const noexcept;
    bool requiresAccelerationStructure() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtr::rendering
