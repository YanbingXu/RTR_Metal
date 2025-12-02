#pragma once

#include <memory>
#include <string>

namespace rtr::rendering {

class MetalContext;

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
    void* rayPipelineState() const noexcept;
    void* accumulationPipelineState() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtr::rendering
