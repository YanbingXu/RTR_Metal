#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <simd/simd.h>

namespace rtr::rendering {

class MetalContext;

class MPSPathTracer {
public:
    MPSPathTracer();
    ~MPSPathTracer();

    MPSPathTracer(const MPSPathTracer&) = delete;
    MPSPathTracer& operator=(const MPSPathTracer&) = delete;
    MPSPathTracer(MPSPathTracer&&) noexcept;
    MPSPathTracer& operator=(MPSPathTracer&&) noexcept;

    bool initialize(MetalContext& context);
    bool uploadTriangleScene(std::span<const vector_float3> positions,
                             std::span<const uint32_t> indices,
                             std::span<const vector_float3> colors);

    bool isValid() const noexcept;

    void* deviceHandle() const noexcept;
    void* commandQueueHandle() const noexcept;
    void* accelerationStructureHandle() const noexcept;
    void* intersectorHandle() const noexcept;

    const std::vector<vector_float3>& vertexPositions() const noexcept;
    const std::vector<uint32_t>& indices() const noexcept;
    const std::vector<vector_float3>& vertexColors() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtr::rendering
