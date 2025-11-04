#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>

#include <simd/simd.h>

#include "RTRMetalEngine/Rendering/AccelerationStructure.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

namespace rtr::rendering {

struct BottomLevelBuildInfo {
    std::size_t accelerationStructureSize = 0;
    std::size_t buildScratchSize = 0;
    std::size_t updateScratchSize = 0;
};

struct TopLevelBuildInfo {
    std::size_t accelerationStructureSize = 0;
    std::size_t buildScratchSize = 0;
    std::size_t updateScratchSize = 0;
    std::size_t instanceDescriptorBufferSize = 0;
};

struct InstanceBuildInput {
    const AccelerationStructure* structure = nullptr;
    simd_float4x4 transform = matrix_identity_float4x4;
    std::uint32_t userID = 0;
    std::uint32_t mask = 0xFF;
    std::uint32_t intersectionFunctionTableOffset = 0;
};

class AccelerationStructureBuilder {
public:
    explicit AccelerationStructureBuilder(MetalContext& context);

    [[nodiscard]] bool isRayTracingSupported() const noexcept;

    std::optional<BottomLevelBuildInfo> queryBottomLevelSizes(const MeshBuffers& meshBuffers,
                                                              const std::string& label) const;

    std::optional<AccelerationStructure> buildBottomLevel(const MeshBuffers& meshBuffers,
                                                         const std::string& label,
                                                         void* commandQueueHandle) const;

    std::optional<TopLevelBuildInfo> queryTopLevelSizes(std::span<const InstanceBuildInput> instances,
                                                        const std::string& label) const;

    std::optional<AccelerationStructure> buildTopLevel(std::span<const InstanceBuildInput> instances,
                                                       const std::string& label,
                                                       void* commandQueueHandle) const;

private:
    MetalContext& context_;
};

}  // namespace rtr::rendering
