#pragma once

#include <optional>
#include <string>

#include "RTRMetalEngine/Rendering/AccelerationStructure.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

namespace rtr::rendering {

struct BottomLevelBuildInfo {
    std::size_t accelerationStructureSize = 0;
    std::size_t buildScratchSize = 0;
    std::size_t updateScratchSize = 0;
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

private:
    MetalContext& context_;
};

}  // namespace rtr::rendering
