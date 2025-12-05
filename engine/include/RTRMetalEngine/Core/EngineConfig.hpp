#pragma once

#include <cstdint>
#include <string>

namespace rtr::core {

struct EngineConfig {
    std::string applicationName;
    std::string shaderLibraryPath;
    std::string shadingMode = "auto";  // "auto" or "hardware"
    bool accumulationEnabled = true;
    std::uint32_t accumulationFrames = 0;  // 0 = unlimited
    std::uint32_t maxHardwareBounces = 2;
};

}  // namespace rtr::core
