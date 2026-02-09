#pragma once

#include <cstdint>
#include <string>

namespace rtr::core {

struct EngineConfig {
    std::string applicationName;
    std::string shaderLibraryPath;
    std::string shadingMode = "auto";  // "auto" or "hardware"
    std::uint32_t maxHardwareBounces = 2;
};

}  // namespace rtr::core
