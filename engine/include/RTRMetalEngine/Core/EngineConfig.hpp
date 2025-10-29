#pragma once

#include <string>

namespace rtr::core {

struct EngineConfig {
    std::string applicationName;
    std::string shaderLibraryPath;
    std::string shadingMode = "auto";  // "auto", "cpu", or "gpu"
};

}  // namespace rtr::core
