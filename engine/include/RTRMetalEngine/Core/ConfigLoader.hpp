#pragma once

#include <filesystem>

#include "RTRMetalEngine/Core/EngineConfig.hpp"

namespace rtr::core {

class ConfigLoader {
public:
    static EngineConfig loadEngineConfig(const std::filesystem::path& path);
};

}  // namespace rtr::core
