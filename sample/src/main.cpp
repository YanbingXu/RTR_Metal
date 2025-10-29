#include <exception>
#include <filesystem>
#include <iostream>

#include "RTRMetalEngine/Core/ConfigLoader.hpp"
#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Core/FileSystem.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/Renderer.hpp"

int main() {
    rtr::core::EngineConfig config{};
    const std::filesystem::path configPath = "config/engine.ini";

    try {
        if (rtr::core::FileSystem::exists(configPath)) {
            config = rtr::core::ConfigLoader::loadEngineConfig(configPath);
        } else {
            config.applicationName = "RTR Metal Sample";
            config.shaderLibraryPath = "shaders/RTRShaders.metallib";
            config.shadingMode = "auto";
            rtr::core::Logger::warn("Sample", "Config file not found at %s, using defaults", configPath.string().c_str());
        }
    } catch (const std::exception& ex) {
        config.applicationName = "RTR Metal Sample";
        config.shaderLibraryPath = "shaders/RTRShaders.metallib";
        config.shadingMode = "auto";
        rtr::core::Logger::error("Sample", "Failed to load config: %s", ex.what());
    }

    rtr::rendering::Renderer renderer{config};
    renderer.renderFrame();

    std::cout << "Sample application executed successfully." << std::endl;
    return 0;
}
