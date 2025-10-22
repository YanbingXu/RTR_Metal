#import <Foundation/Foundation.h>

#include <iostream>

#include "RTRMetalEngine/Core/ConfigLoader.hpp"
#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Core/FileSystem.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/MPS/MPSRenderer.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

int main() {
    rtr::core::EngineConfig config{};
    const std::filesystem::path configPath = "config/engine.ini";
    if (rtr::core::FileSystem::exists(configPath)) {
        config = rtr::core::ConfigLoader::loadEngineConfig(configPath);
    } else {
        config.applicationName = "RTR Metal MPS Sample";
        config.shaderLibraryPath = "shaders/RTRShaders.metallib";
        rtr::core::Logger::warn("MPSSample", "Config file not found, using defaults");
    }

    rtr::rendering::MetalContext context;
    if (!context.isValid()) {
        rtr::core::Logger::error("MPSSample", "Metal context initialization failed");
        return 1;
    }

    rtr::rendering::MPSRenderer renderer(context);
    if (!renderer.initialize()) {
        rtr::core::Logger::warn("MPSSample", "MPS renderer unavailable on this device");
        return 0;
    }

    const char* outputPath = "mps_output.ppm";
    if (renderer.renderFrame(outputPath)) {
        rtr::core::Logger::info("MPSSample", "Frame written to %s", outputPath);
    }

    std::cout << "MPS sample executed successfully." << std::endl;
    return 0;
}
