#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include "RTRMetalEngine/Core/ConfigLoader.hpp"
#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Core/FileSystem.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/MPS/MPSRenderer.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

namespace {

struct CommandLineOptions {
    std::string outputPath = "mps_output.ppm";
    std::optional<std::string> cpuOutputPath;
    std::optional<std::string> gpuOutputPath;
    bool runComparison = false;
    std::optional<rtr::rendering::MPSRenderer::ShadingMode> requestedMode;
};

void printUsage() {
    std::cout << "Usage: RTRMetalMPSSample [options]\n"
              << "  --cpu               Force CPU shading only\n"
              << "  --gpu               Prefer GPU shading (default)\n"
              << "  --compare           Render CPU and GPU outputs together\n"
              << "  --output=<file>     Set output path when not comparing\n"
              << "  --cpu-output=<file> Output path for CPU image in compare mode\n"
              << "  --gpu-output=<file> Output path for GPU image in compare mode\n"
              << std::endl;
}

CommandLineOptions parseOptions(int argc, const char* argv[]) {
    CommandLineOptions options;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            printUsage();
            std::exit(0);
        } else if (arg == "--cpu") {
            options.requestedMode = rtr::rendering::MPSRenderer::ShadingMode::CpuOnly;
        } else if (arg == "--gpu") {
            options.requestedMode = rtr::rendering::MPSRenderer::ShadingMode::GpuPreferred;
        } else if (arg == "--compare") {
            options.runComparison = true;
        } else if (arg.rfind("--output=", 0) == 0) {
            options.outputPath = arg.substr(9);
        } else if (arg.rfind("--cpu-output=", 0) == 0) {
            options.cpuOutputPath = arg.substr(13);
        } else if (arg.rfind("--gpu-output=", 0) == 0) {
            options.gpuOutputPath = arg.substr(13);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage();
            std::exit(1);
        }
    }

    if (options.runComparison) {
        if (!options.cpuOutputPath.has_value()) {
            options.cpuOutputPath = "mps_output_cpu.ppm";
        }
        if (!options.gpuOutputPath.has_value()) {
            options.gpuOutputPath = "mps_output_gpu.ppm";
        }
    }

    return options;
}

static rtr::rendering::MPSRenderer::ShadingMode shadingModeFromString(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "cpu") {
        return rtr::rendering::MPSRenderer::ShadingMode::CpuOnly;
    }
    if (value == "gpu") {
        return rtr::rendering::MPSRenderer::ShadingMode::GpuPreferred;
    }
    return rtr::rendering::MPSRenderer::ShadingMode::Auto;
}

}  // namespace

int main(int argc, const char* argv[]) {
    const CommandLineOptions options = parseOptions(argc, argv);

    rtr::core::EngineConfig config{};
    const std::filesystem::path configPath = "config/engine.ini";
    if (rtr::core::FileSystem::exists(configPath)) {
        config = rtr::core::ConfigLoader::loadEngineConfig(configPath);
    } else {
        config.applicationName = "RTR Metal MPS Sample";
        config.shaderLibraryPath = "shaders/RTRShaders.metallib";
        config.shadingMode = "auto";
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

    const auto configuredMode = shadingModeFromString(config.shadingMode);
    renderer.setShadingMode(options.requestedMode.value_or(configuredMode));

    if (options.runComparison) {
        rtr::rendering::MPSRenderer::FrameComparison comparison;
        const char* cpuPath = options.cpuOutputPath->c_str();
        const char* gpuPath = options.gpuOutputPath->c_str();
        if (!renderer.renderFrameComparison(cpuPath, gpuPath, &comparison)) {
            rtr::core::Logger::error("MPSSample", "Failed to render comparison frame");
            return 1;
        }
        rtr::core::Logger::info("MPSSample",
                                "Comparison complete: cpu=%s gpu=%s maxDiff=%.2f maxFloatDiff=%.6f cpuHash=%llu gpuHash=%llu",
                                cpuPath, gpuPath, comparison.maxByteDifference, comparison.maxFloatDifference,
                                static_cast<unsigned long long>(comparison.cpuPixelHash),
                                static_cast<unsigned long long>(comparison.gpuPixelHash));
    } else {
        if (!renderer.renderFrame(options.outputPath.c_str())) {
            rtr::core::Logger::error("MPSSample", "Failed to render frame to %s", options.outputPath.c_str());
            return 1;
        }
        rtr::core::Logger::info("MPSSample", "Frame written to %s", options.outputPath.c_str());
    }

    std::cout << "MPS sample executed successfully." << std::endl;
    return 0;
}
