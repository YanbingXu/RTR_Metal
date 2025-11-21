#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "RTRMetalEngine/Core/ConfigLoader.hpp"
#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Core/FileSystem.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/Renderer.hpp"
#include "RTRMetalEngine/Scene/CornellBox.hpp"
#include "RTRMetalEngine/Scene/DemoScenes.hpp"

namespace fs = std::filesystem;

namespace {

struct CommandLineOptions {
    fs::path configPath{"config/engine.ini"};
    fs::path assetRoot{"assets"};
    fs::path outputPath{"renderer_output.ppm"};
    std::string sceneName{"cornell"};
    std::string shadingMode;
    std::uint32_t width = 512;
    std::uint32_t height = 512;
    std::uint32_t frames = 1;
    bool printHash = false;
    bool debugAlbedo = false;
    bool overrideShadingMode = false;
    std::optional<bool> accumulationEnabled;
    std::optional<std::uint32_t> accumulationFrames;
    std::optional<std::uint32_t> samplesPerPixel;
    std::optional<std::uint32_t> sampleSeed;
    std::optional<std::uint32_t> maxBounces;
};

void printUsage() {
    std::cout << "Usage: RTRMetalSample [options]\n"
              << "  --output=<file>        输出图像路径 (默认 renderer_output.ppm)\n"
              << "  --scene=<name>         场景: cornell|reflective|glass (默认 cornell)\n"
              << "  --asset-root=<path>    资产目录，用于 reflective/glass 场景 (默认 assets)\n"
              << "  --resolution=WxH       渲染分辨率，例 1280x720 (默认 512x512)\n"
              << "  --frames=N             渲染帧数，用于累计或调试 (默认 1)\n"
              << "  --mode=auto|hardware|fallback  选择硬件 RT 或渐变回退 (默认 auto)\n"
              << "  --config=<file>        配置文件路径 (默认 config/engine.ini)\n"
              << "  --accumulation=on|off  开启或关闭累计 (覆盖配置文件)\n"
              << "  --accumulation-frames=N 限定累计帧数 (0 表示无限制)\n"
              << "  --samples-per-pixel=N  每像素采样次数，0 表示无限制 (默认 1)\n"
              << "  --sample-seed=N        采样随机种子\n"
              << "  --max-bounces=N        硬件 RT 最大弹射次数 (至少 1)\n"
              << "  --hash                 渲染完输出图像的 FNV-1a hash\n"
              << "  --debug-albedo         调试模式：直接输出材质反照率\n"
              << "  --help                 打印帮助\n";
}

std::optional<std::pair<std::uint32_t, std::uint32_t>> parseResolution(const std::string& value) {
    const auto delimiter = value.find_first_of("xX");
    if (delimiter == std::string::npos) {
        return std::nullopt;
    }

    const std::string widthStr = value.substr(0, delimiter);
    const std::string heightStr = value.substr(delimiter + 1);

    try {
        const unsigned long width = std::stoul(widthStr);
        const unsigned long height = std::stoul(heightStr);
        if (width == 0 || height == 0 || width > std::numeric_limits<std::uint32_t>::max() ||
            height > std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        return std::make_pair(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

bool parseBoolArgument(std::string value, const char* optionName) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (value == "1" || value == "true" || value == "on" || value == "yes") {
        return true;
    }
    if (value == "0" || value == "false" || value == "off" || value == "no") {
        return false;
    }
    throw std::runtime_error(std::string("Invalid value for ") + optionName + ": " + value);
}

std::uint32_t parseUIntArgument(const std::string& text, const char* optionName) {
    try {
        const unsigned long parsed = std::stoul(text);
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Value out of range for " + std::string(optionName));
        }
        return static_cast<std::uint32_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric value for " + std::string(optionName) + ": " + text);
    }
}

CommandLineOptions parseOptions(int argc, const char* const* argv) {
    CommandLineOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            printUsage();
            std::exit(0);
        } else if (arg.rfind("--output=", 0) == 0) {
            options.outputPath = fs::path(arg.substr(9));
        } else if (arg.rfind("--scene=", 0) == 0) {
            options.sceneName = arg.substr(8);
        } else if (arg.rfind("--asset-root=", 0) == 0) {
            options.assetRoot = fs::path(arg.substr(13));
        } else if (arg.rfind("--resolution=", 0) == 0) {
            const auto parsed = parseResolution(arg.substr(13));
            if (!parsed.has_value()) {
                throw std::runtime_error("Invalid resolution format. Expected WxH.");
            }
            options.width = parsed->first;
            options.height = parsed->second;
        } else if (arg.rfind("--mode=", 0) == 0) {
            options.shadingMode = arg.substr(7);
            options.overrideShadingMode = true;
        } else if (arg.rfind("--frames=", 0) == 0) {
            try {
                const unsigned long frames = std::stoul(arg.substr(9));
                if (frames == 0 || frames > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("frames value out of range");
                }
                options.frames = static_cast<std::uint32_t>(frames);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid frames value");
            }
        } else if (arg == "--hash") {
            options.printHash = true;
        } else if (arg == "--debug-albedo") {
            options.debugAlbedo = true;
        } else if (arg.rfind("--config=", 0) == 0) {
            options.configPath = fs::path(arg.substr(9));
        } else if (arg.rfind("--accumulation=", 0) == 0) {
            options.accumulationEnabled = parseBoolArgument(arg.substr(15), "--accumulation");
        } else if (arg == "--accumulation") {
            options.accumulationEnabled = true;
        } else if (arg == "--no-accumulation") {
            options.accumulationEnabled = false;
        } else if (arg.rfind("--accumulation-frames=", 0) == 0) {
            options.accumulationFrames = parseUIntArgument(arg.substr(22), "--accumulation-frames");
        } else if (arg.rfind("--samples-per-pixel=", 0) == 0) {
            options.samplesPerPixel = parseUIntArgument(arg.substr(22), "--samples-per-pixel");
        } else if (arg.rfind("--sample-seed=", 0) == 0) {
            options.sampleSeed = parseUIntArgument(arg.substr(14), "--sample-seed");
        } else if (arg.rfind("--max-bounces=", 0) == 0) {
            const auto value = parseUIntArgument(arg.substr(14), "--max-bounces");
            if (value == 0) {
                throw std::runtime_error("--max-bounces must be >= 1");
            }
            options.maxBounces = value;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    return options;
}

rtr::scene::Scene buildScene(const std::string& sceneName, const fs::path& assetRoot) {
    const std::string lower = [&]() {
        std::string s = sceneName;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }();

    if (lower == "cornell") {
        return rtr::scene::createCornellBoxScene(assetRoot);
    }
    if (lower == "reflective") {
        rtr::core::Logger::info("Sample", "Loading reflective scene from %s", assetRoot.string().c_str());
        return rtr::scene::createReflectiveDemoScene(assetRoot);
    }
    if (lower == "glass") {
        rtr::core::Logger::info("Sample", "Loading glass scene from %s", assetRoot.string().c_str());
        return rtr::scene::createGlassDemoScene(assetRoot);
    }

    rtr::core::Logger::warn("Sample", "Unknown scene '%s', falling back to Cornell Box", sceneName.c_str());
    return rtr::scene::createCornellBoxScene();
}

std::uint64_t computeFNVHash(const fs::path& path) {
    constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
    constexpr std::uint64_t kFNVPrime = 1099511628211ull;

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file for hashing: " + path.string());
    }

    std::uint64_t hash = kFNVOffset;
    std::vector<char> buffer(4096);
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize readBytes = file.gcount();
        for (std::streamsize i = 0; i < readBytes; ++i) {
            hash ^= static_cast<std::uint8_t>(buffer[i]);
            hash *= kFNVPrime;
        }
    }
    return hash;
}

rtr::core::EngineConfig loadEngineConfig(const fs::path& configPath) {
    rtr::core::EngineConfig config{};
    if (rtr::core::FileSystem::exists(configPath)) {
        config = rtr::core::ConfigLoader::loadEngineConfig(configPath);
    } else {
        config.applicationName = "RTR Metal Sample";
        config.shaderLibraryPath = "shaders/RTRShaders.metallib";
        config.shadingMode = "auto";
        rtr::core::Logger::warn("Sample", "Config file not found at %s, using defaults", configPath.string().c_str());
    }
    return config;
}

}  // namespace

int main(int argc, const char* argv[]) {
    CommandLineOptions options{};
    try {
        options = parseOptions(argc, argv);
    } catch (const std::exception& ex) {
        rtr::core::Logger::error("Sample", "%s", ex.what());
        printUsage();
        return 1;
    }

    if (!options.assetRoot.empty()) {
        std::error_code ec;
        fs::path absoluteRoot = fs::absolute(options.assetRoot, ec);
        if (!ec) {
            options.assetRoot = absoluteRoot;
        }
        if (!fs::exists(options.assetRoot)) {
            rtr::core::Logger::warn("Sample",
                                    "Asset root '%s' does not exist; reflective/glass scenes may fall back",
                                    options.assetRoot.string().c_str());
        }
    }

    rtr::core::EngineConfig config{};
    try {
        config = loadEngineConfig(options.configPath);
    } catch (const std::exception& ex) {
        rtr::core::Logger::error("Sample", "Failed to load config: %s", ex.what());
        return 1;
    }

    if (options.overrideShadingMode) {
        config.shadingMode = options.shadingMode;
    }

    if (options.accumulationEnabled.has_value()) {
        config.accumulationEnabled = *options.accumulationEnabled;
    }
    if (options.accumulationFrames.has_value()) {
        config.accumulationFrames = *options.accumulationFrames;
    }
    if (options.samplesPerPixel.has_value()) {
        config.samplesPerPixel = *options.samplesPerPixel;
    }
    if (options.sampleSeed.has_value()) {
        config.sampleSeed = *options.sampleSeed;
    }
    if (options.maxBounces.has_value()) {
        config.maxHardwareBounces = *options.maxBounces;
    }
    if (config.maxHardwareBounces == 0) {
        config.maxHardwareBounces = 1;
    }

    rtr::rendering::Renderer renderer{config};
    renderer.setOutputPath(options.outputPath.string());
    renderer.setRenderSize(options.width, options.height);
    renderer.setDebugMode(options.debugAlbedo);

    rtr::scene::Scene scene = buildScene(options.sceneName, options.assetRoot);
    if (!renderer.loadScene(scene)) {
        rtr::core::Logger::error("Sample", "Failed to load requested scene");
        return 1;
    }

    std::string shadingMode = config.shadingMode;
    std::transform(shadingMode.begin(), shadingMode.end(), shadingMode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    const bool expectHardware = shadingMode != "cpu" && shadingMode != "fallback" && shadingMode != "gradient";

    if (expectHardware && !renderer.isRayTracingReady()) {
        rtr::core::Logger::warn("Sample", "Ray tracing pipeline not ready; output will use fallback gradient");
    }

    for (std::uint32_t i = 0; i < options.frames; ++i) {
        renderer.renderFrame();
    }

    std::cout << "Rendered " << options.frames << " frame(s) to " << options.outputPath << std::endl;

    if (options.printHash) {
        try {
            const auto hash = computeFNVHash(options.outputPath);
            std::cout << "FNV-1a hash: 0x" << std::hex << std::uppercase << hash << std::dec << std::nouppercase << std::endl;
        } catch (const std::exception& ex) {
            rtr::core::Logger::warn("Sample", "Hash computation failed: %s", ex.what());
        }
    }

    return 0;
}
