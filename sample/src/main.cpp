#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "RTRMetalEngine/Core/ConfigLoader.hpp"
#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Core/FileSystem.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/Renderer.hpp"
#include "SampleAppUtils.hpp"

namespace fs = std::filesystem;

#ifndef RTR_SOURCE_DIR
#define RTR_SOURCE_DIR ""
#endif
#ifndef RTR_BINARY_DIR
#define RTR_BINARY_DIR ""
#endif

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
    std::optional<std::string> expectedHash;
    bool overrideShadingMode = false;
    std::optional<std::uint32_t> maxBounces;
    std::optional<std::string> debugVisualization;
    bool debugSceneDump = false;
    bool debugGeometryTrace = false;
    bool debugTlasTrace = false;
    bool debugCameraTrace = false;
    bool debugIsolateCornellExtras = false;
    std::optional<std::uint32_t> debugIsolateCornellMeshIndex;
};

void printUsage() {
    std::cout << "Usage: RTRMetalSample [options]\n"
              << "  --output=<file>        输出图像路径 (默认 renderer_output.ppm)\n"
              << "  --scene=<name>         场景: cornell|reflective|glass (默认 cornell)\n"
              << "  --asset-root=<path>    资产目录，用于 reflective/glass 场景 (默认 assets)\n"
              << "  --resolution=WxH       渲染分辨率，例 1280x720 (默认 512x512)\n"
              << "  --frames=N             渲染帧数，用于累计或调试 (默认 1)\n"
              << "  --mode=auto|hardware          选择硬件模式（默认 auto，与硬件模式一致）\n"
              << "  --config=<file>        配置文件路径 (默认 config/engine.ini)\n"
              << "  --max-bounces=N        硬件 RT 最大弹射次数 (至少 1)\n"
              << "  --hash                 渲染完输出图像的 FNV-1a hash\n"
              << "  --expect-hash=0xHASH  计算图像 hash 并与给定值比对\n"
              << "  --debug-albedo         调试模式：直接输出材质反照率 (等价于 --debug-visualization=albedo)\n"
              << "  --debug-visualization=none|albedo|instance-colors|instance-trace|primitive-trace\n"
              << "                         选择交互式可视化模式\n"
              << "  --debug-log=scene|geometry|tlas|camera[,..]\n"
              << "                         启用附加日志/追踪 (可重复)\n"
              << "  --debug-isolate-extras 仅保留 Cornell 的扩展几何（mesh 6/7/8）参与 TLAS\n"
              << "  --debug-isolate-mesh=N      仅保留指定 Cornell mesh（需配合 --debug-isolate-extras）\n"
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

std::string normalizeFlag(std::string value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (char ch : value) {
        if (ch == '-' || ch == '_' || ch == ' ') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return normalized;
}

void applyDebugLogFlag(CommandLineOptions& options, const std::string& value) {
    const std::string flag = normalizeFlag(value);
    if (flag == "scene" || flag == "scenedump") {
        options.debugSceneDump = true;
    } else if (flag == "geometry" || flag == "geometrytrace") {
        options.debugGeometryTrace = true;
    } else if (flag == "tlas" || flag == "tlastrace") {
        options.debugTlasTrace = true;
    } else if (flag == "camera" || flag == "cameratrace") {
        options.debugCameraTrace = true;
    } else {
        throw std::runtime_error("Unknown --debug-log flag: " + value);
    }
}

void applyDebugLogList(CommandLineOptions& options, const std::string& list) {
    if (list.empty()) {
        throw std::runtime_error("--debug-log requires at least one flag");
    }
    std::stringstream stream(list);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            applyDebugLogFlag(options, item);
        }
    }
}

rtr::rendering::DebugVisualization parseDebugVisualization(std::string value) {
    const std::string key = normalizeFlag(value);
    if (key.empty() || key == "none") {
        return rtr::rendering::DebugVisualization::None;
    }
    if (key == "albedo" || key == "debugalbedo") {
        return rtr::rendering::DebugVisualization::Albedo;
    }
    if (key == "instancecolors" || key == "meshcolors") {
        return rtr::rendering::DebugVisualization::InstanceColors;
    }
    if (key == "instancetrace") {
        return rtr::rendering::DebugVisualization::InstanceTrace;
    }
    if (key == "primitivetrace") {
        return rtr::rendering::DebugVisualization::PrimitiveTrace;
    }
    throw std::runtime_error("Unknown debug visualization: " + value);
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
        } else if (arg.rfind("--expect-hash=", 0) == 0) {
            options.printHash = true;
            options.expectedHash = arg.substr(14);
        } else if (arg == "--debug-albedo") {
            options.debugVisualization = "albedo";
        } else if (arg.rfind("--config=", 0) == 0) {
            options.configPath = fs::path(arg.substr(9));
        } else if (arg.rfind("--max-bounces=", 0) == 0) {
            const auto value = parseUIntArgument(arg.substr(14), "--max-bounces");
            if (value == 0) {
                throw std::runtime_error("--max-bounces must be >= 1");
            }
            options.maxBounces = value;
        } else if (arg.rfind("--debug-visualization=", 0) == 0) {
            options.debugVisualization = arg.substr(22);
        } else if (arg.rfind("--debug-log=", 0) == 0) {
            applyDebugLogList(options, arg.substr(12));
        } else if (arg == "--debug-isolate-extras") {
            options.debugIsolateCornellExtras = true;
        } else if (arg.rfind("--debug-isolate-mesh=", 0) == 0) {
            const std::uint32_t meshIndex = parseUIntArgument(arg.substr(21), "--debug-isolate-mesh");
            options.debugIsolateCornellMeshIndex = meshIndex;
            options.debugIsolateCornellExtras = true;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    return options;
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

}  // namespace

int main(int argc, const char* argv[]) {
    const fs::path sourceRoot = fs::path(RTR_SOURCE_DIR);
    const fs::path binaryRoot = fs::path(RTR_BINARY_DIR);
    const fs::path currentDir = fs::current_path();

    auto resolveWithDefaults = [&](const fs::path& candidate,
                                   bool requireDirectory,
                                   std::initializer_list<fs::path> preferredBases = {}) {
        std::vector<fs::path> bases;
        bases.reserve(preferredBases.size() + 3);
        bases.insert(bases.end(), preferredBases.begin(), preferredBases.end());
        bases.push_back(currentDir);
        bases.push_back(binaryRoot);
        bases.push_back(sourceRoot);
        return rtr::sample::resolvePath(candidate, requireDirectory, bases);
    };
    CommandLineOptions options{};
    try {
        options = parseOptions(argc, argv);
    } catch (const std::exception& ex) {
        rtr::core::Logger::error("Sample", "%s", ex.what());
        printUsage();
        return 1;
    }

    if (!options.assetRoot.empty()) {
        if (auto resolvedAssetRoot = resolveWithDefaults(options.assetRoot, true)) {
            options.assetRoot = *resolvedAssetRoot;
            rtr::core::Logger::info("Sample", "Using asset root: %s", options.assetRoot.string().c_str());
        } else {
            rtr::core::Logger::warn("Sample",
                                    "Asset root '%s' does not exist; reflective/glass scenes may fall back",
                                    options.assetRoot.string().c_str());
        }
    }

    rtr::core::EngineConfig config{};
    const auto configCandidate = resolveWithDefaults(options.configPath, false);
    fs::path resolvedConfigPath = configCandidate.value_or(options.configPath);
    fs::path configDirectory = configCandidate ? configCandidate->parent_path() : fs::path{};
    rtr::core::Logger::info("Sample", "Loading config: %s", resolvedConfigPath.string().c_str());

    try {
        config = rtr::sample::loadEngineConfig(resolvedConfigPath);
    } catch (const std::exception& ex) {
        rtr::core::Logger::error("Sample", "Failed to load config: %s", ex.what());
        return 1;
    }

    std::optional<fs::path> resolvedShaderLibrary;
    if (!configDirectory.empty()) {
        resolvedShaderLibrary = resolveWithDefaults(config.shaderLibraryPath, false, {configDirectory});
    } else {
        resolvedShaderLibrary = resolveWithDefaults(config.shaderLibraryPath, false);
    }
    if (resolvedShaderLibrary) {
        config.shaderLibraryPath = resolvedShaderLibrary->string();
        rtr::core::Logger::info("Sample", "Using shader library: %s", config.shaderLibraryPath.c_str());
    } else {
        rtr::core::Logger::warn("Sample",
                                "Shader library '%s' not found relative to working/binary/source/config directories",
                                config.shaderLibraryPath.c_str());
    }

    if (options.overrideShadingMode) {
        config.shadingMode = options.shadingMode;
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

    rtr::rendering::RendererDebugOptions debugOptionsState = renderer.debugOptions();
    bool applyDebugOptions = false;
    if (options.debugVisualization.has_value()) {
        debugOptionsState.visualization = parseDebugVisualization(*options.debugVisualization);
        applyDebugOptions = true;
    }
    if (options.debugSceneDump) {
        debugOptionsState.sceneDump = true;
        applyDebugOptions = true;
    }
    if (options.debugGeometryTrace) {
        debugOptionsState.geometryTrace = true;
        applyDebugOptions = true;
    }
    if (options.debugTlasTrace) {
        debugOptionsState.tlasTrace = true;
        applyDebugOptions = true;
    }
    if (options.debugCameraTrace) {
        debugOptionsState.cameraTrace = true;
        applyDebugOptions = true;
    }
    if (options.debugIsolateCornellExtras) {
        debugOptionsState.isolateCornellExtras = true;
        debugOptionsState.isolateCornellMeshIndex = options.debugIsolateCornellMeshIndex;
        applyDebugOptions = true;
    }
    if (applyDebugOptions) {
        renderer.setDebugOptions(debugOptionsState);
    }

    rtr::scene::Scene scene = rtr::sample::buildScene(options.sceneName, options.assetRoot);
    if (!renderer.loadScene(scene)) {
        rtr::core::Logger::error("Sample", "Failed to load requested scene");
        return 1;
    }

    std::string shadingMode = config.shadingMode;
    std::transform(shadingMode.begin(), shadingMode.end(), shadingMode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (!renderer.isRayTracingReady()) {
        rtr::core::Logger::error("Sample", "Hardware ray tracing pipeline is not ready on this device");
        return 1;
    }

    for (std::uint32_t i = 0; i < options.frames; ++i) {
        renderer.renderFrame();
    }

    std::cout << "Rendered " << options.frames << " frame(s) to " << options.outputPath << std::endl;

    auto normalizeHash = [](std::string value) {
        if (value.rfind("0x", 0) == 0 || value.rfind("0X", 0) == 0) {
            value = value.substr(2);
        }
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        return value;
    };

    if (options.printHash || options.expectedHash.has_value()) {
        try {
            const auto hash = computeFNVHash(options.outputPath);
            std::cout << "FNV-1a hash: 0x" << std::hex << std::uppercase << hash << std::dec << std::nouppercase << std::endl;
            if (options.expectedHash.has_value()) {
                const std::string expected = normalizeHash(*options.expectedHash);
                std::ostringstream stream;
                stream << std::hex << std::uppercase << hash;
                const std::string actual = stream.str();
                if (actual != expected) {
                    rtr::core::Logger::error("Sample",
                                              "Hash mismatch: expected 0x%s, got 0x%s",
                                              expected.c_str(),
                                              actual.c_str());
                    return 1;
                }
            }
        } catch (const std::exception& ex) {
            rtr::core::Logger::warn("Sample", "Hash computation failed: %s", ex.what());
            if (options.expectedHash.has_value()) {
                return 1;
            }
        }
    }

    return 0;
}
