#include "SampleAppUtils.hpp"

#include <algorithm>
#include <cctype>
#include <vector>

#include "RTRMetalEngine/Core/ConfigLoader.hpp"
#include "RTRMetalEngine/Core/FileSystem.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Scene/CornellBox.hpp"
#include "RTRMetalEngine/Scene/DemoScenes.hpp"

namespace rtr::sample {

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

}  // namespace

std::optional<fs::path> resolvePath(const fs::path& candidate,
                                    bool requireDirectory,
                                    const std::vector<fs::path>& preferredBases) {
    if (candidate.empty()) {
        return std::nullopt;
    }

    auto validate = [&](const fs::path& path) -> bool {
        std::error_code ec;
        if (!fs::exists(path, ec)) {
            return false;
        }
        if (!requireDirectory) {
            return true;
        }
        return fs::is_directory(path, ec);
    };

    auto tryResolveRelative = [&](const fs::path& base) -> std::optional<fs::path> {
        if (base.empty()) {
            return std::nullopt;
        }
        std::error_code ec;
        fs::path resolved = fs::weakly_canonical(base / candidate, ec);
        if (!ec && validate(resolved)) {
            return resolved;
        }
        if (validate(base / candidate)) {
            return fs::path(base / candidate);
        }
        return std::nullopt;
    };

    if (candidate.is_absolute()) {
        std::error_code ec;
        fs::path canonical = fs::weakly_canonical(candidate, ec);
        if (!ec && validate(canonical)) {
            return canonical;
        }
        if (validate(candidate)) {
            return candidate;
        }
        return std::nullopt;
    }

    for (const auto& base : preferredBases) {
        if (auto resolved = tryResolveRelative(base)) {
            return resolved;
        }
    }

    return std::nullopt;
}

rtr::core::EngineConfig loadEngineConfig(const fs::path& configPath) {
    rtr::core::EngineConfig config{};
    if (configPath.empty()) {
        config.applicationName = "RTR Metal Sample";
        config.shaderLibraryPath = "shaders/RTRShaders.metallib";
        rtr::core::Logger::warn("Sample",
                                "Config path not provided, using default engine config values");
        return config;
    }

    if (rtr::core::FileSystem::exists(configPath)) {
        config = rtr::core::ConfigLoader::loadEngineConfig(configPath);
    } else {
        config.applicationName = "RTR Metal Sample";
        config.shaderLibraryPath = "shaders/RTRShaders.metallib";
        rtr::core::Logger::warn("Sample",
                                "Config file not found at %s, using defaults",
                                configPath.string().c_str());
    }
    return config;
}

rtr::scene::Scene buildScene(const std::string& sceneName, const fs::path& assetRoot) {
    const std::string lower = toLower(sceneName);

    if (lower == "cornell") {
        if (!assetRoot.empty()) {
            return rtr::scene::createCornellBoxScene(assetRoot);
        }
        return rtr::scene::createCornellBoxScene();
    }
    if (lower == "reflective") {
        rtr::core::Logger::info("Sample", "Loading reflective scene from %s", assetRoot.string().c_str());
        return rtr::scene::createReflectiveDemoScene(assetRoot);
    }
    if (lower == "glass") {
        rtr::core::Logger::info("Sample", "Loading glass scene from %s", assetRoot.string().c_str());
        return rtr::scene::createGlassDemoScene(assetRoot);
    }

    rtr::core::Logger::warn("Sample",
                            "Unknown scene '%s', falling back to Cornell Box",
                            sceneName.c_str());
    if (!assetRoot.empty()) {
        return rtr::scene::createCornellBoxScene(assetRoot);
    }
    return rtr::scene::createCornellBoxScene();
}

}  // namespace rtr::sample
