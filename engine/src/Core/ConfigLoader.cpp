#include "RTRMetalEngine/Core/ConfigLoader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "RTRMetalEngine/Core/FileSystem.hpp"

namespace rtr::core {
namespace {

std::string trim(const std::string& value) {
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };

    const auto begin = std::find_if_not(value.begin(), value.end(), isSpace);
    if (begin == value.end()) {
        return {};
    }

    const auto end = std::find_if_not(value.rbegin(), value.rend(), isSpace).base();
    return std::string(begin, end);
}

std::unordered_map<std::string, std::string> parseKeyValuePairs(const std::string& content) {
    std::unordered_map<std::string, std::string> pairs;
    std::istringstream stream(content);
    std::string line;

    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        auto commentStart = line.find_first_of("#;");
        if (commentStart != std::string::npos) {
            line = line.substr(0, commentStart);
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error("Invalid configuration line: " + line);
        }

        std::string key = trim(line.substr(0, separator));
        std::string value = trim(line.substr(separator + 1));

        if (key.empty()) {
            throw std::runtime_error("Configuration key cannot be empty");
        }

        pairs.emplace(std::move(key), std::move(value));
    }

    return pairs;
}

bool parseBool(const std::string& value, bool defaultValue = false) {
    if (value.empty()) {
        return defaultValue;
    }

    std::string lowered;
    lowered.resize(value.size());
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lowered == "1" || lowered == "true" || lowered == "on" || lowered == "yes") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "off" || lowered == "no") {
        return false;
    }
    return defaultValue;
}

std::uint32_t parseUInt(const std::string& value, std::uint32_t defaultValue = 0) {
    if (value.empty()) {
        return defaultValue;
    }

    try {
        const unsigned long parsed = std::stoul(value);
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
            return defaultValue;
        }
        return static_cast<std::uint32_t>(parsed);
    } catch (const std::exception&) {
        return defaultValue;
    }
}

}  // namespace

EngineConfig ConfigLoader::loadEngineConfig(const std::filesystem::path& path) {
    const std::string content = FileSystem::readTextFile(path);
    const auto pairs = parseKeyValuePairs(content);

    EngineConfig config{};
    config.shadingMode = "auto";

    if (auto it = pairs.find("applicationName"); it != pairs.end()) {
        config.applicationName = it->second;
    } else {
        throw std::runtime_error("Missing required key: applicationName");
    }

    if (auto it = pairs.find("shaderLibraryPath"); it != pairs.end()) {
        config.shaderLibraryPath = it->second;
    } else {
        config.shaderLibraryPath = "shaders/RTRShaders.metallib";
    }

    if (auto it = pairs.find("shadingMode"); it != pairs.end()) {
        config.shadingMode = it->second;
    }

    if (auto it = pairs.find("accumulation"); it != pairs.end()) {
        config.accumulationEnabled = parseBool(it->second, config.accumulationEnabled);
    }

    if (auto it = pairs.find("accumulationFrames"); it != pairs.end()) {
        config.accumulationFrames = parseUInt(it->second, config.accumulationFrames);
    }

    if (auto it = pairs.find("samplesPerPixel"); it != pairs.end()) {
        config.samplesPerPixel = parseUInt(it->second, config.samplesPerPixel);
    }

    if (auto it = pairs.find("sampleSeed"); it != pairs.end()) {
        config.sampleSeed = parseUInt(it->second, config.sampleSeed);
    }

    if (auto it = pairs.find("maxBounces"); it != pairs.end()) {
        config.maxHardwareBounces = parseUInt(it->second, config.maxHardwareBounces);
    }

    if (config.maxHardwareBounces == 0) {
        config.maxHardwareBounces = 1;
    }

    return config;
}

}  // namespace rtr::core
