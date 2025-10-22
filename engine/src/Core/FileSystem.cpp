#include "RTRMetalEngine/Core/FileSystem.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rtr::core {

bool FileSystem::exists(const std::filesystem::path& path) { return std::filesystem::exists(path); }

std::string FileSystem::readTextFile(const std::filesystem::path& path) {
    if (!exists(path)) {
        throw std::runtime_error("File does not exist: " + path.string());
    }

    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

}  // namespace rtr::core
