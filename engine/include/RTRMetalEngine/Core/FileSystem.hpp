#pragma once

#include <filesystem>
#include <string>

namespace rtr::core {

class FileSystem {
public:
    static bool exists(const std::filesystem::path& path);
    static std::string readTextFile(const std::filesystem::path& path);
};

}  // namespace rtr::core
