#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rtr::core {

struct ImageData {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> pixels;  // RGBA32F
};

class ImageLoader {
public:
    static bool loadRGBA32F(const std::filesystem::path& path, ImageData& outImage);
};

}  // namespace rtr::core
