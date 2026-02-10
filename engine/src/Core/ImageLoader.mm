#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

#include "RTRMetalEngine/Core/ImageLoader.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"

#include <algorithm>
#include <cmath>

namespace rtr::core {

namespace {

constexpr std::uint32_t kMaxLoadedTextureDimension = 2048u;

void releaseCFObject(CFTypeRef object) {
    if (object) {
        CFRelease(object);
    }
}

inline float srgbToLinear(float value) {
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return powf((value + 0.055f) / 1.055f, 2.4f);
}

}  // namespace

bool ImageLoader::loadRGBA32F(const std::filesystem::path& path, ImageData& outImage) {
    outImage = {};
    if (!std::filesystem::exists(path)) {
        core::Logger::warn("ImageLoader", "Image not found: %s", path.string().c_str());
        return false;
    }

    CFStringRef pathString = CFStringCreateWithCString(kCFAllocatorDefault,
                                                       path.string().c_str(),
                                                       kCFStringEncodingUTF8);
    if (!pathString) {
        core::Logger::warn("ImageLoader", "Failed to create CFString for %s", path.string().c_str());
        return false;
    }

    CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, pathString, kCFURLPOSIXPathStyle, false);
    releaseCFObject(pathString);
    if (!url) {
        core::Logger::warn("ImageLoader", "Failed to create URL for %s", path.string().c_str());
        return false;
    }

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    releaseCFObject(url);
    if (!source) {
        core::Logger::warn("ImageLoader", "Failed to create image source for %s", path.string().c_str());
        return false;
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    releaseCFObject(source);
    if (!image) {
        core::Logger::warn("ImageLoader", "Failed to decode image %s", path.string().c_str());
        return false;
    }

    const size_t sourceWidth = CGImageGetWidth(image);
    const size_t sourceHeight = CGImageGetHeight(image);
    if (sourceWidth == 0 || sourceHeight == 0) {
        releaseCFObject(image);
        core::Logger::warn("ImageLoader", "Image %s has invalid dimensions", path.string().c_str());
        return false;
    }

    std::uint32_t targetWidth = static_cast<std::uint32_t>(sourceWidth);
    std::uint32_t targetHeight = static_cast<std::uint32_t>(sourceHeight);
    core::Logger::info("ImageLoader",
                       "Decoded %s source size=%zux%zu",
                       path.string().c_str(),
                       sourceWidth,
                       sourceHeight);
    if (targetWidth > kMaxLoadedTextureDimension || targetHeight > kMaxLoadedTextureDimension) {
        const float widthScale = static_cast<float>(kMaxLoadedTextureDimension) /
                                 static_cast<float>(std::max<std::uint32_t>(targetWidth, 1u));
        const float heightScale = static_cast<float>(kMaxLoadedTextureDimension) /
                                  static_cast<float>(std::max<std::uint32_t>(targetHeight, 1u));
        const float scale = std::min(widthScale, heightScale);
        targetWidth = std::max<std::uint32_t>(
            1u, static_cast<std::uint32_t>(std::lround(static_cast<float>(targetWidth) * scale)));
        targetHeight = std::max<std::uint32_t>(
            1u, static_cast<std::uint32_t>(std::lround(static_cast<float>(targetHeight) * scale)));
        core::Logger::info("ImageLoader",
                           "Downscaling %s from %zux%zu to %ux%u for ray tracing texture budget",
                           path.string().c_str(),
                           sourceWidth,
                           sourceHeight,
                           targetWidth,
                           targetHeight);
    }

    std::vector<uint8_t> temp(static_cast<std::size_t>(targetWidth) *
                              static_cast<std::size_t>(targetHeight) * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    const CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast | (CGBitmapInfo)kCGBitmapByteOrder32Big;
    CGContextRef context = CGBitmapContextCreate(temp.data(),
                                                 targetWidth,
                                                 targetHeight,
                                                 8,
                                                 static_cast<size_t>(targetWidth) * 4,
                                                 colorSpace,
                                                 bitmapInfo);
    releaseCFObject(colorSpace);
    if (!context) {
        releaseCFObject(image);
        core::Logger::warn("ImageLoader", "Failed to allocate bitmap context for %s", path.string().c_str());
        return false;
    }

    const CGRect rect = CGRectMake(0, 0, static_cast<CGFloat>(targetWidth), static_cast<CGFloat>(targetHeight));
    CGContextDrawImage(context, rect, image);
    releaseCFObject(image);
    releaseCFObject(context);

    outImage.width = targetWidth;
    outImage.height = targetHeight;
    const std::size_t pixelCount =
        static_cast<std::size_t>(targetWidth) * static_cast<std::size_t>(targetHeight);
    outImage.pixels.resize(pixelCount * 4);
    for (std::size_t i = 0; i < pixelCount; ++i) {
        const std::size_t base = i * 4;
        const float r = static_cast<float>(temp[base + 0]) / 255.0f;
        const float g = static_cast<float>(temp[base + 1]) / 255.0f;
        const float b = static_cast<float>(temp[base + 2]) / 255.0f;
        const float a = static_cast<float>(temp[base + 3]) / 255.0f;
        outImage.pixels[base + 0] = srgbToLinear(r);
        outImage.pixels[base + 1] = srgbToLinear(g);
        outImage.pixels[base + 2] = srgbToLinear(b);
        outImage.pixels[base + 3] = a;
    }

    return true;
}

}  // namespace rtr::core
