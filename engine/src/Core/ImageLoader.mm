#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

#include "RTRMetalEngine/Core/ImageLoader.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"

namespace rtr::core {

namespace {

void releaseCFObject(CFTypeRef object) {
    if (object) {
        CFRelease(object);
    }
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

    const size_t width = CGImageGetWidth(image);
    const size_t height = CGImageGetHeight(image);
    if (width == 0 || height == 0) {
        releaseCFObject(image);
        core::Logger::warn("ImageLoader", "Image %s has invalid dimensions", path.string().c_str());
        return false;
    }

    std::vector<uint8_t> temp(width * height * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    const CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast | (CGBitmapInfo)kCGBitmapByteOrder32Big;
    CGContextRef context = CGBitmapContextCreate(temp.data(),
                                                 width,
                                                 height,
                                                 8,
                                                 width * 4,
                                                 colorSpace,
                                                 bitmapInfo);
    releaseCFObject(colorSpace);
    if (!context) {
        releaseCFObject(image);
        core::Logger::warn("ImageLoader", "Failed to allocate bitmap context for %s", path.string().c_str());
        return false;
    }

    const CGRect rect = CGRectMake(0, 0, static_cast<CGFloat>(width), static_cast<CGFloat>(height));
    CGContextDrawImage(context, rect, image);
    releaseCFObject(image);
    releaseCFObject(context);

    outImage.width = static_cast<std::uint32_t>(width);
    outImage.height = static_cast<std::uint32_t>(height);
    outImage.pixels.resize(width * height * 4);
    for (std::size_t i = 0; i < temp.size(); ++i) {
        outImage.pixels[i] = static_cast<float>(temp[i]) / 255.0f;
    }

    return true;
}

}  // namespace rtr::core
