#import <Metal/Metal.h>

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

#include <memory>
#include <string>

namespace rtr::rendering {

class MetalContext::Impl {
public:
    Impl();
    ~Impl();

    bool isValid() const noexcept { return device_ != nil; }
    const std::string& deviceName() const noexcept { return deviceName_; }
    bool supportsRayTracing() const noexcept { return supportsRayTracing_; }
    id<MTLDevice> device() const noexcept { return device_; }
    id<MTLCommandQueue> commandQueue() const noexcept { return commandQueue_; }

    void logDeviceInfo() const;

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> commandQueue_ = nil;
    std::string deviceName_;
    bool supportsRayTracing_ = false;
};

MetalContext::Impl::Impl() {
    @autoreleasepool {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            core::Logger::error("MetalContext", "Failed to acquire default Metal device");
            return;
        }

        deviceName_ = [[device_ name] UTF8String];
        supportsRayTracing_ = [device_ supportsRaytracing];

        commandQueue_ = [device_ newCommandQueue];
        if (!commandQueue_) {
            core::Logger::error("MetalContext", "Failed to create command queue");
            device_ = nil;
            return;
        }
    }
}

MetalContext::Impl::~Impl() {
    @autoreleasepool {
        commandQueue_ = nil;
        device_ = nil;
    }
}

void MetalContext::Impl::logDeviceInfo() const {
    if (!isValid()) {
        core::Logger::warn("MetalContext", "No Metal device available");
        return;
    }

    core::Logger::info("MetalContext", "Using device: %s (ray tracing %s)", deviceName_.c_str(),
                       supportsRayTracing_ ? "supported" : "not supported");
}

MetalContext::MetalContext()
    : impl_(std::make_unique<Impl>()) {}

MetalContext::~MetalContext() = default;

MetalContext::MetalContext(MetalContext&&) noexcept = default;
MetalContext& MetalContext::operator=(MetalContext&&) noexcept = default;

bool MetalContext::isValid() const noexcept { return impl_ && impl_->isValid(); }

const std::string& MetalContext::deviceName() const noexcept {
    static const std::string kUnknownDevice{"<unknown>"};
    return impl_ && impl_->isValid() ? impl_->deviceName() : kUnknownDevice;
}

bool MetalContext::supportsRayTracing() const noexcept {
    return impl_ && impl_->supportsRayTracing();
}

void MetalContext::logDeviceInfo() const { impl_->logDeviceInfo(); }

void* MetalContext::rawDeviceHandle() const noexcept {
    if (!impl_ || !impl_->isValid()) {
        return nullptr;
    }
    return (__bridge void*)impl_->device();
}

void* MetalContext::rawCommandQueue() const noexcept {
    if (!impl_ || !impl_->isValid()) {
        return nullptr;
    }
    return (__bridge void*)impl_->commandQueue();
}

}  // namespace rtr::rendering
