#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

#include <cstring>
#include <utility>

namespace rtr::rendering {

class BufferHandle::Impl {
public:
    Impl(id<MTLBuffer> buffer, std::size_t length)
        : buffer_(buffer), length_(length) {}

    ~Impl() { buffer_ = nil; }

    id<MTLBuffer> buffer_ = nil;
    std::size_t length_ = 0;
};

BufferHandle::BufferHandle() = default;
BufferHandle::~BufferHandle() = default;
BufferHandle::BufferHandle(BufferHandle&&) noexcept = default;
BufferHandle& BufferHandle::operator=(BufferHandle&&) noexcept = default;

BufferHandle::BufferHandle(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

bool BufferHandle::isValid() const noexcept { return impl_ && impl_->buffer_ != nil; }

std::size_t BufferHandle::length() const noexcept { return impl_ ? impl_->length_ : 0; }

void* BufferHandle::nativeHandle() const noexcept {
    return impl_ && impl_->buffer_ ? (__bridge void*)impl_->buffer_ : nullptr;
}

void BufferHandle::adopt(void* bufferHandle, std::size_t length) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferHandle;
    impl_ = std::make_unique<Impl>(buffer, length);
}

BufferAllocator::BufferAllocator(MetalContext& context)
    : context_(context) {}

bool BufferAllocator::isDeviceAvailable() const noexcept { return context_.isValid(); }

BufferHandle BufferAllocator::createBuffer(std::size_t length, const void* initialData, const char* label) noexcept {
    if (length == 0) {
        core::Logger::warn("BufferAllocator", "Cannot create buffer of zero length");
        return BufferHandle{};
    }

    if (!context_.isValid()) {
        core::Logger::warn("BufferAllocator", "Metal device unavailable; buffer allocation skipped");
        return BufferHandle{};
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_.rawDeviceHandle();
    if (!device) {
        core::Logger::warn("BufferAllocator", "Metal device unavailable; buffer allocation skipped");
        return BufferHandle{};
    }

    @autoreleasepool {
        id<MTLBuffer> buffer = nil;
        buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (!buffer) {
            core::Logger::error("BufferAllocator", "Failed to create Metal buffer of length %zu", length);
            return BufferHandle{};
        }

        if (label && label[0] != '\0') {
            NSString* nsLabel = [[NSString alloc] initWithUTF8String:label];
            if (nsLabel) {
                [buffer setLabel:nsLabel];
            }
        }

        if (initialData) {
            std::memcpy([buffer contents], initialData, length);
            [buffer didModifyRange:NSMakeRange(0, length)];
        }

        return BufferHandle(std::make_unique<BufferHandle::Impl>(buffer, length));
    }
}

}  // namespace rtr::rendering
