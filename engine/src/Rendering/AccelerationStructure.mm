#import <Metal/Metal.h>
#import <CoreFoundation/CoreFoundation.h>

#include "RTRMetalEngine/Rendering/AccelerationStructure.hpp"

namespace rtr::rendering {

class AccelerationStructureImpl {
public:
    explicit AccelerationStructureImpl(void* retainedStructureHandle)
        : structureHandle_(retainedStructureHandle) {}

    ~AccelerationStructureImpl() {
        if (structureHandle_ != nullptr) {
            CFRelease(structureHandle_);
            structureHandle_ = nullptr;
        }
    }

    AccelerationStructureImpl(const AccelerationStructureImpl&) = delete;
    AccelerationStructureImpl& operator=(const AccelerationStructureImpl&) = delete;

    AccelerationStructureImpl(AccelerationStructureImpl&& other) noexcept
        : structureHandle_(other.structureHandle_) {
        other.structureHandle_ = nullptr;
    }

    AccelerationStructureImpl& operator=(AccelerationStructureImpl&& other) noexcept {
        if (this != &other) {
            if (structureHandle_ != nullptr) {
                CFRelease(structureHandle_);
            }
            structureHandle_ = other.structureHandle_;
            other.structureHandle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] id<MTLAccelerationStructure> structure() const noexcept {
        return structureHandle_ != nullptr ? (__bridge id<MTLAccelerationStructure>)structureHandle_ : nil;
    }

    void* structureHandle_ = nullptr;
};

AccelerationStructure::AccelerationStructure() = default;
AccelerationStructure::~AccelerationStructure() = default;

AccelerationStructure::AccelerationStructure(std::string label,
                                             std::size_t size,
                                             std::unique_ptr<AccelerationStructureImpl> impl)
    : label_(std::move(label)), sizeInBytes_(size), impl_(std::move(impl)) {}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept = default;
AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&& other) noexcept = default;

bool AccelerationStructure::isValid() const noexcept {
    return impl_ && impl_->structure() != nil;
}

void* AccelerationStructure::rawHandle() const noexcept {
    return impl_ ? impl_->structureHandle_ : nullptr;
}

AccelerationStructure makeAccelerationStructure(std::string label, std::size_t size, void* handle) {
    auto impl = std::make_unique<AccelerationStructureImpl>(handle);
    return AccelerationStructure(std::move(label), size, std::move(impl));
}

}  // namespace rtr::rendering
