#import <Metal/Metal.h>

#include "RTRMetalEngine/Rendering/AccelerationStructure.hpp"

namespace rtr::rendering {

class AccelerationStructureImpl {
public:
    explicit AccelerationStructureImpl(id<MTLAccelerationStructure> structure)
        : structure_(structure) {}

    ~AccelerationStructureImpl() { structure_ = nil; }

    AccelerationStructureImpl(const AccelerationStructureImpl&) = delete;
    AccelerationStructureImpl& operator=(const AccelerationStructureImpl&) = delete;

    AccelerationStructureImpl(AccelerationStructureImpl&& other) noexcept
        : structure_(other.structure_) {
        other.structure_ = nil;
    }

    AccelerationStructureImpl& operator=(AccelerationStructureImpl&& other) noexcept {
        if (this != &other) {
            structure_ = other.structure_;
            other.structure_ = nil;
        }
        return *this;
    }

    id<MTLAccelerationStructure> structure_ = nil;
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
    return impl_ && impl_->structure_ != nil;
}

void* AccelerationStructure::rawHandle() const noexcept {
    return impl_ && impl_->structure_ ? (__bridge void*)impl_->structure_ : nullptr;
}

AccelerationStructure makeAccelerationStructure(std::string label, std::size_t size, void* handle) {
    id<MTLAccelerationStructure> structure = (__bridge id<MTLAccelerationStructure>)handle;
    auto impl = std::make_unique<AccelerationStructureImpl>(structure);
    return AccelerationStructure(std::move(label), size, std::move(impl));
}

}  // namespace rtr::rendering
