#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"

#include <simd/simd.h>

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

namespace rtr::rendering {

using namespace simd;

class MPSPathTracer::Impl {
public:
    Impl(id<MTLDevice> device, id<MTLCommandQueue> queue)
        : device_(device), queue_(queue) {}

    ~Impl() {
        resetSceneResources();
        intersector_ = nil;
        queue_ = nil;
        device_ = nil;
    }

    bool hasDevice() const noexcept { return device_ != nil && queue_ != nil; }

    bool configureIntersector() {
        if (!hasDevice()) {
            return false;
        }

        intersector_ = nil;
        intersector_ = [[MPSRayIntersector alloc] initWithDevice:device_];
        if (!intersector_) {
            return false;
        }

        intersector_.rayDataType = MPSRayDataTypeOriginMaskDirectionMaxDistance;
        intersector_.rayStride = sizeof(MPSRayOriginMaskDirectionMaxDistance);
        intersector_.intersectionDataType = MPSIntersectionDataTypeDistancePrimitiveIndexCoordinates;
        intersector_.intersectionStride = sizeof(MPSIntersectionDistancePrimitiveIndexCoordinates);
        intersector_.rayMaskOptions = MPSRayMaskOptionNone;
        return true;
    }

    bool isReady() const noexcept {
        return hasDevice() && intersector_ != nil && accelerationStructure_ != nil && vertexBuffer_ != nil &&
               indexBuffer_ != nil;
    }

    bool uploadScene(std::span<const vector_float3> positions,
                     std::span<const uint32_t> indices) {
        if (!hasDevice() || intersector_ == nil) {
            return false;
        }
        if (positions.empty() || indices.empty() || indices.size() % 3 != 0) {
            return false;
        }

        resetSceneResources();

        const NSUInteger positionDataLength = static_cast<NSUInteger>(positions.size() * sizeof(vector_float3));
        const NSUInteger indexDataLength = static_cast<NSUInteger>(indices.size() * sizeof(uint32_t));

        vertexBuffer_ = [device_ newBufferWithBytes:positions.data()
                                             length:positionDataLength
                                            options:MTLResourceStorageModeShared];
        indexBuffer_ = [device_ newBufferWithBytes:indices.data()
                                            length:indexDataLength
                                           options:MTLResourceStorageModeShared];
        if (vertexBuffer_ == nil || indexBuffer_ == nil) {
            resetSceneResources();
            return false;
        }

        accelerationStructure_ = [[MPSTriangleAccelerationStructure alloc] initWithDevice:device_];
        if (accelerationStructure_ == nil) {
            resetSceneResources();
            return false;
        }

        accelerationStructure_.vertexBuffer = vertexBuffer_;
        accelerationStructure_.vertexStride = sizeof(vector_float3);
        accelerationStructure_.triangleCount = indices.size() / 3;
        accelerationStructure_.indexBuffer = indexBuffer_;
        accelerationStructure_.indexType = MPSDataTypeUInt32;
        [accelerationStructure_ rebuild];

        return true;
    }

    void resetSceneResources() {
        @autoreleasepool {
            accelerationStructure_ = nil;
            vertexBuffer_ = nil;
            indexBuffer_ = nil;
        }
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSTriangleAccelerationStructure* accelerationStructure_ = nil;
    MPSRayIntersector* intersector_ = nil;
    id<MTLBuffer> vertexBuffer_ = nil;
    id<MTLBuffer> indexBuffer_ = nil;
};

MPSPathTracer::MPSPathTracer() = default;
MPSPathTracer::~MPSPathTracer() = default;
MPSPathTracer::MPSPathTracer(MPSPathTracer&&) noexcept = default;
MPSPathTracer& MPSPathTracer::operator=(MPSPathTracer&&) noexcept = default;

bool MPSPathTracer::initialize(MetalContext& context) {
#if TARGET_OS_OSX
    if (!context.isValid()) {
        core::Logger::warn("MPSPathTracer", "Metal context invalid; skipping MPS initialization");
        return false;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
    if (!device) {
        core::Logger::warn("MPSPathTracer", "Failed to acquire device for MPS path tracer");
        return false;
    }

    if (!MPSSupportsMTLDevice(device)) {
        core::Logger::warn("MPSPathTracer", "Device does not support MPS ray tracing features");
        return false;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context.rawCommandQueue();
    if (!queue) {
        queue = [device newCommandQueue];
        if (!queue) {
            core::Logger::error("MPSPathTracer", "Failed to create command queue for MPS path tracer");
            return false;
        }
    }

    impl_ = std::make_unique<Impl>(device, queue);
    if (!impl_ || !impl_->hasDevice()) {
        core::Logger::error("MPSPathTracer", "Failed to prepare MPS path tracer device state");
        impl_.reset();
        return false;
    }

    if (!impl_->configureIntersector()) {
        core::Logger::error("MPSPathTracer", "Failed to configure MPS ray intersector");
        impl_.reset();
        return false;
    }

    core::Logger::info("MPSPathTracer", "MPS device ready; awaiting scene upload");
    return true;
#else
    (void)context;
    core::Logger::warn("MPSPathTracer", "MPS path tracer not supported on this platform");
    return false;
#endif
}

bool MPSPathTracer::uploadScene(std::span<const vector_float3> positions,
                                std::span<const uint32_t> indices) {
#if TARGET_OS_OSX
    if (!impl_ || !impl_->hasDevice()) {
        core::Logger::warn("MPSPathTracer", "Cannot upload scene before initialization");
        return false;
    }

    if (!impl_->uploadScene(positions, indices)) {
        core::Logger::error("MPSPathTracer", "Failed to upload scene data to MPS path tracer");
        return false;
    }

    const std::size_t triangleCount = indices.size() / 3;
    core::Logger::info("MPSPathTracer", "Uploaded %zu vertices and %zu triangles", positions.size(), triangleCount);
    return true;
#else
    (void)positions;
    (void)indices;
    core::Logger::warn("MPSPathTracer", "MPS path tracer not supported on this platform");
    return false;
#endif
}

bool MPSPathTracer::isValid() const noexcept { return impl_ && impl_->isReady(); }

void* MPSPathTracer::deviceHandle() const noexcept { return impl_ ? (__bridge void*)impl_->device_ : nullptr; }

void* MPSPathTracer::commandQueueHandle() const noexcept { return impl_ ? (__bridge void*)impl_->queue_ : nullptr; }

void* MPSPathTracer::accelerationStructureHandle() const noexcept { return impl_ ? (__bridge void*)impl_->accelerationStructure_ : nullptr; }

void* MPSPathTracer::intersectorHandle() const noexcept { return impl_ ? (__bridge void*)impl_->intersector_ : nullptr; }

}  // namespace rtr::rendering

#pragma clang diagnostic pop
