#import <Foundation/Foundation.h>
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif
#import <Metal/Metal.h>

#include "RTRMetalEngine/Rendering/RayTracingPipeline.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

#include <filesystem>

namespace rtr::rendering {

class RayTracingPipeline::Impl {
public:
    Impl(id<MTLComputePipelineState> state, bool requiresAS)
        : state_(state), requiresAccelerationStructure_(requiresAS) {}

    ~Impl() { state_ = nil; }

    bool isValid() const noexcept { return state_ != nil; }
    id<MTLComputePipelineState> pipeline() const noexcept { return state_; }
    bool requiresAccelerationStructure() const noexcept { return requiresAccelerationStructure_; }

private:
    id<MTLComputePipelineState> state_ = nil;
    bool requiresAccelerationStructure_ = false;
};

RayTracingPipeline::RayTracingPipeline() = default;
RayTracingPipeline::~RayTracingPipeline() = default;
RayTracingPipeline::RayTracingPipeline(RayTracingPipeline&&) noexcept = default;
RayTracingPipeline& RayTracingPipeline::operator=(RayTracingPipeline&&) noexcept = default;

bool RayTracingPipeline::initialize(MetalContext& context, const std::string& shaderLibraryPath) {
    if (!context.isValid()) {
        core::Logger::warn("RTPipeline", "Metal context invalid; compute pipeline initialization skipped");
        return false;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
    if (!device) {
        core::Logger::error("RTPipeline", "Failed to acquire Metal device");
        return false;
    }

    namespace fs = std::filesystem;

    // Search common build output locations if the provided path is missing.
    std::vector<std::string> candidatePaths = {
        shaderLibraryPath,
        "cmake-build-debug/shaders/RTRShaders.metallib",
        "cmake-build-release/shaders/RTRShaders.metallib",
        "build/shaders/RTRShaders.metallib",
    };

    std::string resolvedPath = shaderLibraryPath;
    for (const auto& candidate : candidatePaths) {
        std::error_code ec;
        if (fs::exists(candidate, ec)) {
            resolvedPath = candidate;
            break;
        }
    }

    NSURL* libraryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:resolvedPath.c_str()]];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
    if (!library || error) {
        core::Logger::error("RTPipeline", "Failed to load shader library %s (%s)", resolvedPath.c_str(),
                            error.localizedDescription.UTF8String);
        return false;
    }

    bool usesHardwareKernel = true;
    id<MTLFunction> kernel = [library newFunctionWithName:@"rtHardwareKernel"];
    if (!kernel) {
        usesHardwareKernel = false;
        core::Logger::warn("RTPipeline", "rtHardwareKernel unavailable; falling back to rtGradientKernel");
        kernel = [library newFunctionWithName:@"rtGradientKernel"];
    }

    if (!kernel) {
        core::Logger::error("RTPipeline", "No suitable ray tracing kernel found in %s", shaderLibraryPath.c_str());
        return false;
    }

    NSError* pipelineError = nil;
    id<MTLComputePipelineState> pipelineState = nil;
    if (@available(macOS 13.0, *)) {
        MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
        descriptor.computeFunction = kernel;
        descriptor.label = @"RTRHardwareRayKernel";
        pipelineState = [device newComputePipelineStateWithDescriptor:descriptor
                                                              options:0
                                                           reflection:nil
                                                                error:&pipelineError];
    } else {
        pipelineState = [device newComputePipelineStateWithFunction:kernel error:&pipelineError];
    }

    if (!pipelineState || pipelineError) {
        core::Logger::error("RTPipeline", "Failed to create compute pipeline: %s",
                            pipelineError.localizedDescription.UTF8String ? pipelineError.localizedDescription.UTF8String : "unknown error");
        return false;
    }

    impl_ = std::make_unique<Impl>(pipelineState, usesHardwareKernel);
    return true;
}

bool RayTracingPipeline::isValid() const noexcept { return impl_ && impl_->isValid(); }

void* RayTracingPipeline::rawPipelineState() const noexcept {
    return impl_ && impl_->isValid() ? (__bridge void*)impl_->pipeline() : nullptr;
}

bool RayTracingPipeline::requiresAccelerationStructure() const noexcept {
    return impl_ && impl_->requiresAccelerationStructure();
}

}  // namespace rtr::rendering
