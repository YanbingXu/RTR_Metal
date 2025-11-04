#import <Foundation/Foundation.h>
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif
#import <Metal/Metal.h>

#include "RTRMetalEngine/Rendering/RayTracingPipeline.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

namespace rtr::rendering {

class RayTracingPipeline::Impl {
public:
    explicit Impl(id<MTLComputePipelineState> state)
        : state_(state) {}

    ~Impl() { state_ = nil; }

    bool isValid() const noexcept { return state_ != nil; }
    id<MTLComputePipelineState> pipeline() const noexcept { return state_; }

private:
    id<MTLComputePipelineState> state_ = nil;
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

    NSURL* libraryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:shaderLibraryPath.c_str()]];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
    if (!library || error) {
        core::Logger::error("RTPipeline", "Failed to load shader library %s (%s)", shaderLibraryPath.c_str(),
                            error.localizedDescription.UTF8String);
        return false;
    }

    id<MTLFunction> kernel = [library newFunctionWithName:@"rtGradientKernel"];

    if (!kernel) {
        core::Logger::error("RTPipeline", "Function rtGradientKernel missing in %s", shaderLibraryPath.c_str());
        return false;
    }

    NSError* pipelineError = nil;
    id<MTLComputePipelineState> pipelineState =
        [device newComputePipelineStateWithFunction:kernel error:&pipelineError];

    if (!pipelineState || pipelineError) {
        core::Logger::error("RTPipeline", "Failed to create compute pipeline: %s",
                            pipelineError.localizedDescription.UTF8String ? pipelineError.localizedDescription.UTF8String : "unknown error");
        return false;
    }

    impl_ = std::make_unique<Impl>(pipelineState);
    return true;
}

bool RayTracingPipeline::isValid() const noexcept { return impl_ && impl_->isValid(); }

void* RayTracingPipeline::rawPipelineState() const noexcept {
    return impl_ && impl_->isValid() ? (__bridge void*)impl_->pipeline() : nullptr;
}

}  // namespace rtr::rendering
