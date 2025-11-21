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
    struct PipelineStates {
        id<MTLComputePipelineState> ray = nil;
        id<MTLComputePipelineState> shade = nil;
        id<MTLComputePipelineState> shadow = nil;
        id<MTLComputePipelineState> accumulate = nil;
    };

    explicit Impl(PipelineStates states)
        : states_(states) {}

    ~Impl() {
        states_.ray = nil;
        states_.shade = nil;
        states_.shadow = nil;
        states_.accumulate = nil;
    }

    bool isValid() const noexcept {
        return states_.ray != nil && states_.shade != nil && states_.shadow != nil && states_.accumulate != nil;
    }

    id<MTLComputePipelineState> pipeline(RayKernelStage stage) const noexcept {
        switch (stage) {
        case RayKernelStage::RayGeneration:
            return states_.ray;
        case RayKernelStage::Shade:
            return states_.shade;
        case RayKernelStage::Shadow:
            return states_.shadow;
        case RayKernelStage::Accumulate:
            return states_.accumulate;
        }
        return nil;
    }

private:
    PipelineStates states_;
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

    auto makePipeline = [&](NSString* functionName, NSString* label) -> id<MTLComputePipelineState> {
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:functionName];
        if (!function) {
            core::Logger::error("RTPipeline", "Missing shader function %s", functionName.UTF8String);
            return nil;
        }

        id<MTLComputePipelineState> state = nil;
        if (@available(macOS 13.0, *)) {
            MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function;
            descriptor.label = label;
            state = [device newComputePipelineStateWithDescriptor:descriptor
                                                          options:0
                                                       reflection:nil
                                                            error:&error];
        } else {
            state = [device newComputePipelineStateWithFunction:function error:&error];
        }

        if (!state || error) {
            const char* message = error ? error.localizedDescription.UTF8String : "unknown error";
            core::Logger::error("RTPipeline", "Failed to create pipeline %s (%s)", label.UTF8String, message);
            return nil;
        }
        return state;
    };

    Impl::PipelineStates pipelines;
    pipelines.ray = makePipeline(@"rayKernel", @"RTRHardwareRayKernel");
    pipelines.shade = makePipeline(@"shadeKernel", @"RTRHardwareShadeKernel");
    pipelines.shadow = makePipeline(@"shadowKernel", @"RTRHardwareShadowKernel");
    pipelines.accumulate = makePipeline(@"accumulateKernel", @"RTRHardwareAccumulateKernel");

    if (pipelines.ray == nil || pipelines.shade == nil || pipelines.shadow == nil || pipelines.accumulate == nil) {
        core::Logger::error("RTPipeline", "Failed to build hardware ray tracing compute pipeline set");
        impl_.reset();
        return false;
    }

    impl_ = std::make_unique<Impl>(pipelines);
    core::Logger::info("RTPipeline", "Hardware ray tracing kernels initialized from %s", resolvedPath.c_str());
    return true;
}

bool RayTracingPipeline::isValid() const noexcept { return impl_ && impl_->isValid(); }

void* RayTracingPipeline::rawPipelineState(RayKernelStage stage) const noexcept {
    return impl_ ? (__bridge void*)impl_->pipeline(stage) : nullptr;
}

bool RayTracingPipeline::hasHardwareKernels() const noexcept { return impl_ && impl_->isValid(); }

bool RayTracingPipeline::requiresAccelerationStructure() const noexcept {
    return hasHardwareKernels();
}

}  // namespace rtr::rendering
