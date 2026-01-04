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
    explicit Impl(id<MTLComputePipelineState> rayState) : rayState_(rayState) {}

    ~Impl() { rayState_ = nil; }

    [[nodiscard]] bool isValid() const noexcept { return rayState_ != nil; }
    [[nodiscard]] id<MTLComputePipelineState> rayPipeline() const noexcept { return rayState_; }

private:
    id<MTLComputePipelineState> rayState_ = nil;
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

    id<MTLComputePipelineState> rayState = makePipeline(@"rayKernel", @"RTRHardwareRayKernel");

    if (rayState == nil) {
        core::Logger::error("RTPipeline", "Failed to build hardware ray tracing kernel");
        impl_.reset();
        return false;
    }

    impl_ = std::make_unique<Impl>(rayState);
    core::Logger::info("RTPipeline", "Hardware ray tracing kernels initialized from %s", resolvedPath.c_str());
    return true;
}

bool RayTracingPipeline::isValid() const noexcept { return impl_ && impl_->isValid(); }

void* RayTracingPipeline::rayPipelineState() const noexcept {
    return impl_ ? (__bridge void*)impl_->rayPipeline() : nullptr;
}

}  // namespace rtr::rendering
