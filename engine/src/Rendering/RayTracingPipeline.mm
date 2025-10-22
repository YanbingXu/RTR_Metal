#import <Foundation/Foundation.h>
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif
#import <Metal/Metal.h>
#if __has_include(<Metal/MetalRayTracing.h>)
#import <Metal/MetalRayTracing.h>
#define RTR_HAS_RAYTRACING_HEADERS 1
#else
#define RTR_HAS_RAYTRACING_HEADERS 0
#endif

#include "RTRMetalEngine/Rendering/RayTracingPipeline.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

namespace rtr::rendering {

#if RTR_HAS_RAYTRACING_HEADERS

class RayTracingPipeline::Impl {
public:
    explicit Impl(id<MTLRayTracingPipelineState> state)
        : state_(state) {}

    ~Impl() { state_ = nil; }

    bool isValid() const noexcept { return state_ != nil; }

private:
    id<MTLRayTracingPipelineState> state_ = nil;
};

RayTracingPipeline::RayTracingPipeline() = default;
RayTracingPipeline::~RayTracingPipeline() = default;
RayTracingPipeline::RayTracingPipeline(RayTracingPipeline&&) noexcept = default;
RayTracingPipeline& RayTracingPipeline::operator=(RayTracingPipeline&&) noexcept = default;

bool RayTracingPipeline::initialize(MetalContext& context, const std::string& shaderLibraryPath) {
    if (!context.isValid() || !context.supportsRayTracing()) {
        core::Logger::warn("RTPipeline", "Ray tracing not supported; pipeline initialization skipped");
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

    id<MTLFunction> rayGen = [library newFunctionWithName:@"rayGenMain"];
    id<MTLFunction> miss = [library newFunctionWithName:@"missMain"];
    id<MTLFunction> closestHit = [library newFunctionWithName:@"closestHitMain"];

    if (!rayGen || !miss || !closestHit) {
        core::Logger::error("RTPipeline", "Required ray tracing functions missing in %s", shaderLibraryPath.c_str());
        return false;
    }

    MTLRayTracingPipelineDescriptor* descriptor = [MTLRayTracingPipelineDescriptor new];
    descriptor.rayGenerationFunction = rayGen;
    descriptor.missFunctions = @[ miss ];

    MTLHitGroupDescriptor* hitGroup = [MTLHitGroupDescriptor new];
    hitGroup.closestHitFunction = closestHit;
    descriptor.hitGroupDescriptors = @[ hitGroup ];

    NSError* pipelineError = nil;
    id<MTLRayTracingPipelineState> pipelineState =
        [device newRayTracingPipelineStateWithDescriptor:descriptor
                                                 options:MTLPipelineOptionNone
                                              reflection:nil
                                                   error:&pipelineError];

    if (!pipelineState || pipelineError) {
        core::Logger::error("RTPipeline", "Failed to create ray tracing pipeline: %s",
                            pipelineError.localizedDescription.UTF8String);
        return false;
    }

    impl_ = std::make_unique<Impl>(pipelineState);
    return true;
}

bool RayTracingPipeline::isValid() const noexcept { return impl_ && impl_->isValid(); }

#else  // RTR_HAS_RAYTRACING_HEADERS

class RayTracingPipeline::Impl {
public:
    bool isValid() const noexcept { return false; }
};

RayTracingPipeline::RayTracingPipeline() = default;
RayTracingPipeline::~RayTracingPipeline() = default;
RayTracingPipeline::RayTracingPipeline(RayTracingPipeline&&) noexcept = default;
RayTracingPipeline& RayTracingPipeline::operator=(RayTracingPipeline&&) noexcept = default;

bool RayTracingPipeline::initialize(MetalContext&, const std::string&) {
    core::Logger::warn("RTPipeline", "Ray tracing headers unavailable; pipeline disabled");
    return false;
}

bool RayTracingPipeline::isValid() const noexcept { return false; }

#endif  // RTR_HAS_RAYTRACING_HEADERS

}  // namespace rtr::rendering
