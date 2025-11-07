#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif

#include "RTRMetalEngine/Rendering/Renderer.hpp"

#include "RTRMetalEngine/Core/ImageLoader.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/AccelerationStructure.hpp"
#include "RTRMetalEngine/Rendering/AccelerationStructureBuilder.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"
#include "RTRMetalEngine/Rendering/RayTracingPipeline.hpp"
#include "RTRMetalEngine/Rendering/RayTracingShaderTypes.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "RTRMetalEngine/Scene/CornellBox.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

namespace rtr::rendering {

namespace {

constexpr std::uint32_t kDiagnosticWidth = 512;
constexpr std::uint32_t kDiagnosticHeight = 512;
constexpr std::size_t kRayTracingPixelStride = sizeof(float) * 4;  // RGBA32F

uint8_t floatToSRGBByte(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    const long rounded = std::lroundf(clamped * 255.0f);
    return static_cast<uint8_t>(std::clamp<long>(rounded, 0, 255));
}

enum class RayTracingShadingMode {
    Auto,
    HardwareOnly,
    GradientOnly,
};

std::string_view shadingModeLabel(RayTracingShadingMode mode) {
    switch (mode) {
    case RayTracingShadingMode::Auto:
        return "auto";
    case RayTracingShadingMode::HardwareOnly:
        return "hardware";
    case RayTracingShadingMode::GradientOnly:
        return "gradient";
    }
    return "auto";
}

RayTracingShadingMode parseShadingMode(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (value == "gpu" || value == "hardware" || value == "rt") {
        return RayTracingShadingMode::HardwareOnly;
    }
    if (value == "cpu" || value == "fallback" || value == "gradient") {
        return RayTracingShadingMode::GradientOnly;
    }
    return RayTracingShadingMode::Auto;
}

}  // namespace

struct RayTracingTarget {
    id<MTLTexture> colorTexture = nil;
    id<MTLBuffer> readbackBuffer = nil;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::size_t bytesPerPixel = 0;

    void reset() {
        colorTexture = nil;
        readbackBuffer = nil;
        width = 0;
        height = 0;
        bytesPerPixel = 0;
    }

    [[nodiscard]] bool matches(std::uint32_t w, std::uint32_t h) const noexcept {
        return colorTexture != nil && readbackBuffer != nil && width == w && height == h;
    }

    [[nodiscard]] bool isValid() const noexcept {
        return colorTexture != nil && readbackBuffer != nil;
    }
};

struct RayTracingResources {
    id<MTLTexture> accumulationTexture = nil;
    id<MTLTexture> randomTexture = nil;
    id<MTLBuffer> resourceBuffer = nil;
    BufferHandle instanceBuffer;
    BufferHandle materialBuffer;
    BufferHandle textureInfoBuffer;
    BufferHandle textureDataBuffer;
    BufferHandle fallbackVertexBuffer;
    BufferHandle fallbackIndexBuffer;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    NSUInteger resourceHeaderSize = 0;
    std::size_t meshCount = 0;
    std::size_t textureCount = 0;

    void reset() {
        accumulationTexture = nil;
        randomTexture = nil;
        resourceBuffer = nil;
        instanceBuffer = {};
        materialBuffer = {};
        textureInfoBuffer = {};
        textureDataBuffer = {};
        fallbackVertexBuffer = {};
        fallbackIndexBuffer = {};
        width = 0;
        height = 0;
        resourceHeaderSize = 0;
        meshCount = 0;
        textureCount = 0;
    }
};

constexpr std::size_t kUniformRingSize = 3;

struct Renderer::Impl {
    explicit Impl(core::EngineConfig cfg)
        : config(std::move(cfg)), context(), bufferAllocator(context), geometryStore(bufferAllocator),
          asBuilder(context), shadingMode(parseShadingMode(config.shadingMode)) {
        if (!context.isValid()) {
            core::Logger::error("Renderer", "Metal context initialization failed");
        } else {
            core::Logger::info("Renderer", "Renderer configured for %s", config.applicationName.c_str());
            context.logDeviceInfo();
            if (shadingMode == RayTracingShadingMode::GradientOnly) {
                core::Logger::info("Renderer",
                                   "Gradient-only shading mode selected; skipping hardware ray tracing setup");
            } else {
                if (!asBuilder.isRayTracingSupported()) {
                    core::Logger::warn("Renderer", "Metal device does not report ray tracing support");
                } else if (!rayTracingPipeline.initialize(context, config.shaderLibraryPath)) {
                    core::Logger::warn("Renderer", "Ray tracing pipeline initialization failed");
                }
            }
            if (!initializeScene()) {
                core::Logger::warn("Renderer", "Scene initialization failed; falling back to gradient pipeline");
            }
        }
    }

    ~Impl() {
        target.reset();
        fallbackRayGenState = nil;
        resources.reset();
        for (std::size_t i = 0; i < uniformBuffers.size(); ++i) {
            uniformBuffers[i] = nil;
        }
    }

    bool renderFrameInternal(bool writeOutput) {
        if (!context.isValid()) {
            core::Logger::warn("Renderer", "Skipping frame: Metal context invalid");
            return false;
        }

        const std::uint32_t requestedWidth = std::max<std::uint32_t>(1u, targetWidth);
        const std::uint32_t requestedHeight = std::max<std::uint32_t>(1u, targetHeight);
        if (!target.matches(requestedWidth, requestedHeight)) {
            if (!ensureOutputTarget(requestedWidth, requestedHeight)) {
                return false;
            }
        }

        bool wroteImage = false;
        const bool hardwareDesired = wantsHardwareRayTracing() && rayTracingPipeline.isValid();
        const bool logFrame = writeOutput || frameCounter == 0;

        if (hardwareDesired && isRayTracingReady()) {
            if (logFrame) {
                core::Logger::info("Renderer",
                                   "Dispatching hardware ray tracing (%ux%u, mode=%s)",
                                   target.width,
                                   target.height,
                                   std::string(shadingModeLabel(shadingMode)).c_str());
            }
            if (dispatchRayTracingPass()) {
                wroteImage = true;
            } else {
                core::Logger::warn("Renderer",
                                   "Hardware ray tracing dispatch failed; %s fallback %s",
                                   allowsGradientFallback() ? "attempting" : "skipping",
                                   allowsGradientFallback() ? "gradient" : "(disabled)");
            }
        } else if (hardwareDesired && shadingMode == RayTracingShadingMode::HardwareOnly) {
            core::Logger::error("Renderer",
                                "Hardware-only mode requested but ray tracing resources are unavailable");
        }

        if (!wroteImage && allowsGradientFallback()) {
            if (!dispatchFallbackGradient()) {
                core::Logger::warn("Renderer", "Fallback gradient dispatch failed");
            } else {
                wroteImage = true;
            }
        }

        if (wroteImage) {
            if (writeOutput) {
                writeRayTracingOutput();
            }
        } else if (logFrame) {
            core::Logger::warn("Renderer", "No renderable output produced this frame");
        }

        frameCounter++;
        if (writeOutput) {
            std::cout << "Renderer frame stub executed using " << context.deviceName() << std::endl;
        }

        return wroteImage;
    }

    [[nodiscard]] void* currentColorTextureHandle() const noexcept {
        return (__bridge void*)target.colorTexture;
    }

    core::EngineConfig config;
    MetalContext context;
    BufferAllocator bufferAllocator;
    GeometryStore geometryStore;
    AccelerationStructureBuilder asBuilder;
    RayTracingPipeline rayTracingPipeline;
    std::vector<AccelerationStructure> bottomLevelStructures;
    AccelerationStructure topLevelStructure;
    RayTracingTarget target;
    RayTracingResources resources;
    id<MTLComputePipelineState> fallbackRayGenState = nil;
    std::array<id<MTLBuffer>, kUniformRingSize> uniformBuffers = {nil, nil, nil};
    std::size_t uniformBufferCursor = 0;
    uint32_t frameCounter = 0;
    std::vector<RayTracingInstanceResource> instanceResources;
    std::vector<RayTracingMaterialResource> materialResources;
    std::vector<RayTracingTextureResource> textureResources;
    std::vector<float> texturePixels;
    std::string outputPath = "renderer_output.ppm";
    std::uint32_t targetWidth = kDiagnosticWidth;
    std::uint32_t targetHeight = kDiagnosticHeight;
    bool debugAlbedo = false;
    RayTracingShadingMode shadingMode = RayTracingShadingMode::Auto;

    [[nodiscard]] bool isRayTracingReady() const noexcept {
        return context.isValid() && rayTracingPipeline.isValid() && topLevelStructure.isValid();
    }

    [[nodiscard]] bool wantsHardwareRayTracing() const noexcept {
        return shadingMode != RayTracingShadingMode::GradientOnly;
    }

    [[nodiscard]] bool allowsGradientFallback() const noexcept {
        return shadingMode != RayTracingShadingMode::HardwareOnly;
    }

    [[nodiscard]] bool ensureFallbackPipeline() {
        if (!context.isValid()) {
            return false;
        }

        if (fallbackRayGenState != nil) {
            return true;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
        if (!device) {
            core::Logger::error("Renderer", "Unable to acquire Metal device for fallback pipeline");
            return false;
        }

        NSURL* libraryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:config.shaderLibraryPath.c_str()]];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
        auto errorMessage = [](NSError* err) {
            return err ? err.localizedDescription.UTF8String : "<unknown error>";
        };
        if (!library || error) {
            core::Logger::error("Renderer", "Failed to load shader library for fallback pipeline (%s)",
                                errorMessage(error));
            return false;
        }

        id<MTLFunction> rayGenFunction = [library newFunctionWithName:@"rayGenMain"];
        if (!rayGenFunction) {
            core::Logger::error("Renderer", "Fallback shader missing function 'rayGenMain'");
            return false;
        }

        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:rayGenFunction
                                                                                           error:&error];
        if (!pipelineState || error) {
            core::Logger::error("Renderer", "Failed to create fallback compute pipeline (%s)",
                                errorMessage(error));
            return false;
        }

        fallbackRayGenState = pipelineState;
        core::Logger::info("Renderer", "Fallback compute pipeline initialized");
        return true;
    }

    [[nodiscard]] bool dispatchFallbackGradient() {
        if (!ensureOutputTarget(targetWidth, targetHeight)) {
            core::Logger::warn("Renderer", "Fallback dispatch skipped: no output target");
            return false;
        }

        if (!ensureFallbackPipeline()) {
            core::Logger::warn("Renderer", "Fallback dispatch skipped: no compute pipeline");
            return false;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context.rawCommandQueue();
        if (!queue) {
            core::Logger::error("Renderer", "Command queue unavailable for fallback dispatch");
            return false;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) {
            core::Logger::error("Renderer", "Failed to create command buffer for fallback dispatch");
            return false;
        }

        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        if (!computeEncoder) {
            core::Logger::error("Renderer", "Failed to create compute encoder for fallback dispatch");
            return false;
        }

        [computeEncoder setComputePipelineState:fallbackRayGenState];
        [computeEncoder setTexture:target.colorTexture atIndex:0];

        const NSUInteger threadWidth = 8;
        const NSUInteger threadHeight = 8;
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
        MTLSize threadgroups = MTLSizeMake((target.width + threadWidth - 1) / threadWidth,
                                           (target.height + threadHeight - 1) / threadHeight,
                                           1);
        [computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [computeEncoder endEncoding];

        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        if (blitEncoder) {
            MTLOrigin origin = MTLOriginMake(0, 0, 0);
            MTLSize size = MTLSizeMake(target.width, target.height, 1);
            const std::size_t bytesPerRow = static_cast<std::size_t>(target.width) * target.bytesPerPixel;
            const std::size_t bytesPerImage =
                static_cast<std::size_t>(target.width) * target.height * target.bytesPerPixel;
            [blitEncoder copyFromTexture:target.colorTexture
                              sourceSlice:0
                              sourceLevel:0
                             sourceOrigin:origin
                               sourceSize:size
                                 toBuffer:target.readbackBuffer
                        destinationOffset:0
                   destinationBytesPerRow:bytesPerRow
                 destinationBytesPerImage:bytesPerImage];
            [blitEncoder endEncoding];
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        core::Logger::info("Renderer", "Fallback gradient dispatch completed");
        return true;
    }

    [[nodiscard]] bool dispatchRayTracingPass() {
        if (!rayTracingPipeline.isValid()) {
            core::Logger::warn("Renderer", "Compute ray tracing pipeline unavailable; skipping dispatch");
            return false;
        }

        if (!ensureOutputTarget(targetWidth, targetHeight)) {
            core::Logger::warn("Renderer", "Unable to prepare output target for ray tracing dispatch");
            return false;
        }

        if (!ensureRayTracingResources(target.width, target.height)) {
            core::Logger::error("Renderer", "Failed to prepare ray tracing resources");
            return false;
        }

        if (!ensureRayTracingUniformBuffer(target.width, target.height)) {
            core::Logger::error("Renderer", "Failed to prepare ray tracing uniform buffer");
            return false;
        }

        id<MTLBuffer> uniformBuffer = acquireUniformBufferForFrame();
        if (!uniformBuffer) {
            core::Logger::error("Renderer", "Uniform buffer unavailable for ray tracing dispatch");
            return false;
        }

        updateRayTracingUniforms(uniformBuffer);

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context.rawCommandQueue();
        if (!queue) {
            core::Logger::error("Renderer", "Command queue unavailable for ray tracing dispatch");
            return false;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) {
            core::Logger::error("Renderer", "Failed to create command buffer for ray tracing dispatch");
            return false;
        }

        id<MTLComputePipelineState> pipelineState =
            (__bridge id<MTLComputePipelineState>)rayTracingPipeline.rawPipelineState();
        if (!pipelineState) {
            core::Logger::error("Renderer", "Ray tracing compute pipeline missing");
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            core::Logger::error("Renderer", "Failed to create compute encoder for ray tracing dispatch");
            return false;
        }

        [encoder setComputePipelineState:pipelineState];
        core::Logger::info("Renderer", "Using compute pipeline '%s' (requires AS: %s)",
                           [[pipelineState label] UTF8String],
                           rayTracingPipeline.requiresAccelerationStructure() ? "yes" : "no");
        if (rayTracingPipeline.requiresAccelerationStructure()) {
            id<MTLAccelerationStructure> accelerationStructure =
                topLevelStructure.isValid() ? (__bridge id<MTLAccelerationStructure>)topLevelStructure.rawHandle() : nil;
            if (accelerationStructure) {
                core::Logger::info("Renderer", "Binding TLAS %p", accelerationStructure);
                if ([encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
                    [encoder setAccelerationStructure:accelerationStructure atBufferIndex:0];
                } else {
                    core::Logger::warn("Renderer", "Compute encoder missing setAccelerationStructure API; TLAS not bound");
                }

                if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                    [encoder useResource:accelerationStructure usage:MTLResourceUsageRead];
                }
            } else {
                core::Logger::warn("Renderer", "TLAS unavailable; ray tracing kernel may miss scene data");
                core::Logger::warn("Renderer", "TLAS unavailable; ray tracing kernel may miss scene data");
            }
        }

        if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
            for (const auto& blas : bottomLevelStructures) {
                if (!blas.isValid()) {
                    continue;
                }
                id<MTLAccelerationStructure> blasHandle = (__bridge id<MTLAccelerationStructure>)blas.rawHandle();
                if (blasHandle) {
                    [encoder useResource:blasHandle usage:MTLResourceUsageRead];
                }
            }

            const auto& uploadedMeshes = geometryStore.uploadedMeshes();
            for (const auto& meshBuffers : uploadedMeshes) {
                id<MTLBuffer> gpuVertex = (__bridge id<MTLBuffer>)meshBuffers.gpuVertexBuffer.nativeHandle();
                if (gpuVertex) {
                    [encoder useResource:gpuVertex usage:MTLResourceUsageRead];
                }
                id<MTLBuffer> gpuIndex = (__bridge id<MTLBuffer>)meshBuffers.gpuIndexBuffer.nativeHandle();
                if (gpuIndex) {
                    [encoder useResource:gpuIndex usage:MTLResourceUsageRead];
                }
            }
        }

        [encoder setBuffer:uniformBuffer offset:0 atIndex:1];
        if (resources.resourceBuffer != nil) {
            [encoder setBuffer:resources.resourceBuffer offset:0 atIndex:2];
            const NSUInteger meshOffset = resources.resourceHeaderSize;
            const NSUInteger bufferLength = [resources.resourceBuffer length];
            const NSUInteger offset = meshOffset < bufferLength ? meshOffset : 0;
            [encoder setBuffer:resources.resourceBuffer offset:offset atIndex:3];
        }
        if (resources.fallbackVertexBuffer.isValid()) {
            id<MTLBuffer> fallbackVertices = (__bridge id<MTLBuffer>)resources.fallbackVertexBuffer.nativeHandle();
            [encoder setBuffer:fallbackVertices offset:0 atIndex:4];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:4];
        }
        if (resources.fallbackIndexBuffer.isValid()) {
            id<MTLBuffer> fallbackIndices = (__bridge id<MTLBuffer>)resources.fallbackIndexBuffer.nativeHandle();
            [encoder setBuffer:fallbackIndices offset:0 atIndex:5];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:5];
        }
        id<MTLBuffer> instanceBuffer = (__bridge id<MTLBuffer>)resources.instanceBuffer.nativeHandle();
        if (instanceBuffer) {
            [encoder setBuffer:instanceBuffer offset:0 atIndex:6];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:6];
        }

        id<MTLBuffer> materialBuffer = (__bridge id<MTLBuffer>)resources.materialBuffer.nativeHandle();
        if (materialBuffer) {
            [encoder setBuffer:materialBuffer offset:0 atIndex:7];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:7];
        }
        id<MTLBuffer> textureInfoBuffer = (__bridge id<MTLBuffer>)resources.textureInfoBuffer.nativeHandle();
        if (textureInfoBuffer) {
            [encoder setBuffer:textureInfoBuffer offset:0 atIndex:8];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:8];
        }
        id<MTLBuffer> textureDataBuffer = (__bridge id<MTLBuffer>)resources.textureDataBuffer.nativeHandle();
        if (textureDataBuffer) {
            [encoder setBuffer:textureDataBuffer offset:0 atIndex:9];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:9];
        }
        [encoder setTexture:target.colorTexture atIndex:0];
        if (resources.accumulationTexture != nil) {
            [encoder setTexture:resources.accumulationTexture atIndex:1];
        }
        if (resources.randomTexture != nil) {
            [encoder setTexture:resources.randomTexture atIndex:2];
        }

        const NSUInteger threadWidth = 8;
        const NSUInteger threadHeight = 8;
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
        MTLSize threadsPerGrid = MTLSizeMake(target.width, target.height, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        if (blitEncoder) {
            MTLOrigin origin = MTLOriginMake(0, 0, 0);
            MTLSize size = MTLSizeMake(target.width, target.height, 1);
            const std::size_t bytesPerRow = static_cast<std::size_t>(target.width) * target.bytesPerPixel;
            const std::size_t bytesPerImage = static_cast<std::size_t>(target.width) * target.height * target.bytesPerPixel;
            [blitEncoder copyFromTexture:target.colorTexture
                              sourceSlice:0
                              sourceLevel:0
                             sourceOrigin:origin
                               sourceSize:size
                                 toBuffer:target.readbackBuffer
                        destinationOffset:0
                   destinationBytesPerRow:bytesPerRow
                 destinationBytesPerImage:bytesPerImage];
            [blitEncoder endEncoding];
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        return true;
    }

    [[nodiscard]] bool ensureRayTracingUniformBuffer(std::uint32_t /*width*/, std::uint32_t /*height*/) {
        if (!context.isValid()) {
            return false;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
        if (!device) {
            core::Logger::error("Renderer", "Unable to acquire Metal device for uniform buffers");
            return false;
        }

        bool allValid = true;
        for (std::size_t i = 0; i < uniformBuffers.size(); ++i) {
            if (uniformBuffers[i] != nil) {
                continue;
            }

            id<MTLBuffer> buffer = [device newBufferWithLength:sizeof(RayTracingUniforms)
                                                        options:MTLResourceStorageModeShared];
            if (!buffer) {
                core::Logger::error("Renderer", "Failed to allocate uniform buffer %zu", i);
                allValid = false;
                break;
            }

            NSString* label = [NSString stringWithFormat:@"rtr.uniform[%zu]", i];
            [buffer setLabel:label];
            uniformBuffers[i] = buffer;
        }

        return allValid && uniformBuffers[0] != nil;
    }

    [[nodiscard]] id<MTLBuffer> acquireUniformBufferForFrame() {
        if (uniformBuffers.empty()) {
            return nil;
        }

        const std::size_t index = uniformBufferCursor;
        uniformBufferCursor = (uniformBufferCursor + 1) % uniformBuffers.size();
        return uniformBuffers[index];
    }

    void updateRayTracingUniforms(id<MTLBuffer> buffer) {
        if (!buffer) {
            return;
        }

        auto* uniforms = reinterpret_cast<RayTracingUniforms*>([buffer contents]);
        if (!uniforms) {
            return;
        }

        uniforms->eye = simd_make_float4(0.0F, 0.0F, 2.0F, 1.0F);
        uniforms->forward = simd_make_float4(0.0F, 0.0F, -1.0F, 0.0F);
        uniforms->right = simd_make_float4(1.0F, 0.0F, 0.0F, 0.0F);
        uniforms->up = simd_make_float4(0.0F, 1.0F, 0.0F, 0.0F);
        uniforms->imagePlaneHalfExtents = simd_make_float2(1.0F, 1.0F);
        uniforms->width = target.width;
        uniforms->height = target.height;
        uniforms->frameIndex = frameCounter;
        uniforms->flags = debugAlbedo ? 0x1u : 0u;

        if (debugAlbedo) {
            core::Logger::info("Renderer", "Debug uniforms: flags=0x%x", uniforms->flags);
        }

        [buffer didModifyRange:NSMakeRange(0, sizeof(RayTracingUniforms))];
    }

    [[nodiscard]] bool ensureOutputTarget(std::uint32_t width, std::uint32_t height) {
        if (!context.isValid()) {
            return false;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
        if (!device) {
            core::Logger::error("Renderer", "Unable to acquire Metal device for output target");
            target.reset();
            return false;
        }

        if (target.matches(width, height)) {
            return true;
        }

        target.reset();

        MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                                      width:width
                                                                                                     height:height
                                                                                                  mipmapped:NO];
        textureDescriptor.storageMode = MTLStorageModePrivate;
        textureDescriptor.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;

        id<MTLTexture> texture = [device newTextureWithDescriptor:textureDescriptor];
        if (!texture) {
            core::Logger::error("Renderer", "Failed to allocate ray tracing output texture (%ux%u)", width, height);
            return false;
        }

        const std::size_t bufferLength = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                                         kRayTracingPixelStride;
        id<MTLBuffer> readback = [device newBufferWithLength:bufferLength options:MTLResourceStorageModeShared];
        if (!readback) {
            core::Logger::error("Renderer", "Failed to allocate ray tracing readback buffer (%zu bytes)",
                                bufferLength);
            return false;
        }

        target.colorTexture = texture;
        target.readbackBuffer = readback;
        target.width = width;
        target.height = height;
        target.bytesPerPixel = kRayTracingPixelStride;
        core::Logger::info("Renderer", "Prepared ray tracing target (%ux%u, %zu bytes)", width, height, bufferLength);
        return true;
    }

    [[nodiscard]] bool ensureRayTracingResources(std::uint32_t width, std::uint32_t height) {
        if (!context.isValid()) {
            return false;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
        if (!device) {
            core::Logger::error("Renderer", "Unable to acquire Metal device for ray tracing resources");
            resources.reset();
            return false;
        }

        bool success = true;

        if (resources.accumulationTexture == nil || resources.width != width || resources.height != height) {
            MTLTextureDescriptor* accumulationDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                  width:width
                                                                 height:height
                                                              mipmapped:NO];
            accumulationDesc.storageMode = MTLStorageModePrivate;
            accumulationDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            id<MTLTexture> accumulation = [device newTextureWithDescriptor:accumulationDesc];
            if (!accumulation) {
                core::Logger::error("Renderer", "Failed to allocate accumulation texture (%ux%u)", width, height);
                success = false;
            } else {
                resources.accumulationTexture = accumulation;
                resources.width = width;
                resources.height = height;
            }
        }

        if (resources.randomTexture == nil) {
            const NSUInteger noiseSize = 128;
            MTLTextureDescriptor* randomDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                  width:noiseSize
                                                                 height:noiseSize
                                                              mipmapped:NO];
            randomDesc.storageMode = MTLStorageModeShared;
            randomDesc.usage = MTLTextureUsageShaderRead;
            id<MTLTexture> randomTexture = [device newTextureWithDescriptor:randomDesc];
            if (!randomTexture) {
                core::Logger::error("Renderer", "Failed to allocate random texture");
                success = false;
            } else {
                std::vector<float> noise(noiseSize * noiseSize * 4);
                std::mt19937 rng(1337u);
                std::uniform_real_distribution<float> dist(0.0F, 1.0F);
                for (float& value : noise) {
                    value = dist(rng);
                }
                MTLRegion region = MTLRegionMake2D(0, 0, noiseSize, noiseSize);
                [randomTexture replaceRegion:region
                                  mipmapLevel:0
                                    withBytes:noise.data()
                                  bytesPerRow:sizeof(float) * 4 * noiseSize];
                resources.randomTexture = randomTexture;
            }
        }

        const auto& uploadedMeshes = geometryStore.uploadedMeshes();
        const std::size_t meshCount = uploadedMeshes.size();
        const std::size_t headerSize = sizeof(RayTracingResourceHeader);
        const std::size_t meshEntrySize = sizeof(RayTracingMeshResource);

        std::vector<std::uint8_t> fallbackVertexData;
        std::vector<std::uint32_t> fallbackIndexData;
        const std::uint32_t kInvalidOffset = std::numeric_limits<std::uint32_t>::max();
        const bool supportsGPUAddress = context.supportsRayTracing();

        if (meshCount > 0) {
            std::size_t estimatedVertexBytes = 0;
            std::size_t estimatedIndexCount = 0;
            for (const auto& mesh : uploadedMeshes) {
                estimatedVertexBytes += mesh.vertexStride * mesh.vertexCount;
                estimatedIndexCount += mesh.indexCount;
            }
            fallbackVertexData.reserve(estimatedVertexBytes);
            fallbackIndexData.reserve(estimatedIndexCount);
        }

        std::size_t requiredLength = headerSize + meshCount * meshEntrySize;
        if (requiredLength == 0) {
            requiredLength = headerSize;
        }

        if (resources.resourceBuffer == nil || static_cast<std::size_t>([resources.resourceBuffer length]) < requiredLength) {
            id<MTLBuffer> buffer = [device newBufferWithLength:requiredLength options:MTLResourceStorageModeShared];
            if (!buffer) {
                core::Logger::error("Renderer", "Failed to allocate ray tracing resource buffer (%zu bytes)",
                                    requiredLength);
                success = false;
            } else {
                NSString* label = @"rtr.resourcePointers";
                [buffer setLabel:label];
                resources.resourceBuffer = buffer;
                resources.resourceHeaderSize = static_cast<NSUInteger>(headerSize);
            }
        }

        if (resources.resourceBuffer != nil) {
            resources.resourceHeaderSize = static_cast<NSUInteger>(headerSize);
            auto* header = reinterpret_cast<RayTracingResourceHeader*>([resources.resourceBuffer contents]);
            if (header) {
                *header = {};
                header->geometryCount = static_cast<std::uint32_t>(std::min<std::size_t>(
                    meshCount, static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
                header->instanceCount = static_cast<std::uint32_t>(std::min<std::size_t>(
                    instanceResources.size(), static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
                header->materialCount = static_cast<std::uint32_t>(std::min<std::size_t>(
                    materialResources.size(), static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
                header->textureCount = static_cast<std::uint32_t>(std::min<std::size_t>(
                    textureResources.size(), static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
                header->randomTextureWidth = resources.randomTexture != nil
                                               ? static_cast<std::uint32_t>([resources.randomTexture width])
                                               : 0U;
                header->randomTextureHeight = resources.randomTexture != nil
                                                ? static_cast<std::uint32_t>([resources.randomTexture height])
                                                : 0U;

                auto* meshEntries = reinterpret_cast<RayTracingMeshResource*>(
                    reinterpret_cast<std::uint8_t*>(header) + headerSize);

                if (meshCount == 0) {
                    core::Logger::warn("Renderer", "No uploaded meshes available for ray tracing scene");
                }

                for (std::size_t i = 0; i < meshCount; ++i) {
                    const auto& meshBuffers = uploadedMeshes[i];
                    RayTracingMeshResource entry{};
                    entry.vertexCount = static_cast<std::uint32_t>(std::min<std::size_t>(
                        meshBuffers.vertexCount, static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
                    entry.indexCount = static_cast<std::uint32_t>(std::min<std::size_t>(
                        meshBuffers.indexCount, static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
                    entry.vertexStride = static_cast<std::uint32_t>(
                        std::min<std::size_t>(meshBuffers.vertexStride,
                                              static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
                    entry.materialIndex = 0U;
                    entry.fallbackVertexOffset = kInvalidOffset;
                    entry.fallbackIndexOffset = kInvalidOffset;

                    id<MTLBuffer> gpuVertexBuffer = (__bridge id<MTLBuffer>)meshBuffers.gpuVertexBuffer.nativeHandle();
                    id<MTLBuffer> gpuIndexBuffer = (__bridge id<MTLBuffer>)meshBuffers.gpuIndexBuffer.nativeHandle();
                    id<MTLBuffer> cpuVertexBuffer = (__bridge id<MTLBuffer>)meshBuffers.cpuVertexBuffer.nativeHandle();
                    id<MTLBuffer> cpuIndexBuffer = (__bridge id<MTLBuffer>)meshBuffers.cpuIndexBuffer.nativeHandle();

                    if (supportsGPUAddress && gpuVertexBuffer) {
                        entry.vertexBufferAddress = gpuVertexBuffer.gpuAddress;
                    }
                    if (supportsGPUAddress && gpuIndexBuffer) {
                        entry.indexBufferAddress = gpuIndexBuffer.gpuAddress;
                    }

                    const bool needsFallback = entry.vertexBufferAddress == 0ULL || entry.indexBufferAddress == 0ULL;
                    if (needsFallback && cpuVertexBuffer && cpuIndexBuffer) {
                        const auto* vertexBytes = static_cast<const std::uint8_t*>([cpuVertexBuffer contents]);
                        const auto* indexBytes = static_cast<const std::uint32_t*>([cpuIndexBuffer contents]);
                        if (vertexBytes && indexBytes) {
                            const std::size_t vertexByteLength = meshBuffers.vertexStride * meshBuffers.vertexCount;
                            const std::size_t indexElementCount = meshBuffers.indexCount;
                            const std::size_t vertexAlignment = 16;

                            std::size_t vertexOffset = fallbackVertexData.size();
                            const std::size_t padding = (vertexAlignment - (vertexOffset % vertexAlignment)) % vertexAlignment;
                            fallbackVertexData.insert(fallbackVertexData.end(), padding, static_cast<std::uint8_t>(0));
                            vertexOffset = fallbackVertexData.size();
                            fallbackVertexData.insert(fallbackVertexData.end(),
                                                      vertexBytes,
                                                      vertexBytes + vertexByteLength);
                            if (vertexOffset <= kInvalidOffset && vertexByteLength <= kInvalidOffset) {
                                entry.fallbackVertexOffset = static_cast<std::uint32_t>(vertexOffset);
                            }

                            const std::size_t indexOffset = fallbackIndexData.size();
                            fallbackIndexData.insert(fallbackIndexData.end(), indexBytes, indexBytes + indexElementCount);
                            if (indexOffset <= kInvalidOffset) {
                                entry.fallbackIndexOffset = static_cast<std::uint32_t>(indexOffset);
                            }
                        }
                    }

                    meshEntries[i] = entry;

                    core::Logger::info("Renderer",
                                        "Mesh[%zu]: vtx=%u idx=%u stride=%u gpuAddr(v=%llx, i=%llx) fallback(v=%u, i=%u)",
                                        i,
                                        entry.vertexCount,
                                        entry.indexCount,
                                        entry.vertexStride,
                                        static_cast<unsigned long long>(entry.vertexBufferAddress),
                                        static_cast<unsigned long long>(entry.indexBufferAddress),
                                        entry.fallbackVertexOffset,
                                        entry.fallbackIndexOffset);
                }

                header->instanceCount = static_cast<std::uint32_t>(
                    std::min<std::size_t>(instanceResources.size(),
                                           static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
                header->materialCount = static_cast<std::uint32_t>(
                    std::min<std::size_t>(materialResources.size(),
                                           static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));

                const NSUInteger bytesToMark = static_cast<NSUInteger>(headerSize + meshCount * meshEntrySize);
                [resources.resourceBuffer didModifyRange:NSMakeRange(0, bytesToMark)];
                resources.meshCount = meshCount;
            }
        }

        auto uploadBuffer = [&](BufferHandle& handle, const void* data, std::size_t length, const char* label) {
            if (length == 0) {
                handle = {};
                return;
            }

            if (!handle.isValid() || handle.length() < length) {
                handle = bufferAllocator.createBuffer(length, data, label);
            } else {
                id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle.nativeHandle();
                if (buffer && data) {
                    std::memcpy([buffer contents], data, length);
                    [buffer didModifyRange:NSMakeRange(0, length)];
                }
            }
        };

        uploadBuffer(resources.fallbackVertexBuffer,
                     fallbackVertexData.empty() ? nullptr : fallbackVertexData.data(),
                     fallbackVertexData.size(),
                     "rtr.fallbackVertices");

        uploadBuffer(resources.fallbackIndexBuffer,
                     fallbackIndexData.empty() ? nullptr : fallbackIndexData.data(),
                     fallbackIndexData.size() * sizeof(std::uint32_t),
                     "rtr.fallbackIndices");

        core::Logger::info("Renderer",
                           "Ray tracing resources: fallback vertices=%zu bytes, fallback indices=%zu elements",
                           fallbackVertexData.size(),
                           fallbackIndexData.size());
        if (!fallbackVertexData.empty()) {
            const float* vertexFloats = reinterpret_cast<const float*>(fallbackVertexData.data());
            core::Logger::info("Renderer",
                               "First fallback vertex position = (%.3f, %.3f, %.3f)",
                               vertexFloats[0],
                               vertexFloats[1],
                               vertexFloats[2]);
        }
        if (!fallbackIndexData.empty()) {
            core::Logger::info("Renderer",
                               "First fallback triangle indices = (%u, %u, %u)",
                               fallbackIndexData[0],
                               fallbackIndexData[1],
                               fallbackIndexData[2]);
        }

        uploadBuffer(resources.instanceBuffer,
                     instanceResources.data(),
                     instanceResources.size() * sizeof(RayTracingInstanceResource),
                     "rtr.instances");

        uploadBuffer(resources.materialBuffer,
                     materialResources.data(),
                     materialResources.size() * sizeof(RayTracingMaterialResource),
                     "rtr.materials");

        uploadBuffer(resources.textureInfoBuffer,
                     textureResources.empty() ? nullptr : textureResources.data(),
                     textureResources.size() * sizeof(RayTracingTextureResource),
                     "rtr.textureInfo");

        uploadBuffer(resources.textureDataBuffer,
                     texturePixels.empty() ? nullptr : texturePixels.data(),
                     texturePixels.size() * sizeof(float),
                     "rtr.textureData");

        resources.textureCount = textureResources.size();

        return success;
    }

    bool initializeScene();
    bool loadSceneInternal(const scene::Scene& scene);
    void writeRayTracingOutput() const;
    void setOutputPathInternal(std::string path);
    void setRenderSizeInternal(std::uint32_t width, std::uint32_t height);
    void setDebugModeInternal(bool enabled) { debugAlbedo = enabled; }

    void setShadingModeInternal(const std::string& value) {
        const RayTracingShadingMode newMode = parseShadingMode(value);
        config.shadingMode = value;
        if (newMode == shadingMode) {
            return;
        }

        shadingMode = newMode;

        if (shadingMode != RayTracingShadingMode::GradientOnly && !rayTracingPipeline.isValid()) {
            if (!asBuilder.isRayTracingSupported()) {
                core::Logger::warn("Renderer", "Metal device does not report ray tracing support");
            } else if (!rayTracingPipeline.initialize(context, config.shaderLibraryPath)) {
                core::Logger::warn("Renderer", "Ray tracing pipeline initialization failed for shading mode switch");
            }
        }
    }
};

bool Renderer::Impl::initializeScene() {
    scene::Scene scene = scene::createCornellBoxScene();
    if (!loadSceneInternal(scene)) {
        core::Logger::warn("Renderer", "Falling back to gradient pipeline; scene load failed");
        return false;
    }
    return true;
}

bool Renderer::Impl::loadSceneInternal(const scene::Scene& scene) {
    bottomLevelStructures.clear();
    topLevelStructure = AccelerationStructure{};
    instanceResources.clear();
    materialResources.clear();
    target.reset();
    resources.reset();

    if (!wantsHardwareRayTracing()) {
        core::Logger::info("Renderer",
                            "Gradient-only mode active; skipping acceleration structure build for scene");
        return true;
    }

    core::Logger::info("Renderer", "Loading scene with %zu meshes, %zu materials, %zu instances",
                       scene.meshes().size(),
                       scene.materials().size(),
                       scene.instances().size());

    if (!context.isValid()) {
        core::Logger::warn("Renderer", "Metal context invalid; cannot load scene");
        return false;
    }

    void* queueHandle = context.rawCommandQueue();
    if (!queueHandle) {
        core::Logger::error("Renderer", "Command queue unavailable for scene load");
        return false;
    }

    const auto& meshes = scene.meshes();
    const auto& materials = scene.materials();
    const auto& instances = scene.instances();

    std::vector<std::size_t> meshUploadIndices(meshes.size(), static_cast<std::size_t>(-1));
    std::vector<std::size_t> meshBLASIndices(meshes.size(), static_cast<std::size_t>(-1));

    for (std::size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
        const std::string label = "scene_mesh_" + std::to_string(meshIndex);
        const auto uploadIndex = geometryStore.uploadMesh(meshes[meshIndex], label);
        if (!uploadIndex.has_value()) {
            core::Logger::warn("Renderer", "Failed to upload mesh %zu", meshIndex);
            continue;
        }

        meshUploadIndices[meshIndex] = *uploadIndex;
        const auto& meshBuffers = geometryStore.uploadedMeshes()[*uploadIndex];
        auto blas = asBuilder.buildBottomLevel(meshBuffers, label, queueHandle);
        if (!blas.has_value()) {
            core::Logger::warn("Renderer", "Failed to build BLAS for mesh %zu", meshIndex);
            continue;
        }

        meshBLASIndices[meshIndex] = bottomLevelStructures.size();
        bottomLevelStructures.push_back(std::move(*blas));
    }

    if (bottomLevelStructures.empty()) {
        core::Logger::warn("Renderer", "No BLAS structures available; scene load aborted");
        return false;
    }

    std::vector<InstanceBuildInput> instanceInputs;
    instanceInputs.reserve(instances.size());
    instanceResources.clear();
    instanceResources.reserve(instances.size());
    std::size_t emissiveInstanceCount = 0;

    for (std::size_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
        const auto& instance = instances[instanceIndex];
        if (!instance.mesh.isValid() || instance.mesh.index >= meshBLASIndices.size()) {
            continue;
        }

        bool isEmissive = false;
        if (instance.material.isValid() && instance.material.index < materials.size()) {
            const auto& material = materials[instance.material.index];
            const auto emission = material.emission;
            isEmissive = emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f;
        }
        if (isEmissive) {
            emissiveInstanceCount++;
            continue;
        }

        const std::size_t blasIndex = meshBLASIndices[instance.mesh.index];
        const std::size_t meshResourceIndex = meshUploadIndices[instance.mesh.index];
        if (blasIndex == static_cast<std::size_t>(-1) || meshResourceIndex == static_cast<std::size_t>(-1)) {
            continue;
        }

        InstanceBuildInput input{};
        input.structure = &bottomLevelStructures[blasIndex];
        input.transform = instance.transform;
        input.userID = static_cast<std::uint32_t>(instanceInputs.size());
        input.mask = 0xFF;
        instanceInputs.push_back(input);

        RayTracingInstanceResource resource{};
        resource.objectToWorld = instance.transform;
        resource.worldToObject = simd_inverse(instance.transform);
        resource.meshIndex = static_cast<std::uint32_t>(meshResourceIndex);
        resource.materialIndex = (instance.material.isValid() && instance.material.index < materials.size())
                                     ? static_cast<std::uint32_t>(instance.material.index)
                                     : 0U;
        instanceResources.push_back(resource);
    }

    if (emissiveInstanceCount > 0) {
        core::Logger::info("Renderer",
                           "Skipped %zu emissive instance(s) when building TLAS (handled analytically)",
                           emissiveInstanceCount);
    }

    if (instanceInputs.empty()) {
        core::Logger::warn("Renderer", "No valid instances for TLAS build");
        return false;
    }

    auto tlas = asBuilder.buildTopLevel(instanceInputs, "scene_tlas", queueHandle);
    if (!tlas.has_value()) {
        core::Logger::warn("Renderer", "Failed to build TLAS for scene");
        return false;
    }
    topLevelStructure = std::move(*tlas);
    core::Logger::info("Renderer", "TLAS built: size=%zu bytes", topLevelStructure.sizeInBytes());

    materialResources.clear();
    materialResources.reserve(materials.size());
    textureResources.clear();
    texturePixels.clear();

    std::unordered_map<std::string, std::uint32_t> textureLookup;
    textureLookup.reserve(materials.size());

    auto canonicalisePath = [](const std::string& path) -> std::string {
        if (path.empty()) {
            return path;
        }
        std::error_code ec;
        std::filesystem::path fsPath(path);
        std::filesystem::path resolved = std::filesystem::weakly_canonical(fsPath, ec);
        if (ec) {
            resolved = fsPath.lexically_normal();
        }
        return resolved.string();
    };

    auto registerTexture = [&](const std::string& texturePath) -> std::optional<std::uint32_t> {
        if (texturePath.empty()) {
            return std::nullopt;
        }
        const std::string key = canonicalisePath(texturePath);
        const auto found = textureLookup.find(key);
        if (found != textureLookup.end()) {
            return found->second;
        }

        core::ImageData imageData{};
        if (!core::ImageLoader::loadRGBA32F(std::filesystem::path(key), imageData)) {
            return std::nullopt;
        }

        RayTracingTextureResource texture{};
        texture.width = imageData.width;
        texture.height = imageData.height;
        texture.rowPitch = imageData.width;
        texture.dataOffset = static_cast<std::uint32_t>(texturePixels.size());
        textureResources.push_back(texture);
        texturePixels.insert(texturePixels.end(), imageData.pixels.begin(), imageData.pixels.end());

        const std::uint32_t textureIndex = static_cast<std::uint32_t>(textureResources.size() - 1);
        textureLookup.emplace(key, textureIndex);
        return textureIndex;
    };

    for (const auto& material : materials) {
        RayTracingMaterialResource resource{};
        resource.albedo = material.albedo;
        resource.roughness = material.roughness;
        resource.emission = material.emission;
        resource.metallic = material.metallic;
        resource.reflectivity = material.reflectivity;
        resource.indexOfRefraction = material.indexOfRefraction;
        resource.textureIndex = kInvalidTextureIndex;
        if (!material.albedoTexturePath.empty()) {
            const auto textureIndex = registerTexture(material.albedoTexturePath);
            if (textureIndex.has_value()) {
                resource.textureIndex = *textureIndex;
            } else {
                core::Logger::warn("Renderer",
                                   "Failed to load texture '%s' for material",
                                   material.albedoTexturePath.c_str());
            }
        }
        materialResources.push_back(resource);
    }

    resources.textureCount = textureResources.size();

    const bool pipelineReady = ensureRayTracingResources(targetWidth, targetHeight);
    if (!pipelineReady) {
        core::Logger::warn("Renderer", "Failed to allocate ray tracing resources for scene");
    }

    return topLevelStructure.isValid();
}

void Renderer::Impl::writeRayTracingOutput() const {
    if (outputPath.empty() || target.readbackBuffer == nil || target.width == 0 || target.height == 0) {
        return;
    }

    const std::size_t pixelCount = static_cast<std::size_t>(target.width) * target.height;
    const float* src = static_cast<const float*>([target.readbackBuffer contents]);
    if (!src) {
        return;
    }

    float minValue = std::numeric_limits<float>::max();
    float maxValue = std::numeric_limits<float>::lowest();
    const std::size_t sampleCount = std::min<std::size_t>(pixelCount * 4, 16);
    for (std::size_t i = 0; i < pixelCount * 4; ++i) {
        minValue = std::min(minValue, src[i]);
        maxValue = std::max(maxValue, src[i]);
    }
    core::Logger::info("Renderer",
                       "First pixel RGBA = (%.4f, %.4f, %.4f, %.4f); min=%.4f max=%.4f",
                       src[0], src[1], src[2], src[3], minValue, maxValue);

    std::vector<uint8_t> bytes(pixelCount * 3);
    for (std::size_t i = 0; i < pixelCount; ++i) {
        const std::size_t base = i * 4;
        bytes[i * 3 + 0] = floatToSRGBByte(src[base + 0]);
        bytes[i * 3 + 1] = floatToSRGBByte(src[base + 1]);
        bytes[i * 3 + 2] = floatToSRGBByte(src[base + 2]);
    }

    std::ofstream file(outputPath, std::ios::binary);
    if (!file) {
        core::Logger::warn("Renderer", "Failed to write %s", outputPath.c_str());
        return;
    }

    file << "P6\n" << target.width << " " << target.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        core::Logger::warn("Renderer", "Failed to flush %s", outputPath.c_str());
    } else {
        core::Logger::info("Renderer", "Wrote ray tracing output to %s", outputPath.c_str());
    }
}

void Renderer::Impl::setOutputPathInternal(std::string path) {
    if (path.empty()) {
        outputPath = "renderer_output.ppm";
    } else {
        outputPath = std::move(path);
    }
}

void Renderer::Impl::setRenderSizeInternal(std::uint32_t width, std::uint32_t height) {
    const std::uint32_t sanitizedWidth = std::max<std::uint32_t>(1u, width);
    const std::uint32_t sanitizedHeight = std::max<std::uint32_t>(1u, height);
    if (sanitizedWidth == targetWidth && sanitizedHeight == targetHeight) {
        return;
    }

    targetWidth = sanitizedWidth;
    targetHeight = sanitizedHeight;
    target.reset();
    resources.reset();
}

Renderer::Renderer(core::EngineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Renderer::~Renderer() = default;

Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

const core::EngineConfig& Renderer::config() const noexcept { return impl_->config; }

bool Renderer::isRayTracingReady() const noexcept { return impl_->isRayTracingReady(); }

void Renderer::renderFrame() { impl_->renderFrameInternal(true); }

bool Renderer::renderFrameInteractive() { return impl_->renderFrameInternal(false); }

void Renderer::setOutputPath(std::string path) { impl_->setOutputPathInternal(std::move(path)); }

void Renderer::setRenderSize(std::uint32_t width, std::uint32_t height) {
    impl_->setRenderSizeInternal(width, height);
}

bool Renderer::loadScene(const scene::Scene& scene) { return impl_->loadSceneInternal(scene); }

void Renderer::setDebugMode(bool enabled) { impl_->setDebugModeInternal(enabled); }

void Renderer::setShadingMode(const std::string& mode) { impl_->setShadingModeInternal(mode); }

void* Renderer::deviceHandle() const noexcept { return impl_->context.rawDeviceHandle(); }

void* Renderer::commandQueueHandle() const noexcept { return impl_->context.rawCommandQueue(); }

void* Renderer::currentColorTexture() const noexcept { return impl_->currentColorTextureHandle(); }

}  // namespace rtr::rendering
