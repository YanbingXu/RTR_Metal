#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif

#include "RTRMetalEngine/Rendering/Renderer.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/AccelerationStructure.hpp"
#include "RTRMetalEngine/Rendering/AccelerationStructureBuilder.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"
#include "RTRMetalEngine/Rendering/RayTracingPipeline.hpp"
#include "RTRMetalEngine/Rendering/RayTracingShaderTypes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

namespace rtr::rendering {

namespace {

constexpr std::uint32_t kDiagnosticWidth = 512;
constexpr std::uint32_t kDiagnosticHeight = 512;
constexpr std::size_t kRayTracingPixelStride = sizeof(float) * 4;  // RGBA32F

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
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    void reset() {
        accumulationTexture = nil;
        randomTexture = nil;
        resourceBuffer = nil;
        width = 0;
        height = 0;
    }
};

struct Renderer::Impl {
    explicit Impl(core::EngineConfig cfg)
        : config(std::move(cfg)), context(), bufferAllocator(context), geometryStore(bufferAllocator),
          asBuilder(context) {
        if (!context.isValid()) {
            core::Logger::error("Renderer", "Metal context initialization failed");
        } else {
            core::Logger::info("Renderer", "Renderer configured for %s", config.applicationName.c_str());
            context.logDeviceInfo();
            if (!asBuilder.isRayTracingSupported()) {
                core::Logger::warn("Renderer", "Metal device does not report ray tracing support");
            } else {
                if (!rayTracingPipeline.initialize(context, config.shaderLibraryPath)) {
                    core::Logger::warn("Renderer", "Ray tracing pipeline initialization failed");
                }
                buildDiagnosticAccelerationStructure();
            }
        }
    }

    ~Impl() {
        target.reset();
        fallbackRayGenState = nil;
        resources.reset();
        rayTracingUniformBuffer = nil;
    }

    void renderFrame() {
        if (!context.isValid()) {
            core::Logger::warn("Renderer", "Skipping frame: Metal context invalid");
            return;
        }
        const bool targetReady = target.isValid() ? true : ensureOutputTarget(kDiagnosticWidth, kDiagnosticHeight);

        if (isRayTracingReady()) {
            core::Logger::info("Renderer",
                               "Ray tracing target ready (%ux%u RGBA32F)",
                               target.width,
                               target.height);
            if (!dispatchRayTracingPass()) {
                core::Logger::warn("Renderer", "Ray tracing dispatch unavailable; attempting fallback gradient");
                if (!dispatchFallbackGradient()) {
                    core::Logger::warn("Renderer", "Fallback gradient dispatch failed");
                }
            }
        } else {
            core::Logger::info("Renderer", "Ray tracing not ready; executing fallback gradient");
            if (!targetReady || !dispatchFallbackGradient()) {
                core::Logger::warn("Renderer", "Fallback gradient dispatch skipped");
            }
        }
        frameCounter++;
        std::cout << "Renderer frame stub executed using " << context.deviceName() << std::endl;
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
    id<MTLBuffer> rayTracingUniformBuffer = nil;
    uint32_t frameCounter = 0;

    [[nodiscard]] bool isRayTracingReady() const noexcept {
        return context.isValid() && rayTracingPipeline.isValid() && topLevelStructure.isValid();
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
        if (!ensureOutputTarget(kDiagnosticWidth, kDiagnosticHeight)) {
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

        if (!ensureOutputTarget(kDiagnosticWidth, kDiagnosticHeight)) {
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

        updateRayTracingUniforms();

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
        [encoder setBuffer:rayTracingUniformBuffer offset:0 atIndex:0];
        if (resources.resourceBuffer != nil) {
            [encoder setBuffer:resources.resourceBuffer offset:0 atIndex:1];
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

    [[nodiscard]] bool ensureRayTracingUniformBuffer(std::uint32_t width, std::uint32_t height) {
        if (!context.isValid()) {
            return false;
        }

        if (rayTracingUniformBuffer != nil) {
            return true;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
        if (!device) {
            core::Logger::error("Renderer", "Unable to acquire Metal device for uniform buffer");
            return false;
        }

        const NSUInteger length = sizeof(RayTracingUniforms);
        rayTracingUniformBuffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (!rayTracingUniformBuffer) {
            core::Logger::error("Renderer", "Failed to allocate ray tracing uniform buffer");
            return false;
        }

        return true;
    }

    void updateRayTracingUniforms() {
        if (!rayTracingUniformBuffer) {
            return;
        }

        auto* uniforms = reinterpret_cast<RayTracingUniforms*>([rayTracingUniformBuffer contents]);
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

        if (resources.resourceBuffer == nil) {
            const NSUInteger length = sizeof(std::uint32_t) * 4;
            id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
            if (!buffer) {
                core::Logger::error("Renderer", "Failed to allocate ray tracing resource buffer");
                success = false;
            } else {
                resources.resourceBuffer = buffer;
            }
        }

        if (resources.resourceBuffer != nil) {
            auto* data = static_cast<std::uint32_t*>([resources.resourceBuffer contents]);
            if (data) {
                data[0] = static_cast<std::uint32_t>(bottomLevelStructures.size());
                data[1] = resources.width;
                data[2] = resources.height;
                data[3] = frameCounter;
            }
        }

        return success;
    }

    void buildDiagnosticAccelerationStructure() {
        std::array<simd_float3, 3> positions = {
            simd_make_float3(0.0F, 0.0F, 0.0F),
            simd_make_float3(0.5F, 0.0F, 0.0F),
            simd_make_float3(0.0F, 0.5F, 0.0F)};
        std::array<std::uint32_t, 3> indices = {0, 1, 2};

        scene::Scene scene;
        scene::SceneBuilder builder(scene);
        auto meshHandle = builder.addTriangleMesh(positions, indices);
        if (!meshHandle.isValid()) {
            core::Logger::error("Renderer", "Failed to create diagnostic mesh");
            return;
        }
        builder.addDefaultMaterial();
        scene.addInstance(meshHandle, scene::MaterialHandle{0}, matrix_identity_float4x4);

        const auto uploadIndex = geometryStore.uploadMesh(scene.meshes()[meshHandle.index], "diagnostic_triangle");
        if (!uploadIndex.has_value()) {
            core::Logger::warn("Renderer", "Geometry upload failed; skipping diagnostic AS build");
            return;
        }

        void* queueHandle = context.rawCommandQueue();
        const auto& meshBuffers = geometryStore.uploadedMeshes()[*uploadIndex];
        auto blas = asBuilder.buildBottomLevel(meshBuffers, "diagnostic_triangle", queueHandle);
        if (blas.has_value()) {
            core::Logger::info("Renderer", "Built diagnostic BLAS (%zu bytes)", blas->sizeInBytes());
            bottomLevelStructures.push_back(std::move(*blas));

            InstanceBuildInput instance{};
            instance.structure = &bottomLevelStructures.back();
            instance.transform = matrix_identity_float4x4;
            instance.userID = 0;
            instance.mask = 0xFF;

            std::array<InstanceBuildInput, 1> instances = {instance};
            auto tlas = asBuilder.buildTopLevel(instances, "diagnostic_scene", queueHandle);
            if (tlas.has_value()) {
                core::Logger::info("Renderer", "Built diagnostic TLAS (%zu bytes)", tlas->sizeInBytes());
                topLevelStructure = std::move(*tlas);
                const bool targetReady = ensureOutputTarget(kDiagnosticWidth, kDiagnosticHeight);
                const bool pipelineReady = ensureRayTracingResources(kDiagnosticWidth, kDiagnosticHeight);
                if (!targetReady) {
                    core::Logger::warn("Renderer", "Failed to prepare ray tracing target after TLAS build");
                }
                if (!pipelineReady) {
                    core::Logger::warn("Renderer", "Failed to prepare ray tracing resources after TLAS build");
                }
            } else {
                core::Logger::warn("Renderer", "Diagnostic TLAS build skipped");
            }
        } else {
            core::Logger::warn("Renderer", "Diagnostic BLAS build skipped");
        }
    }
};

Renderer::Renderer(core::EngineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Renderer::~Renderer() = default;

Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

const core::EngineConfig& Renderer::config() const noexcept { return impl_->config; }

bool Renderer::isRayTracingReady() const noexcept { return impl_->isRayTracingReady(); }

void Renderer::renderFrame() { impl_->renderFrame(); }

}  // namespace rtr::rendering
