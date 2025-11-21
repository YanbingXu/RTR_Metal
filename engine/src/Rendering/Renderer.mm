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
#include "RTRMetalEngine/MPS/MPSSceneConverter.hpp"
#include "RTRMetalEngine/MPS/MPSUniforms.hpp"
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

float srgbChannelToLinear(float value) {
    if (value <= 0.0f) {
        return 0.0f;
    }
    if (value >= 1.0f) {
        return value;
    }
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return powf((value + 0.055f) / 1.055f, 2.4f);
}

simd_float3 srgbToLinear(simd_float3 colour) {
    return simd_make_float3(srgbChannelToLinear(colour.x),
                            srgbChannelToLinear(colour.y),
                            srgbChannelToLinear(colour.z));
}

uint8_t floatToSRGBByte(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    const long rounded = std::lroundf(clamped * 255.0f);
    return static_cast<uint8_t>(std::clamp<long>(rounded, 0, 255));
}

enum class RayTracingShadingMode {
    Auto,
    HardwareOnly,
}; 

std::string_view shadingModeLabel(RayTracingShadingMode mode) {
    switch (mode) {
    case RayTracingShadingMode::Auto:
        return "auto";
    case RayTracingShadingMode::HardwareOnly:
        return "hardware";
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
        core::Logger::warn("Renderer",
                            "Software/fallback shading modes are no longer available; defaulting to auto");
        return RayTracingShadingMode::Auto;
    }
    return RayTracingShadingMode::Auto;
}

scene::Mesh makeCombinedMeshFromSceneData(const MPSSceneData& sceneData) {
    std::vector<scene::Vertex> vertices;
    vertices.reserve(sceneData.positions.size());
    for (std::size_t i = 0; i < sceneData.positions.size(); ++i) {
        scene::Vertex vertex{};
        const vector_float3 pos = sceneData.positions[i];
        vertex.position = simd_make_float3(pos.x, pos.y, pos.z);
        if (i < sceneData.normals.size()) {
            const vector_float3 normal = sceneData.normals[i];
            vertex.normal = simd_make_float3(normal.x, normal.y, normal.z);
        }
        if (i < sceneData.texcoords.size()) {
            const vector_float2 tex = sceneData.texcoords[i];
            vertex.texcoord = simd_make_float2(tex.x, tex.y);
        }
        vertices.push_back(vertex);
    }

    std::vector<std::uint32_t> indices(sceneData.indices.begin(), sceneData.indices.end());
    return scene::Mesh(std::move(vertices), std::move(indices));
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
    id<MTLTexture> shadeTexture = nil;
    id<MTLBuffer> sceneLimitsBuffer = nil;
    BufferHandle instanceBuffer;
    BufferHandle materialBuffer;
    BufferHandle textureInfoBuffer;
    BufferHandle textureDataBuffer;
    BufferHandle positionsBuffer;
    BufferHandle normalsBuffer;
    BufferHandle colorsBuffer;
    BufferHandle texcoordBuffer;
    BufferHandle indicesBuffer;
    BufferHandle primitiveMaterialBuffer;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    MPSSceneLimits sceneLimits{};

    void reset() {
        accumulationTexture = nil;
        randomTexture = nil;
        shadeTexture = nil;
        sceneLimitsBuffer = nil;
        instanceBuffer = {};
        materialBuffer = {};
        textureInfoBuffer = {};
        textureDataBuffer = {};
        positionsBuffer = {};
        normalsBuffer = {};
        colorsBuffer = {};
        texcoordBuffer = {};
        indicesBuffer = {};
        primitiveMaterialBuffer = {};
        width = 0;
        height = 0;
        sceneLimits = {};
    }

    void resetForResize() {
        accumulationTexture = nil;
        shadeTexture = nil;
        width = 0;
        height = 0;
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
            if (!asBuilder.isRayTracingSupported()) {
                core::Logger::warn("Renderer", "Metal device does not report ray tracing support");
            } else if (!rayTracingPipeline.initialize(context, config.shaderLibraryPath)) {
                core::Logger::warn("Renderer", "Ray tracing pipeline initialization failed");
            }
            if (!initializeScene()) {
                core::Logger::error("Renderer", "Scene initialization failed; ray tracing unavailable");
            }
        }
    }

    ~Impl() {
        target.reset();
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
        const bool hardwarePipelineReady = rayTracingPipeline.isValid();
        const bool logFrame = writeOutput || frameCounter == 0;

        if (hardwarePipelineReady && isRayTracingReady()) {
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
                core::Logger::warn("Renderer", "Hardware ray tracing dispatch failed");
            }
        } else if (hardwarePipelineReady) {
            core::Logger::error("Renderer", "Ray tracing resources are unavailable");
        } else if (shadingMode == RayTracingShadingMode::HardwareOnly) {
            core::Logger::error("Renderer", "Hardware-only mode requested but ray tracing pipeline is invalid");
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
    std::array<id<MTLBuffer>, kUniformRingSize> uniformBuffers = {nil, nil, nil};
    std::size_t uniformBufferCursor = 0;
    uint32_t frameCounter = 0;
    bool accumulationInvalidated = false;
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


    [[nodiscard]] bool dispatchRayTracingPass() {
        if (!rayTracingPipeline.hasHardwareKernels()) {
            core::Logger::warn("Renderer", "Hardware ray tracing kernels unavailable; skipping dispatch");
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

        id<MTLBuffer> positions = (__bridge id<MTLBuffer>)resources.positionsBuffer.nativeHandle();
        id<MTLBuffer> normals = (__bridge id<MTLBuffer>)resources.normalsBuffer.nativeHandle();
        id<MTLBuffer> indices = (__bridge id<MTLBuffer>)resources.indicesBuffer.nativeHandle();
        id<MTLBuffer> colors = (__bridge id<MTLBuffer>)resources.colorsBuffer.nativeHandle();
        id<MTLBuffer> texcoords = (__bridge id<MTLBuffer>)resources.texcoordBuffer.nativeHandle();
        id<MTLBuffer> primitiveMaterials = (__bridge id<MTLBuffer>)resources.primitiveMaterialBuffer.nativeHandle();
        id<MTLBuffer> materialBuffer = (__bridge id<MTLBuffer>)resources.materialBuffer.nativeHandle();
        id<MTLBuffer> textureInfoBuffer = (__bridge id<MTLBuffer>)resources.textureInfoBuffer.nativeHandle();
        id<MTLBuffer> textureDataBuffer = (__bridge id<MTLBuffer>)resources.textureDataBuffer.nativeHandle();

        if (!positions || !indices || !resources.shadeTexture || !resources.sceneLimitsBuffer) {
            core::Logger::warn("Renderer", "Hardware scene buffers unavailable; skipping dispatch");
            return false;
        }

        auto dispatch2D = [&](id<MTLComputePipelineState> pipelineState,
                              auto&& encoderSetup) -> bool {
            if (!pipelineState) {
                return false;
            }
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            if (!encoder) {
                return false;
            }
            [encoder setComputePipelineState:pipelineState];
            encoderSetup(encoder);
            const NSUInteger threadWidth = 8;
            const NSUInteger threadHeight = 8;
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
            MTLSize threadsPerGrid = MTLSizeMake(target.width, target.height, 1);
            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            return true;
        };

        id<MTLComputePipelineState> rayPipeline =
            (__bridge id<MTLComputePipelineState>)rayTracingPipeline.rayPipelineState();
        id<MTLComputePipelineState> accumulatePipeline =
            (__bridge id<MTLComputePipelineState>)rayTracingPipeline.accumulationPipelineState();

        if (!rayPipeline) {
            core::Logger::error("Renderer", "Hardware ray tracing kernel unavailable");
            return false;
        }

        id<MTLAccelerationStructure> accelerationStructure =
            (__bridge id<MTLAccelerationStructure>)topLevelStructure.rawHandle();
        if (!accelerationStructure) {
            core::Logger::error("Renderer", "Top-level acceleration structure is invalid");
            return false;
        }

        if (!dispatch2D(rayPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:uniformBuffer offset:0 atIndex:0];
                [encoder setBuffer:positions offset:0 atIndex:1];
                [encoder setBuffer:normals offset:0 atIndex:2];
                [encoder setBuffer:indices offset:0 atIndex:3];
                [encoder setBuffer:colors offset:0 atIndex:4];
                [encoder setBuffer:texcoords offset:0 atIndex:5];
                [encoder setBuffer:primitiveMaterials offset:0 atIndex:6];
                [encoder setBuffer:materialBuffer offset:0 atIndex:7];
                [encoder setBuffer:textureInfoBuffer offset:0 atIndex:8];
                [encoder setBuffer:textureDataBuffer offset:0 atIndex:9];
                [encoder setBuffer:resources.sceneLimitsBuffer offset:0 atIndex:10];
                if (resources.randomTexture) {
                    [encoder setTexture:resources.randomTexture atIndex:0];
                }
                [encoder setTexture:resources.shadeTexture atIndex:1];
                [encoder setAccelerationStructure:accelerationStructure atBufferIndex:15];
            })) {
            core::Logger::error("Renderer", "Failed to encode hardware ray tracing kernel");
            return false;
        }

        const bool doAccumulate = accumulationEnabledThisFrame();
        if (doAccumulate && resources.accumulationTexture && accumulatePipeline != nil) {
            if (!dispatch2D(accumulatePipeline, [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:uniformBuffer offset:0 atIndex:0];
                    [encoder setTexture:resources.shadeTexture atIndex:0];
                    [encoder setTexture:resources.accumulationTexture atIndex:1];
                    [encoder setTexture:target.colorTexture atIndex:2];
                })) {
                core::Logger::error("Renderer", "Failed to encode accumulate kernel");
                return false;
            }

            id<MTLBlitCommandEncoder> copyAccum = [commandBuffer blitCommandEncoder];
            if (copyAccum) {
                MTLOrigin origin = MTLOriginMake(0, 0, 0);
                MTLSize size = MTLSizeMake(target.width, target.height, 1);
                [copyAccum copyFromTexture:target.colorTexture
                                sourceSlice:0
                                sourceLevel:0
                               sourceOrigin:origin
                                 sourceSize:size
                                  toTexture:resources.accumulationTexture
                           destinationSlice:0
                           destinationLevel:0
                          destinationOrigin:origin];
                [copyAccum endEncoding];
            }
        } else {
            id<MTLBlitCommandEncoder> blitColor = [commandBuffer blitCommandEncoder];
            if (blitColor) {
                MTLOrigin origin = MTLOriginMake(0, 0, 0);
                MTLSize size = MTLSizeMake(target.width, target.height, 1);
                [blitColor copyFromTexture:resources.shadeTexture
                                sourceSlice:0
                                sourceLevel:0
                               sourceOrigin:origin
                                 sourceSize:size
                                  toTexture:target.colorTexture
                           destinationSlice:0
                           destinationLevel:0
                          destinationOrigin:origin];
                [blitColor endEncoding];
            }
        }

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

            id<MTLBuffer> buffer = [device newBufferWithLength:sizeof(HardwareRayUniforms)
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

        auto* uniforms = reinterpret_cast<HardwareRayUniforms*>([buffer contents]);
        if (!uniforms) {
            return;
        }

        // Camera approximating the reference Cornell composition: eye in front of the box, looking inward with a ~45°
        // vertical FOV and aspect-derived image plane.
        const float aspect = (target.height > 0) ? static_cast<float>(target.width) / static_cast<float>(target.height)
                                                 : 1.0f;
        const float fovY = 45.0f * (M_PI / 180.0f);
        const float halfHeight = tanf(fovY * 0.5f);
        const float halfWidth = halfHeight * aspect;

        simd_float3 eye = simd_make_float3(0.0f, 1.0f, 3.38f);
        simd_float3 targetPoint = simd_make_float3(0.0f, 1.0f, 0.0f);
        simd_float3 forward = simd_normalize(targetPoint - eye);
        simd_float3 globalUp = simd_make_float3(0.0f, 1.0f, 0.0f);
        simd_float3 right = simd_normalize(simd_cross(forward, globalUp));
        simd_float3 up = simd_cross(right, forward);

        uniforms->camera.eye = simd_make_float4(eye, 1.0f);
        uniforms->camera.forward = simd_make_float4(forward, 0.0f);
        uniforms->camera.right = simd_make_float4(right, 0.0f);
        uniforms->camera.up = simd_make_float4(up, 0.0f);
        uniforms->camera.imagePlaneHalfExtents = simd_make_float2(halfWidth, halfHeight);
        uniforms->camera.width = target.width;
        uniforms->camera.height = target.height;
        uniforms->camera.frameIndex = frameIndexForUniforms();
        std::uint32_t flags = 0u;
        if (debugAlbedo) {
            flags |= RTR_RAY_FLAG_DEBUG;
        }
        if (accumulationEnabledThisFrame()) {
            flags |= RTR_RAY_FLAG_ACCUMULATE;
        }
        uniforms->camera.flags = flags;
        uniforms->camera.samplesPerPixel = config.samplesPerPixel;
        uniforms->camera.sampleSeed = config.sampleSeed;

        uniforms->lightCount = 1u;
        uniforms->maxBounces = std::max<std::uint32_t>(1u, config.maxHardwareBounces);

        HardwareAreaLight& light = uniforms->lights[0];
        light.position = simd_make_float4(0.0f, 1.99f, 0.0f, 1.0f);
        light.right = simd_make_float4(0.25f, 0.0f, 0.0f, 0.0f);
        light.up = simd_make_float4(0.0f, 0.0f, 0.25f, 0.0f);
        light.forward = simd_make_float4(0.0f, -1.0f, 0.0f, 0.0f);
        light.color = simd_make_float4(4.0f, 4.0f, 4.0f, 0.0f);

        if (debugAlbedo) {
            core::Logger::info("Renderer", "Debug uniforms: flags=0x%x", uniforms->camera.flags);
        }

        [buffer didModifyRange:NSMakeRange(0, sizeof(HardwareRayUniforms))];
    }

    void resetAccumulationInternal() {
        frameCounter = 0;
        accumulationInvalidated = true;
    }

    [[nodiscard]] std::uint32_t frameIndexForUniforms() const {
        std::uint32_t index = frameCounter;
        if (config.samplesPerPixel > 0) {
            const std::uint32_t maxSample = config.samplesPerPixel - 1u;
            index = std::min(index, maxSample);
        }
        if (config.accumulationFrames > 0) {
            const std::uint32_t maxFrame = config.accumulationFrames - 1u;
            index = std::min(index, maxFrame);
        }
        return index;
    }

    bool accumulationEnabledThisFrame() const {
        if (!config.accumulationEnabled || resources.accumulationTexture == nil) {
            return false;
        }
        if (config.accumulationFrames > 0 && frameCounter >= config.accumulationFrames) {
            return false;
        }
        if (config.samplesPerPixel > 0 && frameCounter >= config.samplesPerPixel) {
            return false;
        }
        return true;
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
            return false;
        }

        bool success = true;

        if (resources.randomTexture == nil) {
            constexpr NSUInteger kRandomTextureWidth = 256;
            constexpr NSUInteger kRandomTextureHeight = 256;
            MTLTextureDescriptor* randomDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint
                                                                                                   width:kRandomTextureWidth
                                                                                                  height:kRandomTextureHeight
                                                                                               mipmapped:NO];
            randomDesc.storageMode = MTLStorageModeShared;
            randomDesc.usage = MTLTextureUsageShaderRead;
            id<MTLTexture> randomTexture = [device newTextureWithDescriptor:randomDesc];
            if (!randomTexture) {
                core::Logger::error("Renderer", "Failed to allocate random texture");
                success = false;
            } else {
                std::vector<std::uint32_t> randomData(kRandomTextureWidth * kRandomTextureHeight * 4);
                std::mt19937 rng(static_cast<std::uint32_t>(std::random_device{}()));
                std::uniform_int_distribution<std::uint32_t> dist(0, std::numeric_limits<std::uint32_t>::max());
                for (auto& value : randomData) {
                    value = dist(rng);
                }
                MTLRegion region = MTLRegionMake2D(0, 0, kRandomTextureWidth, kRandomTextureHeight);
                const NSUInteger bytesPerRow = kRandomTextureWidth * sizeof(std::uint32_t) * 4;
                [randomTexture replaceRegion:region
                                 mipmapLevel:0
                                   withBytes:randomData.data()
                                 bytesPerRow:bytesPerRow];
                resources.randomTexture = randomTexture;
            }
        }

        auto ensureTexture = [&](id<MTLTexture>& texture, MTLPixelFormat format, NSString* label) {
            if (texture != nil && resources.width == width && resources.height == height) {
                return true;
            }
            MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                            width:width
                                                                                           height:height
                                                                                        mipmapped:NO];
            desc.storageMode = MTLStorageModePrivate;
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            id<MTLTexture> newTexture = [device newTextureWithDescriptor:desc];
            if (!newTexture) {
                core::Logger::error("Renderer", "Failed to allocate %s (%ux%u)", label.UTF8String, width, height);
                return false;
            }
            [newTexture setLabel:label];
            texture = newTexture;
            return true;
        };

        if (!ensureTexture(resources.shadeTexture, MTLPixelFormatRGBA32Float, @"rtr.hw.lighting")) {
            success = false;
        }

        const bool needsAccumulation = accumulationEnabledThisFrame();
        if (needsAccumulation) {
            if (!ensureTexture(resources.accumulationTexture, MTLPixelFormatRGBA32Float, @"rtr.hw.accum")) {
                success = false;
            }
        } else if (resources.accumulationTexture != nil && (resources.width != width || resources.height != height)) {
            resources.accumulationTexture = nil;
        }

        if (resources.sceneLimitsBuffer == nil) {
            resources.sceneLimitsBuffer = [device newBufferWithLength:sizeof(MPSSceneLimits)
                                                             options:MTLResourceStorageModeShared];
            if (!resources.sceneLimitsBuffer) {
                core::Logger::error("Renderer", "Failed to allocate scene limits buffer");
                success = false;
            }
        }

        if (resources.sceneLimitsBuffer != nil) {
            std::memcpy([resources.sceneLimitsBuffer contents], &resources.sceneLimits, sizeof(MPSSceneLimits));
            [resources.sceneLimitsBuffer didModifyRange:NSMakeRange(0, sizeof(MPSSceneLimits))];
        }

        resources.width = width;
        resources.height = height;

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

        resources.sceneLimits.textureCount = static_cast<std::uint32_t>(textureResources.size());

        if (success) {
            resources.width = width;
            resources.height = height;
        }

        return success;
    }

    bool initializeScene();
    bool loadSceneInternal(const scene::Scene& scene);
    bool prepareHardwareSceneData(const MPSSceneData& sceneData);
    void writeRayTracingOutput() const;
    void setOutputPathInternal(std::string path);
    void setRenderSizeInternal(std::uint32_t width, std::uint32_t height);
    void setDebugModeInternal(bool enabled) {
        if (debugAlbedo == enabled) {
            return;
        }
        debugAlbedo = enabled;
        resetAccumulationInternal();
    }

    void setShadingModeInternal(const std::string& value) {
        const RayTracingShadingMode newMode = parseShadingMode(value);
        config.shadingMode = value;
        if (newMode == shadingMode) {
            return;
        }

        shadingMode = newMode;

        if (!rayTracingPipeline.isValid()) {
            if (!asBuilder.isRayTracingSupported()) {
                core::Logger::warn("Renderer", "Metal device does not report ray tracing support");
            } else if (!rayTracingPipeline.initialize(context, config.shaderLibraryPath)) {
                core::Logger::warn("Renderer", "Ray tracing pipeline initialization failed for shading mode switch");
            }
        }

        resetAccumulationInternal();
    }
};

bool Renderer::Impl::prepareHardwareSceneData(const MPSSceneData& sceneData) {

    auto uploadSceneBuffer = [&](BufferHandle& handle, const void* data, std::size_t byteLength, const char* label) {
        if (byteLength == 0 || data == nullptr) {
            handle = {};
            return;
        }
        if (!handle.isValid() || handle.length() < byteLength) {
            handle = bufferAllocator.createBuffer(byteLength, data, label);
        } else {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle.nativeHandle();
            if (buffer) {
                std::memcpy([buffer contents], data, byteLength);
                [buffer didModifyRange:NSMakeRange(0, byteLength)];
            }
        }
    };

    uploadSceneBuffer(resources.positionsBuffer,
                      sceneData.positions.empty() ? nullptr : sceneData.positions.data(),
                      sceneData.positions.size() * sizeof(vector_float3),
                      "rtr.hw.positions");
    uploadSceneBuffer(resources.normalsBuffer,
                      sceneData.normals.empty() ? nullptr : sceneData.normals.data(),
                      sceneData.normals.size() * sizeof(vector_float3),
                      "rtr.hw.normals");
    uploadSceneBuffer(resources.colorsBuffer,
                      sceneData.colors.empty() ? nullptr : sceneData.colors.data(),
                      sceneData.colors.size() * sizeof(vector_float3),
                      "rtr.hw.colors");
    uploadSceneBuffer(resources.texcoordBuffer,
                      sceneData.texcoords.empty() ? nullptr : sceneData.texcoords.data(),
                      sceneData.texcoords.size() * sizeof(vector_float2),
                      "rtr.hw.texcoords");
    uploadSceneBuffer(resources.indicesBuffer,
                      sceneData.indices.empty() ? nullptr : sceneData.indices.data(),
                      sceneData.indices.size() * sizeof(std::uint32_t),
                      "rtr.hw.indices");
    uploadSceneBuffer(resources.primitiveMaterialBuffer,
                      sceneData.primitiveMaterials.empty() ? nullptr : sceneData.primitiveMaterials.data(),
                      sceneData.primitiveMaterials.size() * sizeof(std::uint32_t),
                      "rtr.hw.primitiveMaterials");

    resources.sceneLimits.vertexCount = static_cast<std::uint32_t>(sceneData.positions.size());
    resources.sceneLimits.indexCount = static_cast<std::uint32_t>(sceneData.indices.size());
    resources.sceneLimits.colorCount = static_cast<std::uint32_t>(sceneData.colors.size());
    resources.sceneLimits.primitiveCount = static_cast<std::uint32_t>(sceneData.indices.size() / 3u);
    resources.sceneLimits.normalCount = static_cast<std::uint32_t>(sceneData.normals.size());
    resources.sceneLimits.texcoordCount = static_cast<std::uint32_t>(sceneData.texcoords.size());
    resources.sceneLimits.materialCount = static_cast<std::uint32_t>(materialResources.size());
    resources.sceneLimits.textureCount = static_cast<std::uint32_t>(textureResources.size());

    return true;
}


bool Renderer::Impl::initializeScene() {
    scene::Scene scene = scene::createCornellBoxScene();
    if (!loadSceneInternal(scene)) {
        core::Logger::warn("Renderer", "Scene load failed; hardware ray tracing unavailable");
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
    geometryStore.clear();

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

    const auto& materials = scene.materials();

    const MPSSceneData sceneData = buildSceneData(scene);
    geometryStore.clear();
    if (sceneData.positions.empty() || sceneData.indices.empty() || sceneData.indexOffsets.empty()) {
        core::Logger::warn("Renderer", "Flattened scene data empty; scene load aborted");
        return false;
    }

    std::vector<std::size_t> meshUploadIndices(1, static_cast<std::size_t>(-1));
    std::vector<std::size_t> meshBLASIndices(1, static_cast<std::size_t>(-1));

    geometryStore.clear();

    scene::Mesh flattenedMesh = makeCombinedMeshFromSceneData(sceneData);
    const auto uploadIndex = geometryStore.uploadMesh(flattenedMesh, "scene_mesh_combined");
    if (!uploadIndex.has_value()) {
        core::Logger::error("Renderer", "Failed to upload flattened mesh for scene");
        return false;
    }

    meshUploadIndices[0] = *uploadIndex;
    const auto& meshBuffers = geometryStore.uploadedMeshes()[*uploadIndex];
    auto blas = asBuilder.buildBottomLevel(meshBuffers, "scene_mesh_combined", queueHandle);
    if (!blas.has_value()) {
        core::Logger::error("Renderer", "Failed to build BLAS for flattened mesh");
        return false;
    }

    meshBLASIndices[0] = bottomLevelStructures.size();
    bottomLevelStructures.push_back(std::move(*blas));

    std::vector<InstanceBuildInput> instanceInputs;
    instanceInputs.reserve(1);
    instanceResources.clear();
    instanceResources.reserve(1);

    InstanceBuildInput input{};
    input.structure = &bottomLevelStructures.back();
    input.transform = matrix_identity_float4x4;
    input.userID = 0u;
    input.mask = RTR_TRIANGLE_MASK_GEOMETRY;
    instanceInputs.push_back(input);

    RayTracingInstanceResource resource{};
    resource.objectToWorld = matrix_identity_float4x4;
    resource.worldToObject = matrix_identity_float4x4;
    resource.meshIndex = static_cast<std::uint32_t>(*uploadIndex);
    resource.materialIndex = 0u;
    instanceResources.push_back(resource);

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
        resource.albedo = srgbToLinear(material.albedo);
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
                resource.albedo = simd_make_float3(1.0f, 1.0f, 1.0f);
            } else {
                core::Logger::warn("Renderer",
                                   "Failed to load texture '%s' for material",
                                   material.albedoTexturePath.c_str());
            }
        }
        materialResources.push_back(resource);
    }

    resources.textureCount = textureResources.size();

    const bool hardwareSceneReady = prepareHardwareSceneData(sceneData);
    if (!hardwareSceneReady) {
        core::Logger::warn("Renderer", "Failed to prepare hardware scene data");
    }

    bool pipelineReady = false;
    if (hardwareSceneReady) {
        pipelineReady = ensureRayTracingResources(targetWidth, targetHeight);
        if (!pipelineReady) {
            core::Logger::warn("Renderer", "Failed to allocate ray tracing resources for scene");
        }
    }

    const bool loaded = topLevelStructure.isValid() && hardwareSceneReady && pipelineReady;
    if (loaded) {
        resetAccumulationInternal();
    }
    return loaded;
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
    resources.resetForResize();
    resetAccumulationInternal();
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

void Renderer::resetAccumulation() { impl_->resetAccumulationInternal(); }

void* Renderer::deviceHandle() const noexcept { return impl_->context.rawDeviceHandle(); }

void* Renderer::commandQueueHandle() const noexcept { return impl_->context.rawCommandQueue(); }

void* Renderer::currentColorTexture() const noexcept { return impl_->currentColorTextureHandle(); }

}  // namespace rtr::rendering
