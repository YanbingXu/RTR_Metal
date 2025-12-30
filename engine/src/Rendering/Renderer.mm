#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Metal/MTLCaptureManager.h>
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif

#include "RTRMetalEngine/Rendering/Renderer.hpp"

#include "RTRMetalEngine/Core/ImageLoader.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Core/Math.hpp"
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
#include <cmath>
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
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdlib>

#include "RTRMetalEngine/Scene/CornellBox.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

namespace rtr::rendering {

namespace {

constexpr std::uint32_t kDiagnosticWidth = 512;
constexpr std::uint32_t kDiagnosticHeight = 512;
constexpr std::size_t kRayTracingPixelStride = sizeof(float) * 4;  // RGBA32F
constexpr float kDefaultVerticalFovDegrees = 43.0f;

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

scene::Mesh makeMeshFromRange(const MPSSceneData& sceneData, const MPSMeshRange& range) {
    if (range.vertexCount == 0 || range.indexCount == 0) {
        return scene::Mesh();
    }

    const std::size_t vertexEnd = static_cast<std::size_t>(range.vertexOffset) + range.vertexCount;
    const std::size_t indexEnd = static_cast<std::size_t>(range.indexOffset) + range.indexCount;
    if (vertexEnd > sceneData.positions.size() || indexEnd > sceneData.indices.size()) {
        return scene::Mesh();
    }

    std::vector<scene::Vertex> vertices;
    vertices.reserve(range.vertexCount);
    for (std::size_t i = 0; i < range.vertexCount; ++i) {
        const std::size_t sourceIndex = static_cast<std::size_t>(range.vertexOffset) + i;
        scene::Vertex vertex{};
        const vector_float3 pos = sceneData.positions[sourceIndex];
        vertex.position = simd_make_float3(pos.x, pos.y, pos.z);
        if (sourceIndex < sceneData.normals.size()) {
            const vector_float3 normal = sceneData.normals[sourceIndex];
            vertex.normal = simd_make_float3(normal.x, normal.y, normal.z);
        }
        if (sourceIndex < sceneData.texcoords.size()) {
            const vector_float2 tex = sceneData.texcoords[sourceIndex];
            vertex.texcoord = simd_make_float2(tex.x, tex.y);
        }
        vertices.push_back(vertex);
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(range.indexCount);
    for (std::size_t i = 0; i < range.indexCount; ++i) {
        const std::size_t idx = static_cast<std::size_t>(range.indexOffset) + i;
        const std::uint32_t globalIndex = sceneData.indices[idx];
        if (globalIndex < range.vertexOffset ||
            globalIndex >= range.vertexOffset + range.vertexCount) {
            return scene::Mesh();
        }
        indices.push_back(globalIndex - range.vertexOffset);
    }

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
    id<MTLTexture> accumulationHistoryTexture = nil;
    id<MTLTexture> randomTexture = nil;
    id<MTLTexture> shadeTexture = nil;
    id<MTLBuffer> sceneLimitsBuffer = nil;
    BufferHandle instanceBuffer;
    BufferHandle materialBuffer;
    BufferHandle meshResourceBuffer;
    BufferHandle textureInfoBuffer;
    BufferHandle textureDataBuffer;
    BufferHandle positionsBuffer;
    BufferHandle normalsBuffer;
    BufferHandle colorsBuffer;
    BufferHandle texcoordBuffer;
    BufferHandle indicesBuffer;
    BufferHandle hitDebugBuffer;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    MPSSceneLimits sceneLimits{};

    void reset() {
        accumulationHistoryTexture = nil;
        randomTexture = nil;
        shadeTexture = nil;
        sceneLimitsBuffer = nil;
        instanceBuffer = {};
        materialBuffer = {};
        meshResourceBuffer = {};
        textureInfoBuffer = {};
        textureDataBuffer = {};
        positionsBuffer = {};
        normalsBuffer = {};
        colorsBuffer = {};
        texcoordBuffer = {};
        indicesBuffer = {};
        hitDebugBuffer = {};
        width = 0;
        height = 0;
        sceneLimits = {};
    }

    void resetForResize() {
        accumulationHistoryTexture = nil;
        shadeTexture = nil;
        width = 0;
        height = 0;
    }
};

constexpr std::size_t kUniformRingSize = 3;

struct CameraRig {
    simd_float3 eye = simd_make_float3(0.0f, 1.2f, 5.4f);
    simd_float3 target = simd_make_float3(0.0f, 1.0f, 0.0f);
};

struct Renderer::Impl {
    explicit Impl(core::EngineConfig cfg)
        : config(std::move(cfg)), context(), bufferAllocator(context), geometryStore(bufferAllocator),
          asBuilder(context), shadingMode(parseShadingMode(config.shadingMode)) {
        if (!context.isValid()) {
            core::Logger::error("Renderer", "Metal context initialization failed");
        } else {
            core::Logger::info("Renderer", "Renderer configured for %s", config.applicationName.c_str());
            context.logDeviceInfo();
            const char* captureFlag = std::getenv("RTR_METAL_CAPTURE");
            if (captureFlag && captureFlag[0] != '\0' && std::strcmp(captureFlag, "0") != 0) {
                metalCaptureEnabled = true;
                std::filesystem::path capturePath = std::filesystem::current_path() / "metal_capture.gputrace";
                metalCaptureOutputPath = capturePath.string();
                core::Logger::info("Renderer",
                                   "Metal capture enabled via RTR_METAL_CAPTURE (output=%s)",
                                   metalCaptureOutputPath.c_str());
            }
            if (!asBuilder.isRayTracingSupported()) {
                core::Logger::warn("Renderer", "Metal device does not report ray tracing support");
            } else if (!rayTracingPipeline.initialize(context, config.shaderLibraryPath)) {
                core::Logger::warn("Renderer", "Ray tracing pipeline initialization failed");
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
            frameCounter++;
        } else if (logFrame) {
            core::Logger::warn("Renderer", "No renderable output produced this frame");
        }

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
    uint32_t accumulationFrameIndex = 0;
    bool accumulationInvalidated = false;
    std::vector<RayTracingInstanceResource> instanceResources;
    std::vector<RayTracingMeshResource> meshResources;
    std::vector<RayTracingMaterialResource> materialResources;
    std::vector<RayTracingTextureResource> textureResources;
    std::vector<float> texturePixels;
    std::string outputPath = "renderer_output.ppm";
    std::uint32_t targetWidth = kDiagnosticWidth;
    std::uint32_t targetHeight = kDiagnosticHeight;
    bool debugAlbedo = false;
    RayTracingShadingMode shadingMode = RayTracingShadingMode::Auto;
    bool metalCaptureEnabled = false;
    bool metalCaptureInProgress = false;
    bool metalCaptureCompleted = false;
    std::string metalCaptureOutputPath;
    CameraRig cameraRig;
    core::math::BoundingBox sceneBounds = core::math::BoundingBox::makeEmpty();
    bool hitDebugLogged = false;

    [[nodiscard]] bool isRayTracingReady() const noexcept {
        return context.isValid() && rayTracingPipeline.isValid() && topLevelStructure.isValid();
    }

    [[nodiscard]] static CameraRig makeReferenceCameraRig() {
        return CameraRig{};
    }

    [[nodiscard]] bool hasValidSceneBounds() const noexcept {
        const simd_float3 minPoint = sceneBounds.min;
        const simd_float3 maxPoint = sceneBounds.max;
        return std::isfinite(minPoint.x) && std::isfinite(minPoint.y) && std::isfinite(minPoint.z) &&
               std::isfinite(maxPoint.x) && std::isfinite(maxPoint.y) && std::isfinite(maxPoint.z) &&
               (minPoint.x <= maxPoint.x) && (minPoint.y <= maxPoint.y) && (minPoint.z <= maxPoint.z);
    }

    void updateCameraRigFromBounds() {
        if (!hasValidSceneBounds()) {
            cameraRig = makeReferenceCameraRig();
            return;
        }

        const simd_float3 center = (sceneBounds.min + sceneBounds.max) * 0.5f;
        const simd_float3 extent = sceneBounds.extent();
        const float sceneWidth = std::max(extent.x, 1.0f);
        const float sceneHeight = std::max(extent.y, 1.0f);
        const float sceneDepth = std::max(extent.z, 1.0f);

        const float aspect = (targetHeight > 0u) ? static_cast<float>(targetWidth) / static_cast<float>(targetHeight)
                                                 : 1.0f;
        const float fovY = rtr::core::math::radians(kDefaultVerticalFovDegrees);
        const float halfFovY = std::max(fovY * 0.5f, 1.0e-3f);
        const float halfFovX = std::max(std::atan(std::tan(halfFovY) * aspect), 1.0e-3f);

        constexpr float kVerticalCoverage = 0.6f;
        constexpr float kHorizontalCoverage = 1.0f;
        const float verticalDistance = (sceneHeight * kVerticalCoverage) * 0.5f / std::tan(halfFovY);
        const float horizontalDistance = (sceneWidth * kHorizontalCoverage) * 0.5f / std::tan(halfFovX);
        const float depthMargin = sceneDepth * 0.75f + sceneWidth * 0.35f;
        const float aspectScale = std::max(aspect, 1.0f);
        const float distanceFromFrontWall = std::max({verticalDistance, horizontalDistance, depthMargin}) * aspectScale;

        CameraRig rig;
        rig.target = simd_make_float3(center.x, center.y, center.z);
        rig.eye = simd_make_float3(center.x,
                                   center.y + sceneHeight * 0.08f,
                                   sceneBounds.max.z + distanceFromFrontWall);
        cameraRig = rig;

        core::Logger::info("Renderer",
                            "Camera rig eye=(%.3f, %.3f, %.3f) target=(%.3f, %.3f, %.3f) aspect=%.3f",
                            rig.eye.x,
                            rig.eye.y,
                            rig.eye.z,
                            rig.target.x,
                            rig.target.y,
                            rig.target.z,
                            aspect);
    }


    [[nodiscard]] bool dispatchRayTracingPass() {
        if (!rayTracingPipeline.isValid()) {
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

        startMetalCaptureIfNeeded(queue);
        auto fail = [&]() {
            stopMetalCaptureIfNeeded();
            return false;
        };

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) {
            core::Logger::error("Renderer", "Failed to create command buffer for ray tracing dispatch");
            return fail();
        }

        id<MTLBuffer> positions = (__bridge id<MTLBuffer>)resources.positionsBuffer.nativeHandle();
        id<MTLBuffer> normals = (__bridge id<MTLBuffer>)resources.normalsBuffer.nativeHandle();
        id<MTLBuffer> indices = (__bridge id<MTLBuffer>)resources.indicesBuffer.nativeHandle();
        id<MTLBuffer> colors = (__bridge id<MTLBuffer>)resources.colorsBuffer.nativeHandle();
        id<MTLBuffer> texcoords = (__bridge id<MTLBuffer>)resources.texcoordBuffer.nativeHandle();
        id<MTLBuffer> meshResourcesBuffer =
            resources.meshResourceBuffer.isValid() ? (__bridge id<MTLBuffer>)resources.meshResourceBuffer.nativeHandle() : nil;
        id<MTLBuffer> materialBuffer = resources.materialBuffer.isValid() ? (__bridge id<MTLBuffer>)resources.materialBuffer.nativeHandle() : nil;
        id<MTLBuffer> textureInfoBuffer = resources.textureInfoBuffer.isValid() ? (__bridge id<MTLBuffer>)resources.textureInfoBuffer.nativeHandle() : nil;
        id<MTLBuffer> textureDataBuffer = resources.textureDataBuffer.isValid() ? (__bridge id<MTLBuffer>)resources.textureDataBuffer.nativeHandle() : nil;
        id<MTLBuffer> instanceBuffer = resources.instanceBuffer.isValid() ? (__bridge id<MTLBuffer>)resources.instanceBuffer.nativeHandle() : nil;

        id<MTLBuffer> hitDebug = (__bridge id<MTLBuffer>)resources.hitDebugBuffer.nativeHandle();

        if (!positions || !indices || !resources.shadeTexture || !resources.sceneLimitsBuffer || hitDebug == nil) {
            core::Logger::warn("Renderer", "Hardware scene buffers unavailable; skipping dispatch");
            return fail();
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
            return fail();
        }

        id<MTLAccelerationStructure> accelerationStructure =
            (__bridge id<MTLAccelerationStructure>)topLevelStructure.rawHandle();
        if (!accelerationStructure) {
            core::Logger::error("Renderer", "Top-level acceleration structure is invalid");
            return fail();
        }
        const NSUInteger tlasSize = [accelerationStructure size];
        core::Logger::info("Renderer",
                           "TLAS handle=%p size=%lu bytes",
                           accelerationStructure,
                           static_cast<unsigned long>(tlasSize));

        if (!dispatch2D(rayPipeline, [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:uniformBuffer offset:0 atIndex:0];
                [encoder setBuffer:positions offset:0 atIndex:1];
                [encoder setBuffer:normals offset:0 atIndex:2];
                [encoder setBuffer:indices offset:0 atIndex:3];
                [encoder setBuffer:colors offset:0 atIndex:4];
                [encoder setBuffer:texcoords offset:0 atIndex:5];
                if (meshResourcesBuffer) {
                    [encoder setBuffer:meshResourcesBuffer offset:0 atIndex:6];
                }
                if (materialBuffer) {
                    [encoder setBuffer:materialBuffer offset:0 atIndex:7];
                }
                if (textureInfoBuffer) {
                    [encoder setBuffer:textureInfoBuffer offset:0 atIndex:8];
                }
                if (textureDataBuffer) {
                    [encoder setBuffer:textureDataBuffer offset:0 atIndex:9];
                }
                [encoder setBuffer:resources.sceneLimitsBuffer offset:0 atIndex:10];
                [encoder setBuffer:hitDebug offset:0 atIndex:11];
                if (instanceBuffer) {
                    [encoder setBuffer:instanceBuffer offset:0 atIndex:12];
                }
                if (resources.randomTexture) {
                    [encoder setTexture:resources.randomTexture atIndex:0];
                }
                [encoder setTexture:resources.shadeTexture atIndex:1];
                [encoder setAccelerationStructure:accelerationStructure atBufferIndex:15];
                if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                    [encoder useResource:accelerationStructure usage:MTLResourceUsageRead];
                    [encoder useResource:positions usage:MTLResourceUsageRead];
                    if (normals) {
                        [encoder useResource:normals usage:MTLResourceUsageRead];
                    }
                    [encoder useResource:indices usage:MTLResourceUsageRead];
                    if (colors) {
                        [encoder useResource:colors usage:MTLResourceUsageRead];
                    }
                    if (texcoords) {
                        [encoder useResource:texcoords usage:MTLResourceUsageRead];
                    }
                    if (meshResourcesBuffer) {
                        [encoder useResource:meshResourcesBuffer usage:MTLResourceUsageRead];
                    }
                    if (materialBuffer) {
                        [encoder useResource:materialBuffer usage:MTLResourceUsageRead];
                    }
                    if (textureInfoBuffer) {
                        [encoder useResource:textureInfoBuffer usage:MTLResourceUsageRead];
                    }
                    if (textureDataBuffer) {
                        [encoder useResource:textureDataBuffer usage:MTLResourceUsageRead];
                    }
                    if (resources.sceneLimitsBuffer) {
                        [encoder useResource:resources.sceneLimitsBuffer usage:MTLResourceUsageRead];
                    }
                    if (instanceBuffer) {
                        [encoder useResource:instanceBuffer usage:MTLResourceUsageRead];
                    }
                }
            })) {
            core::Logger::error("Renderer", "Failed to encode hardware ray tracing kernel");
            return fail();
        }

        const bool doAccumulate = accumulationEnabledThisFrame();
        bool accumulationDispatched = false;
        core::Logger::info("Renderer",
                           "Accumulation state: enabled=%s frameIndex=%u limit=%u history=%s",
                           doAccumulate ? "true" : "false",
                           accumulationFrameIndex,
                           accumulationLimit(),
                           resources.accumulationHistoryTexture ? "yes" : "no");
        if (doAccumulate && resources.accumulationHistoryTexture && accumulatePipeline != nil) {
            if (!dispatch2D(accumulatePipeline, [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:uniformBuffer offset:0 atIndex:0];
                    [encoder setTexture:resources.shadeTexture atIndex:0];
                    [encoder setTexture:resources.accumulationHistoryTexture atIndex:1];
                    [encoder setTexture:target.colorTexture atIndex:2];
                })) {
                core::Logger::error("Renderer", "Failed to encode accumulate kernel");
                return fail();
            }

            accumulationDispatched = true;
        }

        if (!doAccumulate) {
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

        if (accumulationDispatched) {
            if (accumulationFrameIndex < std::numeric_limits<std::uint32_t>::max()) {
                accumulationFrameIndex++;
            }
            if (resources.accumulationHistoryTexture && accumulationFrameIndex <= 4) {
                core::Logger::info("Renderer",
                                    "Accumulation frame advanced to %u (history texture %p)",
                                    accumulationFrameIndex,
                                    resources.accumulationHistoryTexture);
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
        MTLCommandBufferStatus status = [commandBuffer status];
        if (status != MTLCommandBufferStatusCompleted) {
            NSError* error = commandBuffer.error;
            const char* message = error ? error.localizedDescription.UTF8String : "unknown error";
            core::Logger::error("Renderer",
                                "Command buffer completed with status=%ld (%s)",
                                static_cast<long>(status),
                                message);
            stopMetalCaptureIfNeeded();
            return false;
        }

        if (!hitDebugLogged && resources.hitDebugBuffer.isValid()) {
            const bool debugInstanceTrace = std::getenv("RTR_DEBUG_INSTANCE_TRACE") != nullptr;
            id<MTLBuffer> hitBuffer = (__bridge id<MTLBuffer>)resources.hitDebugBuffer.nativeHandle();
            if (hitBuffer) {
                const std::uint32_t* hits = static_cast<const std::uint32_t*>([hitBuffer contents]);
                if (hits) {
                    const std::size_t totalPixels =
                        static_cast<std::size_t>(resources.width) * static_cast<std::size_t>(resources.height);
                    if (debugInstanceTrace) {
                        std::vector<std::size_t> hitCounts(meshResources.size(), 0u);
                        for (std::size_t i = 0; i < totalPixels; ++i) {
                            const std::uint32_t value = hits[i];
                            if (value == 0u) {
                                continue;
                            }
                            const std::size_t meshIndex = static_cast<std::size_t>(value - 1u);
                            if (meshIndex < hitCounts.size()) {
                                hitCounts[meshIndex]++;
                            }
                        }
                        std::string hitLog;
                        for (std::size_t meshIndex = 0; meshIndex < hitCounts.size(); ++meshIndex) {
                            if (hitCounts[meshIndex] == 0) {
                                continue;
                            }
                            if (!hitLog.empty()) {
                                hitLog += ' ';
                            }
                            hitLog += "[" + std::to_string(meshIndex) + ":" + std::to_string(hitCounts[meshIndex]) + "]";
                        }
                        if (hitLog.empty()) {
                            hitLog = "<none>";
                        }
                        core::Logger::info("Renderer", "Hit trace counts: %s", hitLog.c_str());
                        hitDebugLogged = true;
                    } else {
                        const std::size_t sampleCount =
                            std::min<std::size_t>(totalPixels, static_cast<std::size_t>(32));
                        if (sampleCount > 0) {
                            std::string hitLog;
                            hitLog.reserve(sampleCount);
                            for (std::size_t i = 0; i < sampleCount; ++i) {
                                hitLog += hits[i] ? '1' : '0';
                            }
                            core::Logger::info("Renderer", "Hit debug sample: %s", hitLog.c_str());
                            hitDebugLogged = true;
                        }
                    }
                }
            }
        }

        stopMetalCaptureIfNeeded();
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

        const float aspect = (target.height > 0) ? static_cast<float>(target.width) / static_cast<float>(target.height)
                                                 : 1.0f;
        const float fovY = rtr::core::math::radians(kDefaultVerticalFovDegrees);
        const float halfHeight = tanf(fovY * 0.5f);
        const float halfWidth = halfHeight * aspect;

        const bool debugInstanceColors = std::getenv("RTR_DEBUG_INSTANCE_COLORS") != nullptr;
        const bool debugInstanceTrace = std::getenv("RTR_DEBUG_INSTANCE_TRACE") != nullptr;

        simd_float3 eye = cameraRig.eye;
        simd_float3 targetPoint = cameraRig.target;
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
        if (debugInstanceColors) {
            flags |= RTR_RAY_FLAG_INSTANCE_COLOR;
        }
        if (debugInstanceTrace) {
            flags |= RTR_RAY_FLAG_INSTANCE_TRACE;
        }
        uniforms->camera.flags = flags;
        uniforms->camera.samplesPerPixel = 1u;
        uniforms->camera.sampleSeed = frameCounter;

        uniforms->lightCount = 1u;
        uniforms->maxBounces = std::max<std::uint32_t>(1u, config.maxHardwareBounces);

        HardwareAreaLight& light = uniforms->lights[0];
        light.position = simd_make_float4(0.0f, 1.99f, 0.0f, 1.0f);
        light.right = simd_make_float4(0.25f, 0.0f, 0.0f, 0.0f);
        light.up = simd_make_float4(0.0f, 0.0f, 0.25f, 0.0f);
        light.forward = simd_make_float4(0.0f, -1.0f, 0.0f, 0.0f);
        light.color = simd_make_float4(18.0f, 17.5f, 17.0f, 0.0f);

        if (debugAlbedo) {
            core::Logger::info("Renderer", "Debug uniforms: flags=0x%x", uniforms->camera.flags);
        }

        if ([buffer storageMode] == MTLStorageModeManaged) {
            [buffer didModifyRange:NSMakeRange(0, sizeof(HardwareRayUniforms))];
        }
    }

    void startMetalCaptureIfNeeded(id<MTLCommandQueue> queue) {
        if (!metalCaptureEnabled || metalCaptureInProgress || metalCaptureCompleted || queue == nil) {
            return;
        }

        MTLCaptureManager* manager = [MTLCaptureManager sharedCaptureManager];
        if (!manager) {
            core::Logger::error("Renderer", "Metal capture manager unavailable");
            return;
        }
        if (manager.isCapturing) {
            metalCaptureInProgress = true;
            return;
        }

        if (!metalCaptureOutputPath.empty()) {
            std::error_code removeError;
            std::filesystem::remove(metalCaptureOutputPath, removeError);
        }

        MTLCaptureDescriptor* descriptor = [[MTLCaptureDescriptor alloc] init];
        descriptor.captureObject = queue;
        descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        if (!metalCaptureOutputPath.empty()) {
            NSString* capturePath = [NSString stringWithUTF8String:metalCaptureOutputPath.c_str()];
            if (capturePath.length > 0) {
                descriptor.outputURL = [NSURL fileURLWithPath:capturePath isDirectory:NO];
            }
        }

        NSError* error = nil;
        if (![manager startCaptureWithDescriptor:descriptor error:&error]) {
            const char* message = error ? error.localizedDescription.UTF8String : "unknown error";
            core::Logger::error("Renderer", "Failed to start Metal capture (%s)", message);
        } else {
            metalCaptureInProgress = true;
            core::Logger::info("Renderer", "Metal capture started (output=%s)", metalCaptureOutputPath.c_str());
        }
    }

    void stopMetalCaptureIfNeeded() {
        if (!metalCaptureInProgress) {
            return;
        }
        [[MTLCaptureManager sharedCaptureManager] stopCapture];
        metalCaptureInProgress = false;
        metalCaptureCompleted = true;
        core::Logger::info("Renderer", "Metal capture saved to %s", metalCaptureOutputPath.c_str());
    }

    void resetAccumulationInternal() {
        frameCounter = 0;
        accumulationFrameIndex = 0;
        accumulationInvalidated = true;
        resources.accumulationHistoryTexture = nil;
    }

    [[nodiscard]] std::uint32_t accumulationLimit() const {
        std::uint32_t limit = std::numeric_limits<std::uint32_t>::max();
        if (config.accumulationFrames > 0) {
            limit = std::min(limit, config.accumulationFrames);
        }
        return limit;
    }

    [[nodiscard]] std::uint32_t frameIndexForUniforms() const {
        return accumulationFrameIndex;
    }

    bool accumulationEnabledThisFrame() const {
        if (!config.accumulationEnabled) {
            return false;
        }
        const std::uint32_t limit = accumulationLimit();
        return accumulationFrameIndex < limit;
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

        auto ensureTexture = [&](id<MTLTexture> __strong* textureSlot,
                                 MTLPixelFormat format,
                                 NSString* label,
                                 bool clearOnCreate = false) {
            id<MTLTexture> texture = textureSlot ? *textureSlot : nil;
            if (texture != nil && resources.width == width && resources.height == height) {
                return true;
            }
            MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                           width:width
                                                                                          height:height
                                                                                       mipmapped:NO];
            desc.storageMode = clearOnCreate ? MTLStorageModeShared : MTLStorageModePrivate;
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            id<MTLTexture> newTexture = [device newTextureWithDescriptor:desc];
            if (!newTexture) {
                core::Logger::error("Renderer", "Failed to allocate %s (%ux%u)", label.UTF8String, width, height);
                return false;
            }
            [newTexture setLabel:label];
            if (textureSlot) {
                *textureSlot = newTexture;
            }

            if (clearOnCreate) {
                std::vector<float> zeros(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
                MTLRegion region = MTLRegionMake2D(0, 0, width, height);
                const NSUInteger bytesPerRow = static_cast<NSUInteger>(width) * sizeof(float) * 4u;
                [newTexture replaceRegion:region mipmapLevel:0 withBytes:zeros.data() bytesPerRow:bytesPerRow];
            }
            return true;
        };

        if (!ensureTexture(&resources.shadeTexture, MTLPixelFormatRGBA32Float, @"rtr.hw.lighting")) {
            success = false;
        }

        const bool needsAccumulation = accumulationEnabledThisFrame();
        if (needsAccumulation) {
            if (!ensureTexture(&resources.accumulationHistoryTexture,
                               MTLPixelFormatRGBA32Float,
                               @"rtr.hw.accum",
                               true)) {
                success = false;
            }
        } else if (resources.accumulationHistoryTexture != nil && (resources.width != width || resources.height != height)) {
            resources.accumulationHistoryTexture = nil;
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
            if ([resources.sceneLimitsBuffer storageMode] == MTLStorageModeManaged) {
                [resources.sceneLimitsBuffer didModifyRange:NSMakeRange(0, sizeof(MPSSceneLimits))];
            }
        }

        const std::size_t debugBufferLength = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * sizeof(std::uint32_t);
        if (debugBufferLength > 0) {
            if (!resources.hitDebugBuffer.isValid() || resources.hitDebugBuffer.length() < debugBufferLength) {
                resources.hitDebugBuffer = bufferAllocator.createBuffer(debugBufferLength, nullptr, "rtr.hw.hitDebug");
            }
            if (resources.hitDebugBuffer.isValid()) {
                id<MTLBuffer> hitBuffer = (__bridge id<MTLBuffer>)resources.hitDebugBuffer.nativeHandle();
                if (hitBuffer) {
                    std::memset([hitBuffer contents], 0, debugBufferLength);
                    if ([hitBuffer storageMode] == MTLStorageModeManaged) {
                        [hitBuffer didModifyRange:NSMakeRange(0, debugBufferLength)];
                    }
                }
            }
        } else {
            resources.hitDebugBuffer = {};
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
                    if ([buffer storageMode] == MTLStorageModeManaged) {
                        [buffer didModifyRange:NSMakeRange(0, length)];
                    }
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

        if (success) {
            resources.width = width;
            resources.height = height;
        }

        return success;
    }

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
    const bool debugSceneDump = std::getenv("RTR_DEBUG_SCENE_DUMP") != nullptr;
    const bool debugGeometryTrace = std::getenv("RTR_DEBUG_GEOMETRY_TRACE") != nullptr;

    std::vector<float> stagedPositions;
    std::vector<float> stagedNormals;
    std::vector<float> stagedColors;
    std::vector<vector_float2> stagedTexcoords;
    std::vector<std::uint32_t> stagedIndices;

    stagedPositions.reserve(sceneData.positions.size() * 3);
    stagedNormals.reserve(sceneData.normals.size() * 3);
    stagedColors.reserve(sceneData.colors.size() * 3);
    stagedTexcoords.reserve(sceneData.texcoords.size());
    stagedIndices.reserve(sceneData.indices.size());

    auto fetchVec3 = [](const std::vector<vector_float3>& buffer,
                        std::uint32_t index,
                        vector_float3 fallback) {
        if (index < buffer.size()) {
            return buffer[index];
        }
        return fallback;
    };

    auto fetchVec2 = [](const std::vector<vector_float2>& buffer,
                        std::uint32_t index,
                        vector_float2 fallback) {
        if (index < buffer.size()) {
            return buffer[index];
        }
        return fallback;
    };

    std::uint32_t currentVertexBase = 0u;
    std::uint32_t currentIndexBase = 0u;

    for (std::size_t meshIndex = 0; meshIndex < sceneData.meshRanges.size(); ++meshIndex) {
        const MPSMeshRange& meshRange = sceneData.meshRanges[meshIndex];
        const std::uint32_t vertexOffset = meshRange.vertexOffset;
        const std::uint32_t vertexCount = meshRange.vertexCount;
        const std::uint32_t indexOffset = meshRange.indexOffset;
        const std::uint32_t indexCount = meshRange.indexCount;
        if (vertexCount == 0 || indexCount == 0) {
            continue;
        }

        for (std::uint32_t i = 0; i < vertexCount; ++i) {
            const std::uint32_t sourceIndex = vertexOffset + i;
            const vector_float3 position =
                fetchVec3(sceneData.positions, sourceIndex, vector_float3{0.0f, 0.0f, 0.0f});
            stagedPositions.push_back(position.x);
            stagedPositions.push_back(position.y);
            stagedPositions.push_back(position.z);

            const vector_float3 normal = fetchVec3(sceneData.normals, sourceIndex, vector_float3{0.0f, 1.0f, 0.0f});
            stagedNormals.push_back(normal.x);
            stagedNormals.push_back(normal.y);
            stagedNormals.push_back(normal.z);

            const vector_float3 colour = fetchVec3(sceneData.colors, sourceIndex, vector_float3{1.0f, 1.0f, 1.0f});
            stagedColors.push_back(colour.x);
            stagedColors.push_back(colour.y);
            stagedColors.push_back(colour.z);

            const vector_float2 tex = fetchVec2(sceneData.texcoords, sourceIndex, vector_float2{0.0f, 0.0f});
            stagedTexcoords.push_back(tex);
        }

        for (std::uint32_t i = 0; i < indexCount; ++i) {
            const std::uint32_t sourceIndex = indexOffset + i;
            std::uint32_t globalIndex =
                (sourceIndex < sceneData.indices.size()) ? sceneData.indices[sourceIndex] : vertexOffset;
            if (globalIndex < vertexOffset) {
                globalIndex = vertexOffset;
            }
            const std::uint32_t localIndex = globalIndex - vertexOffset;
            stagedIndices.push_back(currentVertexBase + localIndex);
        }

        RayTracingMeshResource& meshResource = meshResources[meshIndex];
        meshResource.positionOffset = currentVertexBase;
        meshResource.normalOffset = currentVertexBase;
        meshResource.texcoordOffset = currentVertexBase;
        meshResource.colorOffset = currentVertexBase;
        meshResource.indexOffset = currentIndexBase;
        meshResource.vertexCount = vertexCount;
        meshResource.indexCount = indexCount;

        if (debugGeometryTrace && meshIndex >= 6) {
            const std::uint32_t sampleVertices = std::min<std::uint32_t>(vertexCount, 4u);
            for (std::uint32_t i = 0; i < sampleVertices; ++i) {
                const std::uint32_t base = meshResource.positionOffset + i;
                if (base * 3u + 2u < stagedPositions.size()) {
                    const float x = stagedPositions[base * 3u + 0u];
                    const float y = stagedPositions[base * 3u + 1u];
                    const float z = stagedPositions[base * 3u + 2u];
                    core::Logger::info("Renderer",
                                       "TRACE mesh[%zu] vertex[%u] = (%.3f, %.3f, %.3f)",
                                       meshIndex,
                                       i,
                                       x,
                                       y,
                                       z);
                }
            }

            const std::uint32_t sampleIndices = std::min<std::uint32_t>(indexCount, 6u);
            for (std::uint32_t i = 0; i < sampleIndices; ++i) {
                const std::uint32_t base = meshResource.indexOffset + i;
                if (base < stagedIndices.size()) {
                    core::Logger::info("Renderer",
                                       "TRACE mesh[%zu] index[%u] = %u",
                                       meshIndex,
                                       i,
                                       stagedIndices[base]);
                }
            }
        }

        currentVertexBase += vertexCount;
        currentIndexBase += indexCount;
    }

    for (auto& instance : instanceResources) {
        if (instance.meshIndex < meshResources.size()) {
            const RayTracingMeshResource& mesh = meshResources[instance.meshIndex];
            instance.primitiveOffset = mesh.indexOffset / 3u;
            instance.primitiveCount = mesh.indexCount / 3u;
        } else {
            instance.primitiveOffset = 0u;
            instance.primitiveCount = 0u;
        }
    }

    core::Logger::info("Renderer", "DEBUG: Preparing hardware scene data with %zu vertices.", sceneData.positions.size());
    for (int i = 0; i < 3 && i < sceneData.positions.size(); ++i) {
        const auto& v = sceneData.positions[i];
        core::Logger::info("Renderer", "DEBUG: Shader-Data Vtx[%d]: (%.3f, %.3f, %.3f)", i, v.x, v.y, v.z);
    }

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
                if ([buffer storageMode] == MTLStorageModeManaged) {
                    [buffer didModifyRange:NSMakeRange(0, byteLength)];
                }
            }
        }
    };

    uploadSceneBuffer(resources.positionsBuffer,
                      stagedPositions.empty() ? nullptr : stagedPositions.data(),
                      stagedPositions.size() * sizeof(float),
                      "rtr.hw.positions");
    uploadSceneBuffer(resources.normalsBuffer,
                      stagedNormals.empty() ? nullptr : stagedNormals.data(),
                      stagedNormals.size() * sizeof(float),
                      "rtr.hw.normals");
    uploadSceneBuffer(resources.colorsBuffer,
                      stagedColors.empty() ? nullptr : stagedColors.data(),
                      stagedColors.size() * sizeof(float),
                      "rtr.hw.colors");
    uploadSceneBuffer(resources.texcoordBuffer,
                      stagedTexcoords.empty() ? nullptr : stagedTexcoords.data(),
                      stagedTexcoords.size() * sizeof(vector_float2),
                      "rtr.hw.texcoords");
    uploadSceneBuffer(resources.indicesBuffer,
                      stagedIndices.empty() ? nullptr : stagedIndices.data(),
                      stagedIndices.size() * sizeof(std::uint32_t),
                      "rtr.hw.indices");
    resources.sceneLimits.vertexCount = static_cast<std::uint32_t>(stagedPositions.size() / 3u);
    resources.sceneLimits.indexCount = static_cast<std::uint32_t>(stagedIndices.size());
    resources.sceneLimits.colorCount = static_cast<std::uint32_t>(stagedColors.size() / 3u);
    resources.sceneLimits.primitiveCount = static_cast<std::uint32_t>(stagedIndices.size() / 3u);
    resources.sceneLimits.normalCount = static_cast<std::uint32_t>(stagedNormals.size() / 3u);
    resources.sceneLimits.texcoordCount = static_cast<std::uint32_t>(stagedTexcoords.size());
    resources.sceneLimits.materialCount = static_cast<std::uint32_t>(materialResources.size());
    resources.sceneLimits.textureCount = static_cast<std::uint32_t>(textureResources.size());
    resources.sceneLimits.instanceCount = static_cast<std::uint32_t>(instanceResources.size());
    resources.sceneLimits.meshCount = static_cast<std::uint32_t>(meshResources.size());

    uploadSceneBuffer(resources.meshResourceBuffer,
                      meshResources.empty() ? nullptr : meshResources.data(),
                      meshResources.size() * sizeof(RayTracingMeshResource),
                      "rtr.meshResources");

    for (std::size_t i = 0; i < meshResources.size() && i < 9; ++i) {
        const auto& meshResource = meshResources[i];
        core::Logger::info("Renderer",
                           "MeshResource[%zu]: vertexBase=%u vertexCount=%u indexBase=%u indexCount=%u material=%u",
                           i,
                           meshResource.positionOffset,
                           meshResource.vertexCount,
                           meshResource.indexOffset,
                           meshResource.indexCount,
                           meshResource.materialIndex);
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

    const std::size_t meshCount = scene.meshes().size();
    const std::size_t materialCount = scene.materials().size();
    const std::size_t instanceCount = scene.instances().size();

    core::Logger::info("Renderer",
                       "Loading scene with %zu meshes, %zu materials, %zu instances",
                       meshCount,
                       materialCount,
                       instanceCount);

    const bool debugSceneDump = std::getenv("RTR_DEBUG_SCENE_DUMP") != nullptr;
    if (debugSceneDump) {
        core::Logger::info("Renderer", "RTR_DEBUG_SCENE_DUMP enabled; dumping mesh/instance metadata");
    }

    sceneBounds = scene.computeSceneBounds();
    core::Logger::info("Renderer",
                       "Scene bounds min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f)",
                       sceneBounds.min.x,
                       sceneBounds.min.y,
                       sceneBounds.min.z,
                       sceneBounds.max.x,
                       sceneBounds.max.y,
                       sceneBounds.max.z);
    updateCameraRigFromBounds();

    const MPSSceneData sceneData = buildSceneData(scene);
    if (sceneData.positions.empty() || sceneData.indices.empty() || sceneData.meshRanges.empty() ||
        sceneData.instanceRanges.empty()) {
        core::Logger::warn("Renderer", "Flattened scene data empty; scene load aborted");
        return false;
    }

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

    bottomLevelStructures.reserve(sceneData.meshRanges.size());

    std::vector<InstanceBuildInput> instanceInputs;
    instanceInputs.reserve(sceneData.instanceRanges.size());
    instanceResources.clear();
    instanceResources.reserve(sceneData.instanceRanges.size());
    meshResources.clear();
    meshResources.resize(sceneData.meshRanges.size());

    std::vector<std::size_t> meshBlasLookup(sceneData.meshRanges.size(), std::numeric_limits<std::size_t>::max());

    for (std::size_t meshIndex = 0; meshIndex < sceneData.meshRanges.size(); ++meshIndex) {
        const MPSMeshRange& meshRange = sceneData.meshRanges[meshIndex];
        scene::Mesh mesh = makeMeshFromRange(sceneData, meshRange);
        if (mesh.vertices().empty() || mesh.indices().empty()) {
            core::Logger::warn("Renderer", "Skipping mesh range %zu due to empty geometry", meshIndex);
            continue;
        }

        RayTracingMeshResource& meshResource = meshResources[meshIndex];
        meshResource.positionOffset = meshRange.vertexOffset;
        meshResource.normalOffset = meshRange.vertexOffset;
        meshResource.texcoordOffset = meshRange.vertexOffset;
        meshResource.colorOffset = meshRange.vertexOffset;
        meshResource.indexOffset = meshRange.indexOffset;
        meshResource.vertexCount = meshRange.vertexCount;
        meshResource.indexCount = meshRange.indexCount;
        meshResource.materialIndex = meshRange.materialIndex;

        if (debugSceneDump) {
            core::Logger::info("Renderer",
                               "MeshRange[%zu]: verts=%u indices=%u material=%u",
                               meshIndex,
                               meshRange.vertexCount,
                               meshRange.indexCount,
                               meshRange.materialIndex);
        }

        const std::string meshLabel = "scene_mesh_" + std::to_string(meshIndex);
        const auto uploadIndex = geometryStore.uploadMesh(mesh, meshLabel);
        if (!uploadIndex.has_value()) {
            core::Logger::warn("Renderer", "Failed to upload mesh %zu", meshIndex);
            continue;
        }

        const auto& meshBuffers = geometryStore.uploadedMeshes()[*uploadIndex];
        auto blas = asBuilder.buildBottomLevel(meshBuffers, meshLabel, queueHandle);
        if (!blas.has_value()) {
            core::Logger::warn("Renderer", "Failed to build BLAS for mesh %zu", meshIndex);
            continue;
        }

        const std::size_t blasIndex = bottomLevelStructures.size();
        bottomLevelStructures.push_back(std::move(*blas));
        meshBlasLookup[meshIndex] = blasIndex;
    }

    auto isMatrixFinite = [](const simd_float4x4& matrix) {
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                if (!std::isfinite(matrix.columns[column][row])) {
                    return false;
                }
            }
        }
        return true;
    };

    for (std::size_t instanceIndex = 0; instanceIndex < sceneData.instanceRanges.size(); ++instanceIndex) {
        const MPSInstanceRange& range = sceneData.instanceRanges[instanceIndex];
        if (range.meshIndex >= meshBlasLookup.size()) {
            core::Logger::warn("Renderer", "Instance %zu references invalid mesh index %u",
                                instanceIndex,
                                range.meshIndex);
            continue;
        }
        const std::size_t blasIndex = meshBlasLookup[range.meshIndex];
        if (blasIndex == std::numeric_limits<std::size_t>::max()) {
            core::Logger::warn("Renderer", "Mesh %u missing BLAS; instance %zu skipped",
                                range.meshIndex,
                                instanceIndex);
            continue;
        }

        simd_float4x4 objectToWorld = range.transform;
        if (!isMatrixFinite(objectToWorld)) {
            objectToWorld = matrix_identity_float4x4;
        }
        simd_float4x4 worldToObject = range.inverseTransform;
        if (!isMatrixFinite(worldToObject)) {
            worldToObject = simd_inverse(objectToWorld);
        }
        if (!isMatrixFinite(worldToObject)) {
            worldToObject = matrix_identity_float4x4;
        }

        if (debugSceneDump) {
            const simd_float4& translation = objectToWorld.columns[3];
            core::Logger::info("Renderer",
                               "Instance[%zu]: meshRange=%u material=%u translation=(%.3f, %.3f, %.3f)",
                               instanceIndex,
                               range.meshIndex,
                               range.materialIndex,
                               translation.x,
                               translation.y,
                               translation.z);
        }

        const std::uint32_t tlasInstanceIndex = static_cast<std::uint32_t>(instanceResources.size());

        InstanceBuildInput input{};
        input.structure = &bottomLevelStructures[blasIndex];
        input.transform = objectToWorld;
        input.userID = tlasInstanceIndex;
        input.mask = RTR_TRIANGLE_MASK_GEOMETRY;
        input.intersectionFunctionTableOffset = 0u;
        instanceInputs.push_back(input);

        RayTracingInstanceResource resource{};
        resource.objectToWorld = objectToWorld;
        resource.worldToObject = worldToObject;
        resource.meshIndex = static_cast<std::uint32_t>(range.meshIndex);
        const std::uint32_t safeMaterialIndex = (range.materialIndex < materials.size()) ? range.materialIndex : 0u;
        resource.materialIndex = safeMaterialIndex;
        instanceResources.push_back(resource);
    }

    if (instanceInputs.empty()) {
        core::Logger::error("Renderer", "No valid instances were prepared for scene");
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

    core::Logger::info("Renderer",
                       "Captured %zu material resources (logging first %zu)",
                       materialResources.size(),
                       std::min<std::size_t>(materialResources.size(), 8));
    for (std::size_t i = 0; i < materialResources.size() && i < 8; ++i) {
        const RayTracingMaterialResource& mat = materialResources[i];
        core::Logger::info("Renderer",
                           "Material[%zu]: albedo=(%.3f, %.3f, %.3f) emission=(%.3f, %.3f, %.3f) rough=%.3f metal=%.3f refl=%.3f ior=%.3f tex=%u flags=0x%X",
                           i,
                           mat.albedo.x,
                           mat.albedo.y,
                           mat.albedo.z,
                           mat.emission.x,
                           mat.emission.y,
                           mat.emission.z,
                           mat.roughness,
                           mat.metallic,
                           mat.reflectivity,
                           mat.indexOfRefraction,
                           mat.textureIndex,
                           mat.materialFlags);
    }

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
    updateCameraRigFromBounds();
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
