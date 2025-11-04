#import <Metal/Metal.h>
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif
#if __has_include(<Metal/MetalRayTracing.h>)
#import <Metal/MetalRayTracing.h>
#define RTR_RENDERER_HAS_RAYTRACING_HEADERS 1
#else
#define RTR_RENDERER_HAS_RAYTRACING_HEADERS 0
#endif

#include "RTRMetalEngine/Rendering/Renderer.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/AccelerationStructure.hpp"
#include "RTRMetalEngine/Rendering/AccelerationStructureBuilder.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"
#include "RTRMetalEngine/Rendering/RayTracingPipeline.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
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

struct ShaderBindingTable {
    id<MTLBuffer> raygenRecord = nil;
    id<MTLBuffer> missRecord = nil;
    id<MTLBuffer> hitGroupRecord = nil;
    std::size_t raygenStride = 0;
    std::size_t missStride = 0;
    std::size_t hitGroupStride = 0;

    void reset() {
        raygenRecord = nil;
        missRecord = nil;
        hitGroupRecord = nil;
        raygenStride = 0;
        missStride = 0;
        hitGroupStride = 0;
    }

    [[nodiscard]] bool isValid() const noexcept {
        return raygenRecord != nil && missRecord != nil && hitGroupRecord != nil;
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
        sbt.reset();
    }

    void renderFrame() {
        if (!context.isValid()) {
            core::Logger::warn("Renderer", "Skipping frame: Metal context invalid");
            return;
        }
        if (isRayTracingReady()) {
            core::Logger::info("Renderer",
                               "Ray tracing target ready (%ux%u RGBA32F)",
                               target.width,
                               target.height);
        } else {
            core::Logger::info("Renderer", "Ray tracing not ready; falling back to stub output");
        }
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
    ShaderBindingTable sbt;

    [[nodiscard]] bool ensureRayTracingTarget(std::uint32_t width, std::uint32_t height) {
        if (!context.isValid() || !rayTracingPipeline.isValid()) {
            return false;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
        if (!device) {
            core::Logger::error("Renderer", "Unable to acquire Metal device for ray tracing target");
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

    [[nodiscard]] bool ensureShaderBindingTable() {
        if (!context.isValid() || !rayTracingPipeline.isValid()) {
            return false;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context.rawDeviceHandle();
        if (!device) {
            core::Logger::error("Renderer", "Unable to acquire Metal device for shader binding table");
            sbt.reset();
            return false;
        }

#if RTR_RENDERER_HAS_RAYTRACING_HEADERS
        id<MTLRayTracingPipelineState> pipelineState =
            (__bridge id<MTLRayTracingPipelineState>)rayTracingPipeline.rawPipelineState();
        if (!pipelineState) {
            core::Logger::warn("Renderer", "Ray tracing pipeline unavailable; SBT allocation skipped");
            sbt.reset();
            return false;
        }
        (void)pipelineState;
#else
        (void)rayTracingPipeline;
        core::Logger::warn("Renderer", "Ray tracing headers unavailable; SBT allocation skipped");
        sbt.reset();
        return false;
#endif

        if (sbt.isValid()) {
            return true;
        }

        sbt.reset();

        const std::size_t recordSize = 256U;  // Placeholder record stride until binding layout is defined.
        auto allocateBuffer = [&](const char* usageLabel) -> id<MTLBuffer> {
            id<MTLBuffer> buffer =
                [device newBufferWithLength:recordSize options:MTLResourceStorageModeShared];
            if (!buffer) {
                core::Logger::error("Renderer", "Failed to allocate %s shader binding table buffer (%zu bytes)",
                                    usageLabel, recordSize);
            }
            return buffer;
        };

        id<MTLBuffer> raygen = allocateBuffer("ray generation");
        id<MTLBuffer> miss = allocateBuffer("miss");
        id<MTLBuffer> hit = allocateBuffer("hit group");

        if (!raygen || !miss || !hit) {
            sbt.reset();
            return false;
        }

        sbt.raygenRecord = raygen;
        sbt.missRecord = miss;
        sbt.hitGroupRecord = hit;
        sbt.raygenStride = recordSize;
        sbt.missStride = recordSize;
        sbt.hitGroupStride = recordSize;

        core::Logger::info("Renderer", "Allocated shader binding table buffers (%zu bytes each)", recordSize);
        return true;
    }

    [[nodiscard]] bool isRayTracingReady() const noexcept {
        return context.isValid() && rayTracingPipeline.isValid() && topLevelStructure.isValid() && target.isValid() &&
               sbt.isValid();
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
                const bool targetReady = ensureRayTracingTarget(kDiagnosticWidth, kDiagnosticHeight);
                const bool sbtReady = ensureShaderBindingTable();
                if (!targetReady) {
                    core::Logger::warn("Renderer", "Failed to prepare ray tracing target after TLAS build");
                }
                if (!sbtReady) {
                    core::Logger::warn("Renderer", "Failed to prepare shader binding table after TLAS build");
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
