#include "RTRMetalEngine/Rendering/Renderer.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/AccelerationStructure.hpp"
#include "RTRMetalEngine/Rendering/AccelerationStructureBuilder.hpp"
#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

#include <array>
#include <iostream>
#include <utility>
#include <vector>

#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

namespace rtr::rendering {

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
                buildDiagnosticAccelerationStructure();
            }
        }
    }

    void renderFrame() {
        if (!context.isValid()) {
            core::Logger::warn("Renderer", "Skipping frame: Metal context invalid");
            return;
        }
        std::cout << "Renderer frame stub executed using " << context.deviceName() << std::endl;
    }

    core::EngineConfig config;
    MetalContext context;
    BufferAllocator bufferAllocator;
    GeometryStore geometryStore;
    AccelerationStructureBuilder asBuilder;
    std::vector<AccelerationStructure> bottomLevelStructures;

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

void Renderer::renderFrame() { impl_->renderFrame(); }

}  // namespace rtr::rendering
