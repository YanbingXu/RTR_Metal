#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "RTRMetalEngine/Scene/Scene.hpp"

#include "RTRMetalEngine/Core/EngineConfig.hpp"

namespace rtr::rendering {

enum class DebugVisualization {
    None,
    Albedo,
    InstanceColors,
    InstanceTrace,
    PrimitiveTrace,
};

struct RendererDebugOptions {
    DebugVisualization visualization = DebugVisualization::None;
    bool sceneDump = false;
    bool geometryTrace = false;
    bool tlasTrace = false;
    bool cameraTrace = false;
    bool isolateCornellExtras = false;
    std::optional<std::uint32_t> isolateCornellMeshIndex;
};

class Renderer {
public:
    explicit Renderer(core::EngineConfig config);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;

    [[nodiscard]] const core::EngineConfig& config() const noexcept;
    [[nodiscard]] bool isRayTracingReady() const noexcept;
    void renderFrame();
    [[nodiscard]] bool renderFrameInteractive();
    void setOutputPath(std::string path);
    void setRenderSize(std::uint32_t width, std::uint32_t height);
    bool loadScene(const scene::Scene& scene);
    void setDebugMode(bool enabled);
    void setDebugVisualization(DebugVisualization visualization);
    void setDebugOptions(const RendererDebugOptions& options);
    [[nodiscard]] RendererDebugOptions debugOptions() const;
    void setShadingMode(const std::string& mode);
    void resetAccumulation();

    [[nodiscard]] void* deviceHandle() const noexcept;
    [[nodiscard]] void* commandQueueHandle() const noexcept;
    [[nodiscard]] void* currentColorTexture() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtr::rendering
