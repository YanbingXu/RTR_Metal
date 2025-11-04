#pragma once

#include <memory>
#include <string>

#include "RTRMetalEngine/Core/EngineConfig.hpp"

namespace rtr::rendering {

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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtr::rendering
