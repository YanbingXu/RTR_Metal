#include "RTRMetalEngine/Rendering/Renderer.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

#include <iostream>
#include <utility>

namespace rtr::rendering {

struct Renderer::Impl {
    explicit Impl(core::EngineConfig cfg)
        : config(std::move(cfg)), context() {
        if (!context.isValid()) {
            core::Logger::error("Renderer", "Metal context initialization failed");
        } else {
            core::Logger::info("Renderer", "Renderer configured for %s", config.applicationName.c_str());
            context.logDeviceInfo();
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
};

Renderer::Renderer(core::EngineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Renderer::~Renderer() = default;

Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;

const core::EngineConfig& Renderer::config() const noexcept { return impl_->config; }

void Renderer::renderFrame() { impl_->renderFrame(); }

}  // namespace rtr::rendering
