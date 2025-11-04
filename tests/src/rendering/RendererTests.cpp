#include <gtest/gtest.h>

#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Rendering/Renderer.hpp"

TEST(RendererTests, RayTracingNotReadyWithoutValidPipeline) {
    rtr::core::EngineConfig config;
    config.applicationName = "RendererTests";
    config.shaderLibraryPath = "/invalid/path/RTRShaders.metallib";

    rtr::rendering::Renderer renderer(config);
    EXPECT_FALSE(renderer.isRayTracingReady());
}

TEST(RendererTests, RenderFrameExecutesWithoutCrash) {
    rtr::core::EngineConfig config;
    config.applicationName = "RendererTests";
    config.shaderLibraryPath = "/invalid/path/RTRShaders.metallib";

    rtr::rendering::Renderer renderer(config);
    renderer.renderFrame();
    SUCCEED();
}
