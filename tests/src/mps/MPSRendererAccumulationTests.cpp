#include <gtest/gtest.h>

#include <TargetConditionals.h>

#include "RTRMetalEngine/MPS/MPSRenderer.hpp"
#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

TEST(MPSRendererAccumulationTests, TracksAccumulatedFrames) {
    rtr::rendering::MetalContext context;
    if (!context.isValid()) {
        GTEST_SKIP() << "Metal context unavailable";
    }

#if TARGET_OS_OSX
    {
        rtr::rendering::MPSPathTracer probe;
        if (!probe.initialize(context)) {
            GTEST_SKIP() << "Device does not support MPS ray tracing";
        }
    }
#else
    GTEST_SKIP() << "MPS path tracer not supported on this platform";
#endif

    rtr::rendering::MPSRenderer renderer(context);
    renderer.setShadingMode(rtr::rendering::MPSRenderer::ShadingMode::GpuPreferred);
    renderer.setSamplingParameters(2, 0);
    renderer.setAccumulationParameters(true, 2);
    ASSERT_TRUE(renderer.initialize()) << "Failed to initialise renderer";

    if (!renderer.usesGPUShading()) {
        GTEST_SKIP() << "GPU shading unavailable on this configuration";
    }

    renderer.resetAccumulation();
    EXPECT_EQ(renderer.accumulatedFrames(), 0u);

    ASSERT_TRUE(renderer.renderFrame(nullptr)) << "First accumulated frame failed";
    EXPECT_EQ(renderer.accumulatedFrames(), 1u);

    ASSERT_TRUE(renderer.renderFrame(nullptr)) << "Second accumulated frame failed";
    EXPECT_EQ(renderer.accumulatedFrames(), 2u);

    ASSERT_TRUE(renderer.renderFrame(nullptr)) << "Frame with accumulation cap failed";
    EXPECT_EQ(renderer.accumulatedFrames(), 2u) << "Accumulation exceeded configured frame count";

    renderer.setAccumulationParameters(false, 0);
    EXPECT_EQ(renderer.accumulatedFrames(), 0u);
}
