#include <gtest/gtest.h>
#include <TargetConditionals.h>

#include "RTRMetalEngine/MPS/MPSRenderer.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

TEST(MPSRendererGpuHashTests, GpuShadingStableWithinRun) {
    rtr::rendering::MetalContext context;
    if (!context.isValid()) {
        GTEST_SKIP() << "Metal context unavailable";
    }

#if TARGET_OS_OSX
    rtr::rendering::MPSRenderer renderer(context);
    ASSERT_TRUE(renderer.initialize()) << "Failed to initialise MPSRenderer";

    if (!renderer.usesGPUShading()) {
        GTEST_SKIP() << "GPU shading unavailable on this configuration";
    }

    renderer.setShadingMode(rtr::rendering::MPSRenderer::ShadingMode::GpuPreferred);

    rtr::rendering::MPSRenderer::FrameComparison first;
    ASSERT_TRUE(renderer.renderFrameComparison(nullptr, nullptr, &first))
        << "First GPU render failed";
    ASSERT_FALSE(first.gpuPixels.empty()) << "GPU render produced no pixels";

    rtr::rendering::MPSRenderer::FrameComparison second;
    ASSERT_TRUE(renderer.renderFrameComparison(nullptr, nullptr, &second))
        << "Second GPU render failed";

    ASSERT_EQ(first.gpuPixels.size(), second.gpuPixels.size());
    EXPECT_EQ(first.gpuPixelHash, second.gpuPixelHash)
        << "GPU pixel hash differed between consecutive renders";
#else
    GTEST_SKIP() << "MPS GPU path requires macOS";
#endif
}
