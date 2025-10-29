#include <gtest/gtest.h>

#include "RTRMetalEngine/MPS/MPSRenderer.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

TEST(MPSRendererDeterminismTests, CpuShadingProducesStablePixelsAcrossRuns) {
    rtr::rendering::MetalContext context;
    if (!context.isValid()) {
        GTEST_SKIP() << "Metal context unavailable";
    }

    rtr::rendering::MPSRenderer renderer(context);
    ASSERT_TRUE(renderer.initialize()) << "Failed to initialise MPSRenderer";

    renderer.setShadingMode(rtr::rendering::MPSRenderer::ShadingMode::CpuOnly);

    rtr::rendering::MPSRenderer::FrameComparison first;
    ASSERT_TRUE(renderer.renderFrameComparison(nullptr, nullptr, &first))
        << "First CPU render failed";

    ASSERT_FALSE(first.cpuPixels.empty()) << "CPU render produced no pixels";
    ASSERT_NE(first.cpuPixelHash, 0u) << "CPU pixel hash should not be zero";

    rtr::rendering::MPSRenderer::FrameComparison second;
    ASSERT_TRUE(renderer.renderFrameComparison(nullptr, nullptr, &second))
        << "Second CPU render failed";

    ASSERT_EQ(first.cpuPixels.size(), second.cpuPixels.size())
        << "CPU renders produced buffers of different sizes";

    EXPECT_EQ(first.cpuPixels, second.cpuPixels)
        << "CPU renders differed between runs";
    EXPECT_EQ(first.cpuPixelHash, second.cpuPixelHash)
        << "CPU pixel hash differed between runs";
}
