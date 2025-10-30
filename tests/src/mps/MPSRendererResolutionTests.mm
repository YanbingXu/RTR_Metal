#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <gtest/gtest.h>

#include <TargetConditionals.h>

#include <cstdint>
#include <cstdlib>
#include <optional>

#include "RTRMetalEngine/MPS/MPSRenderer.hpp"
#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

namespace {

class EnvOverride {
public:
    explicit EnvOverride(std::string name)
        : name_(std::move(name)) {
        const char* current = std::getenv(name_.c_str());
        if (current) {
            original_ = std::string(current);
        }
    }

    EnvOverride(const EnvOverride&) = delete;
    EnvOverride& operator=(const EnvOverride&) = delete;

    ~EnvOverride() {
        if (original_.has_value()) {
            setenv(name_.c_str(), original_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    void set(std::string_view value) {
        setenv(name_.c_str(), std::string(value).c_str(), 1);
    }

private:
    std::string name_;
    std::optional<std::string> original_;
};

}  // namespace

TEST(MPSRendererResolutionTests, CpuOutputMatchesCustomResolution) {
    rtr::rendering::MetalContext context;
    if (!context.isValid()) {
        GTEST_SKIP() << "Metal context unavailable on this system";
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

    EnvOverride envGuard("RTR_MPS_GPU_SHADING");
    envGuard.set("0");

    constexpr std::uint32_t width = 320;
    constexpr std::uint32_t height = 180;

    rtr::rendering::MPSRenderer renderer(context);
    renderer.setFrameDimensions(width, height);
    renderer.setShadingMode(rtr::rendering::MPSRenderer::ShadingMode::CpuOnly);
    ASSERT_TRUE(renderer.initialize()) << "Failed to initialise MPSRenderer";

    rtr::rendering::MPSRenderer::FrameComparison comparison;
    ASSERT_TRUE(renderer.renderFrameComparison(nullptr, nullptr, &comparison))
        << "Failed to render comparison frame";

    ASSERT_FALSE(comparison.cpuPixels.empty()) << "CPU shading produced no data";
    EXPECT_EQ(comparison.width, width);
    EXPECT_EQ(comparison.height, height);
    EXPECT_EQ(comparison.cpuPixels.size(), static_cast<std::size_t>(width) * height * 3)
        << "CPU pixel buffer size mismatch";
}
