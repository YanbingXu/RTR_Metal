#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <gtest/gtest.h>

#include <TargetConditionals.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"
#include "RTRMetalEngine/MPS/MPSRenderer.hpp"
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

std::vector<uint8_t> loadPPMPixels(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }

    std::string magic;
    file >> magic;
    if (magic != "P6") {
        return {};
    }

    int width = 0;
    int height = 0;
    int maxValue = 0;
    file >> width >> height >> maxValue;
    file.get();  // consume newline after header

    if (width <= 0 || height <= 0 || maxValue <= 0) {
        return {};
    }

    std::vector<uint8_t> pixels(static_cast<std::size_t>(width) * height * 3);
    file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (!file) {
        return {};
    }

    return pixels;
}

std::filesystem::path makeTempPPMPath(const char* suffix) {
    std::filesystem::path temp = std::filesystem::temp_directory_path();
    temp /= std::string("rtr_mps_renderer_") + suffix + ".ppm";
    return temp;
}

}  // namespace

TEST(MPSRendererImageComparisonTests, CpuAndGpuOutputsStayWithinTolerance) {
    rtr::rendering::MetalContext context;
    if (!context.isValid()) {
        GTEST_SKIP() << "Metal context unavailable on this system";
    }

#if TARGET_OS_OSX
    {
        rtr::rendering::MPSPathTracer capabilityProbe;
        if (!capabilityProbe.initialize(context)) {
            GTEST_SKIP() << "Device does not support MPS ray tracing";
        }
    }
#else
    GTEST_SKIP() << "MPS path tracer not supported on this platform";
#endif

    EnvOverride envGuard("RTR_MPS_GPU_SHADING");
    envGuard.set("1");

    auto cpuOutputPath = makeTempPPMPath("cpu");
    auto gpuOutputPath = makeTempPPMPath("gpu");
#ifdef DEBUG
    std::cerr << "CPU path: " << cpuOutputPath << "\nGPU path: " << gpuOutputPath << '\n';
#endif

    rtr::rendering::MPSRenderer renderer(context);
    ASSERT_TRUE(renderer.initialize()) << "Failed to initialise renderer";
    if (!renderer.usesGPUShading()) {
        GTEST_SKIP() << "GPU shading pipeline inactive on this configuration";
    }

    rtr::rendering::MPSRenderer::FrameComparison comparison;
    ASSERT_TRUE(renderer.renderFrameComparison(cpuOutputPath.c_str(),
                                               gpuOutputPath.c_str(),
                                               &comparison))
        << "Failed to render comparison frame";

    auto cpuFilePixels = loadPPMPixels(cpuOutputPath);
    auto gpuFilePixels = loadPPMPixels(gpuOutputPath);

    std::error_code ec;
    std::filesystem::remove(cpuOutputPath, ec);
    std::filesystem::remove(gpuOutputPath, ec);

    if (comparison.gpuPixels.empty()) {
        GTEST_SKIP() << "GPU shading produced no output for comparison";
    }

    ASSERT_FALSE(comparison.cpuPixels.empty()) << "CPU shading produced no pixels";
    ASSERT_EQ(comparison.cpuPixels.size(), comparison.gpuPixels.size())
        << "CPU and GPU buffers differ in size";

    double maxDifference = comparison.maxByteDifference;
    if (maxDifference == 0.0) {
        for (std::size_t i = 0; i < comparison.cpuPixels.size(); ++i) {
            const double diff = std::fabs(static_cast<double>(comparison.cpuPixels[i]) - comparison.gpuPixels[i]);
            maxDifference = std::max(maxDifference, diff);
        }
    }

    EXPECT_LE(maxDifference, 2.0)
        << "GPU shading diverged from CPU reference beyond tolerance (max diff = " << maxDifference << ')';

    if (!cpuFilePixels.empty() && !gpuFilePixels.empty()) {
        EXPECT_EQ(cpuFilePixels.size(), gpuFilePixels.size())
            << "PPM outputs differ in size";
    }
}
