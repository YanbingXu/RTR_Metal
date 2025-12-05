#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

struct AccumulationResult {
    std::vector<float> runningSum;
    std::vector<float> averaged;
};

AccumulationResult simulateAccumulation(const std::vector<float>& previousSum,
                                        const std::vector<float>& newSample,
                                        std::uint32_t frameIndex) {
    EXPECT_TRUE(previousSum.empty() || previousSum.size() == newSample.size());
    std::vector<float> sum = newSample;
    if (!previousSum.empty()) {
        for (std::size_t i = 0; i < sum.size(); ++i) {
            sum[i] += previousSum[i];
        }
    }

    const float divisor = static_cast<float>(frameIndex + 1u);
    std::vector<float> averaged(sum.size());
    for (std::size_t i = 0; i < sum.size(); ++i) {
        averaged[i] = sum[i] / divisor;
    }

    return {sum, averaged};
}

constexpr float kTolerance = 1.0e-6f;

}  // namespace

TEST(RendererAccumulation, FirstFrameKeepsExactSample) {
    const std::vector<float> sample = {0.25f, 0.5f, 1.0f};
    const auto result = simulateAccumulation({}, sample, 0u);
    ASSERT_EQ(result.runningSum.size(), sample.size());
    for (std::size_t i = 0; i < sample.size(); ++i) {
        EXPECT_NEAR(result.runningSum[i], sample[i], kTolerance);
        EXPECT_NEAR(result.averaged[i], sample[i], kTolerance);
    }
}

TEST(RendererAccumulation, RunningAverageMatchesReference) {
    const std::vector<float> first = {1.0f, 0.0f, 0.0f};
    const auto firstFrame = simulateAccumulation({}, first, 0u);
    const std::vector<float> second = {0.0f, 1.0f, 0.0f};
    const auto secondFrame = simulateAccumulation(firstFrame.runningSum, second, 1u);

    ASSERT_EQ(secondFrame.runningSum.size(), first.size());
    EXPECT_NEAR(secondFrame.runningSum[0], 1.0f, kTolerance);
    EXPECT_NEAR(secondFrame.runningSum[1], 1.0f, kTolerance);
    EXPECT_NEAR(secondFrame.averaged[0], 0.5f, kTolerance);
    EXPECT_NEAR(secondFrame.averaged[1], 0.5f, kTolerance);
    EXPECT_NEAR(secondFrame.averaged[2], 0.0f, kTolerance);
}

TEST(RendererAccumulation, EnergyDoesNotDecayOverFrames) {
    std::vector<float> sum;
    std::vector<float> averaged;
    for (std::uint32_t frame = 0; frame < 32; ++frame) {
        const std::vector<float> sample = {2.0f, 2.0f, 2.0f};
        const auto result = simulateAccumulation(sum, sample, frame);
        sum = result.runningSum;
        averaged = result.averaged;
    }

    ASSERT_EQ(averaged.size(), 3u);
    EXPECT_NEAR(averaged[0], 2.0f, kTolerance);
    EXPECT_NEAR(averaged[1], 2.0f, kTolerance);
    EXPECT_NEAR(averaged[2], 2.0f, kTolerance);
}
