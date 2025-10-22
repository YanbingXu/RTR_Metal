#include <gtest/gtest.h>

#include "RTRMetalEngine/Core/EngineConfig.hpp"

TEST(EngineConfigTests, StoresPassedValues) {
    const rtr::core::EngineConfig config{
        .applicationName = "UnitTestApp",
        .shaderLibraryPath = "shaders/Test.metallib",
    };

    EXPECT_EQ(config.applicationName, "UnitTestApp");
    EXPECT_EQ(config.shaderLibraryPath, "shaders/Test.metallib");
}
