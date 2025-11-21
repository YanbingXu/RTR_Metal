#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "RTRMetalEngine/Core/ConfigLoader.hpp"
#include "RTRMetalEngine/Core/FileSystem.hpp"

namespace {

std::filesystem::path makeTempFile(const std::string& name, const std::string& contents) {
    const auto tempDir = std::filesystem::temp_directory_path();
    const auto filePath = tempDir / name;
    std::ofstream stream(filePath);
    stream << contents;
    stream.close();
    return filePath;
}

}  // namespace

TEST(FileSystem, ReadsBackWrittenFile) {
    const auto path = makeTempFile("rtr_config_test.ini", "applicationName = Test\n");
    ASSERT_TRUE(rtr::core::FileSystem::exists(path));
    const std::string contents = rtr::core::FileSystem::readTextFile(path);
    EXPECT_EQ(contents, "applicationName = Test\n");
    std::filesystem::remove(path);
}

TEST(ConfigLoader, ParsesRequiredFields) {
    const auto path = makeTempFile("rtr_engine_config.ini",
                                   R"(applicationName = Sample
shaderLibraryPath = assets/RTR.metallib
)");

    const rtr::core::EngineConfig config = rtr::core::ConfigLoader::loadEngineConfig(path);
    EXPECT_EQ(config.applicationName, "Sample");
    EXPECT_EQ(config.shaderLibraryPath, "assets/RTR.metallib");
    EXPECT_EQ(config.shadingMode, "auto");

    std::filesystem::remove(path);
}

TEST(ConfigLoader, AppliesDefaultShaderLibraryPath) {
    const auto path = makeTempFile("rtr_engine_config_defaults.ini",
                                   R"(applicationName=NoShaderPath
)");

    const rtr::core::EngineConfig config = rtr::core::ConfigLoader::loadEngineConfig(path);
    EXPECT_EQ(config.applicationName, "NoShaderPath");
    EXPECT_EQ(config.shaderLibraryPath, "shaders/RTRShaders.metallib");
    EXPECT_EQ(config.shadingMode, "auto");

    std::filesystem::remove(path);
}

TEST(ConfigLoader, ParsesOptionalShadingMode) {
    const auto path = makeTempFile("rtr_engine_config_shading.ini",
                                   R"(applicationName = Sample
shaderLibraryPath = assets/RTR.metallib
shadingMode = cpu
)");

    const rtr::core::EngineConfig config = rtr::core::ConfigLoader::loadEngineConfig(path);
    EXPECT_EQ(config.applicationName, "Sample");
    EXPECT_EQ(config.shaderLibraryPath, "assets/RTR.metallib");
    EXPECT_EQ(config.shadingMode, "cpu");

    std::filesystem::remove(path);
}

TEST(ConfigLoader, ParsesAccumulationOptions) {
    const auto path = makeTempFile("rtr_engine_config_accum.ini",
                                   R"(applicationName = Sample
shaderLibraryPath = assets/RTR.metallib
accumulation = off
accumulationFrames = 8
samplesPerPixel = 4
sampleSeed = 123
maxBounces = 3
)");

    const rtr::core::EngineConfig config = rtr::core::ConfigLoader::loadEngineConfig(path);
    EXPECT_FALSE(config.accumulationEnabled);
    EXPECT_EQ(config.accumulationFrames, 8u);
    EXPECT_EQ(config.samplesPerPixel, 4u);
    EXPECT_EQ(config.sampleSeed, 123u);
    EXPECT_EQ(config.maxHardwareBounces, 3u);

    std::filesystem::remove(path);
}

TEST(ConfigLoader, ClampsZeroMaxBounces) {
    const auto path = makeTempFile("rtr_engine_config_bounces.ini",
                                   R"(applicationName = Sample
shaderLibraryPath = assets/RTR.metallib
maxBounces = 0
)");

    const rtr::core::EngineConfig config = rtr::core::ConfigLoader::loadEngineConfig(path);
    EXPECT_EQ(config.maxHardwareBounces, 1u);

    std::filesystem::remove(path);
}

TEST(ConfigLoader, ThrowsOnMissingApplicationName) {
    const auto path = makeTempFile("rtr_engine_config_invalid.ini",
                                   R"(shaderLibraryPath = foo
)");

    EXPECT_THROW({ rtr::core::ConfigLoader::loadEngineConfig(path); }, std::runtime_error);

    std::filesystem::remove(path);
}
