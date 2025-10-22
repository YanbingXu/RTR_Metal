#include <gtest/gtest.h>

#include "RTRMetalEngine/Core/Logger.hpp"

TEST(Logger, EmitsWithoutCrashing) {
    rtr::core::Logger::info("Test", "Hello %d", 42);
    rtr::core::Logger::warn("Test", "Warning %s", "message");
    rtr::core::Logger::error("Test", "Error code %d", -1);
    SUCCEED();
}
