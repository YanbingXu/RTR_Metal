#include <gtest/gtest.h>

#include "RTRMetalEngine/Rendering/BufferAllocator.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"

TEST(BufferAllocator, HandlesInvalidContextGracefully) {
    rtr::rendering::MetalContext context;
    ASSERT_FALSE(context.isValid());

    rtr::rendering::BufferAllocator allocator(context);
    const auto handle = allocator.createBuffer(256);

    EXPECT_FALSE(handle.isValid());
    EXPECT_EQ(handle.length(), 0U);
}

TEST(BufferAllocator, ZeroLengthBuffersAreRejected) {
    rtr::rendering::MetalContext context;
    rtr::rendering::BufferAllocator allocator(context);

    const auto handle = allocator.createBuffer(0);
    EXPECT_FALSE(handle.isValid());
}
