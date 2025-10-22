#include <gtest/gtest.h>

#include <cmath>
#include <simd/simd.h>

#include "RTRMetalEngine/Core/Math.hpp"

namespace {

constexpr float kEpsilon = 1e-4F;

}  // namespace

TEST(MathConstants, RadiansToDegreesRoundTrip) {
    constexpr float deg = 45.0F;
    const float rad = rtr::core::math::radians(deg);
    EXPECT_NEAR(rtr::core::math::degrees(rad), deg, kEpsilon);
}

TEST(MathPerspective, ProducesExpectedZMapping) {
    const float nearPlane = 0.1F;
    const float farPlane = 10.0F;
    const simd_float4x4 matrix = rtr::core::math::makePerspective(rtr::core::math::halfPi(), 16.0F / 9.0F, nearPlane, farPlane);

    EXPECT_NEAR(matrix.columns[0][0], 1.0F / std::tan(rtr::core::math::halfPi() * 0.5F) * (9.0F / 16.0F), 1e-3F);
    EXPECT_FLOAT_EQ(matrix.columns[2][2], (farPlane + nearPlane) / (nearPlane - farPlane));
    EXPECT_FLOAT_EQ(matrix.columns[3][2], (2.0F * farPlane * nearPlane) / (nearPlane - farPlane));
}

TEST(MathLookAt, GeneratesOrthogonalBasis) {
    const simd_float3 eye = simd_make_float3(0.0F, 0.0F, 5.0F);
    const simd_float3 target = simd_make_float3(0.0F, 0.0F, 0.0F);
    const simd_float3 up = simd_make_float3(0.0F, 1.0F, 0.0F);

    const simd_float4x4 matrix = rtr::core::math::makeLookAt(eye, target, up);

    const simd_float3 right = simd_make_float3(matrix.columns[0][0], matrix.columns[0][1], matrix.columns[0][2]);
    const simd_float3 upVector = simd_make_float3(matrix.columns[1][0], matrix.columns[1][1], matrix.columns[1][2]);
    const simd_float3 forward = -simd_make_float3(matrix.columns[2][0], matrix.columns[2][1], matrix.columns[2][2]);

    EXPECT_NEAR(simd_length(right), 1.0F, kEpsilon);
    EXPECT_NEAR(simd_length(upVector), 1.0F, kEpsilon);
    EXPECT_NEAR(simd_length(forward), 1.0F, kEpsilon);
    EXPECT_NEAR(simd_dot(right, upVector), 0.0F, kEpsilon);
    EXPECT_NEAR(simd_dot(right, forward), 0.0F, kEpsilon);
    EXPECT_NEAR(simd_dot(upVector, forward), 0.0F, kEpsilon);
}

TEST(BoundingBox, ExpandsCorrectly) {
    using rtr::core::math::BoundingBox;
    BoundingBox box = BoundingBox::makeEmpty();
    box.expand(simd_make_float3(1.0F, -1.0F, 0.5F));
    box.expand(simd_make_float3(-2.0F, 2.0F, 3.0F));

    EXPECT_FLOAT_EQ(box.min.x, -2.0F);
    EXPECT_FLOAT_EQ(box.min.y, -1.0F);
    EXPECT_FLOAT_EQ(box.min.z, 0.5F);
    EXPECT_FLOAT_EQ(box.max.x, 1.0F);
    EXPECT_FLOAT_EQ(box.max.y, 2.0F);
    EXPECT_FLOAT_EQ(box.max.z, 3.0F);

    const simd_float3 extent = box.extent();
    EXPECT_FLOAT_EQ(extent.x, 3.0F);
    EXPECT_FLOAT_EQ(extent.y, 3.0F);
    EXPECT_FLOAT_EQ(extent.z, 2.5F);
}
