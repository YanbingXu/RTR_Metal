#pragma once

#include <simd/simd.h>

namespace rtr::core::math {

constexpr float pi() noexcept { return 3.14159265358979323846F; }
constexpr float tau() noexcept { return 2.0F * pi(); }
constexpr float halfPi() noexcept { return 0.5F * pi(); }

constexpr float radians(float degrees) noexcept { return degrees * (pi() / 180.0F); }
constexpr float degrees(float radiansValue) noexcept { return radiansValue * (180.0F / pi()); }

/// Builds a right-handed perspective projection matrix matching Metal's clip space.
simd_float4x4 makePerspective(float verticalFovRadians, float aspectRatio, float nearPlane, float farPlane) noexcept;

/// Builds a right-handed look-at matrix with `forward = normalize(target - eye)`.
simd_float4x4 makeLookAt(simd_float3 eye, simd_float3 target, simd_float3 up) noexcept;

/// Axis-aligned bounding box utility for scene import and AS construction.
struct BoundingBox {
    simd_float3 min;
    simd_float3 max;

    static BoundingBox makeEmpty() noexcept;
    static BoundingBox fromPoints(simd_float3 a, simd_float3 b) noexcept;

    void expand(simd_float3 point) noexcept;
    [[nodiscard]] simd_float3 extent() const noexcept;
};

}  // namespace rtr::core::math
