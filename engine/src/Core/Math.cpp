#include "RTRMetalEngine/Core/Math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rtr::core::math {

namespace {

constexpr simd_float3 safeNormalize(simd_float3 v) noexcept {
    const float lengthSquared = simd_dot(v, v);
    if (lengthSquared <= std::numeric_limits<float>::epsilon()) {
        return simd_make_float3(0.0F, 0.0F, 0.0F);
    }
    return simd_normalize(v);
}

}  // namespace

simd_float4x4 makePerspective(float verticalFovRadians, float aspectRatio, float nearPlane, float farPlane) noexcept {
    const float f = 1.0F / std::tan(0.5F * verticalFovRadians);
    const float invDepth = 1.0F / (nearPlane - farPlane);

    simd_float4x4 matrix = {
        simd_make_float4(f / aspectRatio, 0.0F, 0.0F, 0.0F),
        simd_make_float4(0.0F, f, 0.0F, 0.0F),
        simd_make_float4(0.0F, 0.0F, (farPlane + nearPlane) * invDepth, -1.0F),
        simd_make_float4(0.0F, 0.0F, (2.0F * farPlane * nearPlane) * invDepth, 0.0F)};

    return matrix;
}

simd_float4x4 makeLookAt(simd_float3 eye, simd_float3 target, simd_float3 up) noexcept {
    const simd_float3 forward = safeNormalize(target - eye);
    const simd_float3 right = safeNormalize(simd_cross(forward, up));
    const simd_float3 correctedUp = simd_cross(right, forward);

    simd_float4x4 matrix = {
        simd_make_float4(right.x, correctedUp.x, -forward.x, 0.0F),
        simd_make_float4(right.y, correctedUp.y, -forward.y, 0.0F),
        simd_make_float4(right.z, correctedUp.z, -forward.z, 0.0F),
        simd_make_float4(-simd_dot(right, eye), -simd_dot(correctedUp, eye), simd_dot(forward, eye), 1.0F)};

    return matrix;
}

BoundingBox BoundingBox::makeEmpty() noexcept {
    constexpr float inf = std::numeric_limits<float>::infinity();
    return BoundingBox{simd_make_float3(inf, inf, inf), simd_make_float3(-inf, -inf, -inf)};
}

BoundingBox BoundingBox::fromPoints(simd_float3 a, simd_float3 b) noexcept {
    BoundingBox box;
    box.min = simd_min(a, b);
    box.max = simd_max(a, b);
    return box;
}

void BoundingBox::expand(simd_float3 point) noexcept {
    min = simd_min(min, point);
    max = simd_max(max, point);
}

simd_float3 BoundingBox::extent() const noexcept { return max - min; }

}  // namespace rtr::core::math
