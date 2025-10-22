#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "RTRMetalEngine/MPS/MPSRenderer.hpp"

#include <array>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <limits>
#include <span>
#include <simd/simd.h>

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"
#include "RTRMetalEngine/MPS/MPSSceneConverter.hpp"
#include "RTRMetalEngine/Scene/Scene.hpp"
#include "RTRMetalEngine/Scene/SceneBuilder.hpp"

namespace rtr::rendering {

namespace {

bool writePPM(const char* path, const std::vector<uint8_t>& data, std::uint32_t width, std::uint32_t height) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(file);
}

}  // namespace

MPSRenderer::MPSRenderer(MetalContext& context)
    : context_(context) {}

bool MPSRenderer::initialize() {
    scene::Scene scene;
    scene::SceneBuilder builder(scene);

    const std::array<simd_float3, 3> positions = {
        simd_make_float3(-0.5f, -0.5f, 0.0f),
        simd_make_float3(0.5f, -0.5f, 0.0f),
        simd_make_float3(0.0f, 0.5f, 0.0f),
    };
    const std::array<std::uint32_t, 3> indices = {0, 1, 2};

    auto meshHandle = builder.addTriangleMesh(positions, indices);
    auto materialHandle = builder.addDefaultMaterial();
    scene.addInstance(meshHandle, materialHandle, matrix_identity_float4x4);

    return initialize(scene);
}

bool MPSRenderer::initialize(const scene::Scene& scene) {
    if (!pathTracer_.initialize(context_)) {
        core::Logger::warn("MPSRenderer", "Failed to initialize MPS path tracer device state");
        return false;
    }

    const MPSSceneData sceneData = buildSceneData(scene);
    if (sceneData.positions.empty() || sceneData.indices.empty()) {
        core::Logger::warn("MPSRenderer", "Scene conversion produced no geometry");
        return false;
    }

    const std::span<const vector_float3> positionSpan(sceneData.positions.data(), sceneData.positions.size());
    const std::span<const uint32_t> indexSpan(sceneData.indices.data(), sceneData.indices.size());
    const std::span<const vector_float3> colorSpan(sceneData.colors.data(), sceneData.colors.size());

    if (!pathTracer_.uploadTriangleScene(positionSpan, indexSpan, colorSpan)) {
        core::Logger::error("MPSRenderer", "Failed to upload scene geometry to MPS path tracer");
        return false;
    }

    core::Logger::info("MPSRenderer", "Loaded scene with %zu vertices and %zu triangles",
                       sceneData.positions.size(), sceneData.indices.size() / 3);
    return true;
}

bool MPSRenderer::renderFrame(const char* outputPath) {
    if (!pathTracer_.isValid()) {
        core::Logger::warn("MPSRenderer", "Path tracer not initialized; skipping frame");
        return false;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)pathTracer_.deviceHandle();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)pathTracer_.commandQueueHandle();
    MPSTriangleAccelerationStructure* accelerationStructure = (__bridge MPSTriangleAccelerationStructure*)pathTracer_.accelerationStructureHandle();
    MPSRayIntersector* intersector = (__bridge MPSRayIntersector*)pathTracer_.intersectorHandle();

    if (!device || !queue || !accelerationStructure || !intersector) {
        core::Logger::error("MPSRenderer", "Invalid MPS state; aborting render");
        return false;
    }

    constexpr std::uint32_t width = 512;
    constexpr std::uint32_t height = 512;
    const std::size_t pixelCount = static_cast<std::size_t>(width) * height;

    id<MTLBuffer> rayBuffer =
        [device newBufferWithLength:pixelCount * sizeof(MPSRayOriginMaskDirectionMaxDistance)
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> intersectionBuffer = [device newBufferWithLength:pixelCount * sizeof(MPSIntersectionDistancePrimitiveIndexCoordinates)
                                                           options:MTLResourceStorageModeShared];

    if (!rayBuffer || !intersectionBuffer) {
        core::Logger::error("MPSRenderer", "Failed to allocate ray or intersection buffers");
        return false;
    }

    auto* rays = reinterpret_cast<MPSRayOriginMaskDirectionMaxDistance*>([rayBuffer contents]);
    const vector_float3 cameraOrigin = {0.0f, 0.0f, 1.5f};
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const float u = (static_cast<float>(x) / static_cast<float>(width) - 0.5f) * 2.0f;
            const float v = (static_cast<float>(y) / static_cast<float>(height) - 0.5f) * 2.0f;
            const vector_float3 targetPoint = {u, v, 0.0f};
            const vector_float3 direction = simd_normalize(targetPoint - cameraOrigin);

            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            rays[idx].origin = cameraOrigin;
            rays[idx].direction = direction;
            rays[idx].mask = 0xFFFFFFFF;
            rays[idx].maxDistance = FLT_MAX;
        }
    }
    [rayBuffer didModifyRange:NSMakeRange(0, rayBuffer.length)];

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) {
        core::Logger::error("MPSRenderer", "Failed to create command buffer");
        return false;
    }

    [intersector encodeIntersectionToCommandBuffer:commandBuffer
                                 intersectionType:MPSIntersectionTypeNearest
                                         rayBuffer:rayBuffer
                                   rayBufferOffset:0
                              intersectionBuffer:intersectionBuffer
                        intersectionBufferOffset:0
                                         rayCount:pixelCount
                          accelerationStructure:accelerationStructure];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    auto* intersections = reinterpret_cast<MPSIntersectionDistancePrimitiveIndexCoordinates*>([intersectionBuffer contents]);
    const auto& vertices = pathTracer_.vertexPositions();
    const auto& indices = pathTracer_.indices();
    const auto& colors = pathTracer_.vertexColors();

    vector_float3 lightDir = simd_normalize((vector_float3){0.2f, 0.8f, 0.6f});
    std::vector<uint8_t> pixels(pixelCount * 3);
    std::size_t hitCount = 0;
    constexpr std::uint32_t kNoHit = std::numeric_limits<std::uint32_t>::max();
    for (std::size_t i = 0; i < pixelCount; ++i) {
        const auto& intersection = intersections[i];
        const bool hit = intersection.distance < FLT_MAX && intersection.primitiveIndex != kNoHit && indices.size() >= 3;
        vector_float3 color = {0.08f, 0.08f, 0.12f};
        if (hit) {
            ++hitCount;
            const uint32_t primitiveIndex = intersection.primitiveIndex;
            const std::size_t base = static_cast<std::size_t>(primitiveIndex) * 3;
            if ((base + 2) < indices.size()) {
                const uint32_t i0 = indices[base + 0];
                const uint32_t i1 = indices[base + 1];
                const uint32_t i2 = indices[base + 2];
                if (i0 < vertices.size() && i1 < vertices.size() && i2 < vertices.size()) {
                    const vector_float3& v0 = vertices[i0];
                    const vector_float3& v1 = vertices[i1];
                    const vector_float3& v2 = vertices[i2];

                    vector_float3 e1 = v1 - v0;
                    vector_float3 e2 = v2 - v0;
                    vector_float3 normal = simd_normalize(simd_cross(e1, e2));

                    float intensity = fmaxf(0.0f, simd_dot(normal, lightDir));
                    intensity = intensity * 0.8f + 0.2f;

                    float u = intersection.coordinates.x;
                    float v = intersection.coordinates.y;
                    float w = 1.0f - u - v;

                    vector_float3 c0 = (i0 < colors.size()) ? colors[i0] : vector_float3{0.85f, 0.4f, 0.25f};
                    vector_float3 c1 = (i1 < colors.size()) ? colors[i1] : vector_float3{0.25f, 0.85f, 0.4f};
                    vector_float3 c2 = (i2 < colors.size()) ? colors[i2] : vector_float3{0.4f, 0.25f, 0.85f};
                    color = (c0 * w + c1 * u + c2 * v) * intensity;
                }
            }
        }
        color = simd_clamp(color, (vector_float3){0.0f, 0.0f, 0.0f}, (vector_float3){1.0f, 1.0f, 1.0f});
        pixels[i * 3 + 0] = static_cast<uint8_t>(color.x * 255.0f);
        pixels[i * 3 + 1] = static_cast<uint8_t>(color.y * 255.0f);
        pixels[i * 3 + 2] = static_cast<uint8_t>(color.z * 255.0f);
    }

    core::Logger::info("MPSRenderer", "Intersection hits: %zu / %zu", hitCount, pixelCount);

    if (!writePPM(outputPath, pixels, width, height)) {
        core::Logger::error("MPSRenderer", "Failed to write %s", outputPath);
        return false;
    }

    core::Logger::info("MPSRenderer", "Wrote diagnostic image to %s", outputPath);
    return true;
}

}  // namespace rtr::rendering

#pragma clang diagnostic pop
