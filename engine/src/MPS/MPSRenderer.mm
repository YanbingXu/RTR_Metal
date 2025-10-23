#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "RTRMetalEngine/MPS/MPSRenderer.hpp"

#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <simd/simd.h>

#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/MPS/MPSPathTracer.hpp"
#include "RTRMetalEngine/MPS/MPSUniforms.hpp"
#include "RTRMetalEngine/MPS/MPSSceneConverter.hpp"
#include "RTRMetalEngine/Rendering/MetalContext.hpp"
#include "RTRMetalEngine/Rendering/GeometryStore.hpp"
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

struct MPSRenderer::GPUState {
    id<MTLComputePipelineState> rayPipeline = nil;
    id<MTLComputePipelineState> shadePipeline = nil;
    BufferHandle positionsBuffer;
    BufferHandle indicesBuffer;
    BufferHandle colorsBuffer;
    BufferHandle intersectionsBuffer;
    BufferHandle shadingOutputBuffer;
};

MPSRenderer::MPSRenderer(MetalContext& context)
    : context_(context), bufferAllocator_(context), geometryStore_(bufferAllocator_) {}

MPSRenderer::~MPSRenderer() = default;

void MPSRenderer::createUniformBuffer() {
    uniformBuffer_ = bufferAllocator_.createBuffer(sizeof(MPSCameraUniforms), nullptr, "mps.cameraUniforms");
    if (!uniformBuffer_.isValid()) {
        core::Logger::warn("MPSRenderer", "Failed to allocate camera uniform buffer");
    }
}

void MPSRenderer::updateCameraUniforms(std::uint32_t width, std::uint32_t height) {
    const vector_float3 cameraOrigin = {0.0f, 0.0f, 1.5f};
    const vector_float3 target = {0.0f, 0.0f, 0.0f};
    const vector_float3 worldUp = {0.0f, 1.0f, 0.0f};

    cameraUniforms_.eye = cameraOrigin;
    cameraUniforms_.forward = simd_normalize(target - cameraOrigin);
    vector_float3 right = simd_cross(cameraUniforms_.forward, worldUp);
    if (simd_length(right) < 1e-5f) {
        right = {1.0f, 0.0f, 0.0f};
    }
    cameraUniforms_.right = simd_normalize(right);
    cameraUniforms_.up = simd_normalize(simd_cross(cameraUniforms_.right, cameraUniforms_.forward));
    cameraUniforms_.imagePlaneHalfExtents = {1.0f, 1.0f};
    cameraUniforms_.width = width;
    cameraUniforms_.height = height;

    if (!uniformBuffer_.isValid()) {
        return;
    }

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)uniformBuffer_.nativeHandle();
    if (!buffer) {
        return;
    }

    std::memcpy([buffer contents], &cameraUniforms_, sizeof(cameraUniforms_));
    [buffer didModifyRange:NSMakeRange(0, sizeof(cameraUniforms_))];
}

bool MPSRenderer::initializeGPUResources() {
    if (!kEnableGPUShading) {
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_.rawDeviceHandle();
    if (!device) {
        core::Logger::warn("MPSRenderer", "Cannot create GPU pipelines: Metal device unavailable");
        return false;
    }

    if (!gpuState_) {
        gpuState_ = std::make_unique<GPUState>();
    }

    NSError* error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) {
        core::Logger::warn("MPSRenderer", "Failed to load default Metal library for GPU shading");
        return false;
    }

    if (!gpuState_->rayPipeline) {
        id<MTLFunction> rayFunction = [library newFunctionWithName:@"mpsRayKernel"];
        if (!rayFunction) {
            core::Logger::warn("MPSRenderer", "Metal library missing mpsRayKernel; GPU shading disabled");
            return false;
        }

        gpuState_->rayPipeline = [device newComputePipelineStateWithFunction:rayFunction error:&error];
        if (error || !gpuState_->rayPipeline) {
            core::Logger::warn("MPSRenderer", "Failed to create ray compute pipeline state: %s",
                               error.localizedDescription.UTF8String);
            gpuState_.reset();
            return false;
        }
        core::Logger::info("MPSRenderer", "GPU ray generation pipeline initialised");
    }

    if (!gpuState_->shadePipeline) {
        id<MTLFunction> shadeFunction = [library newFunctionWithName:@"mpsShadeKernel"];
        if (!shadeFunction) {
            core::Logger::warn("MPSRenderer", "Metal library missing mpsShadeKernel; GPU shading disabled");
            return false;
        }

        gpuState_->shadePipeline = [device newComputePipelineStateWithFunction:shadeFunction error:&error];
        if (error || !gpuState_->shadePipeline) {
            core::Logger::warn("MPSRenderer", "Failed to create shade compute pipeline state: %s",
                               error.localizedDescription.UTF8String);
            gpuState_->shadePipeline = nil;
            return false;
        }
        core::Logger::info("MPSRenderer", "GPU shading pipeline initialised");
    }

    auto uploadStaticBuffer = [&](BufferHandle& handle, const void* data, std::size_t length, const char* label) {
        if (length == 0) {
            return;
        }

        if (!handle.isValid() || handle.length() < length) {
            handle = bufferAllocator_.createBuffer(length, data, label);
        } else {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle.nativeHandle();
            if (buffer && data) {
                std::memcpy([buffer contents], data, length);
                [buffer didModifyRange:NSMakeRange(0, length)];
            }
        }
    };

    uploadStaticBuffer(gpuState_->positionsBuffer, cpuScenePositions_.data(),
                       cpuScenePositions_.size() * sizeof(vector_float3), "mps.positions");
    uploadStaticBuffer(gpuState_->indicesBuffer, cpuSceneIndices_.data(),
                       cpuSceneIndices_.size() * sizeof(uint32_t), "mps.indices");
    uploadStaticBuffer(gpuState_->colorsBuffer, cpuSceneColors_.data(),
                       cpuSceneColors_.size() * sizeof(vector_float3), "mps.colors");

    return true;
}

bool MPSRenderer::initialize() {
    scene::Scene scene;
    scene::SceneBuilder builder(scene);
    const std::array<simd_float3, 6> floorPositions = {
        simd_make_float3(-1.0f, -0.3f, -1.0f),
        simd_make_float3(1.0f, -0.3f, -1.0f),
        simd_make_float3(1.0f, -0.3f, 1.0f),
        simd_make_float3(-1.0f, -0.3f, -1.0f),
        simd_make_float3(1.0f, -0.3f, 1.0f),
        simd_make_float3(-1.0f, -0.3f, 1.0f),
    };
    const std::array<std::uint32_t, 6> floorIndices = {0U, 1U, 2U, 3U, 4U, 5U};
    auto floorMesh = builder.addTriangleMesh(floorPositions, floorIndices);

    scene::Material floorMaterial{};
    floorMaterial.albedo = {0.85f, 0.85f, 0.85f};
    auto floorMaterialHandle = scene.addMaterial(floorMaterial);
    scene.addInstance(floorMesh, floorMaterialHandle, matrix_identity_float4x4);

    const std::array<simd_float3, 3> prismPositions = {
        simd_make_float3(-0.35f, -0.1f, 0.25f),
        simd_make_float3(0.35f, -0.1f, -0.05f),
        simd_make_float3(0.0f, 0.45f, 0.35f),
    };
    const std::array<std::uint32_t, 3> prismIndices = {0U, 1U, 2U};
    auto prismMesh = builder.addTriangleMesh(prismPositions, prismIndices);

    scene::Material prismMaterial{};
    prismMaterial.albedo = {0.9f, 0.45f, 0.25f};
    auto prismMaterialHandle = scene.addMaterial(prismMaterial);
    scene.addInstance(prismMesh, prismMaterialHandle, matrix_identity_float4x4);

    return initialize(scene);
}

bool MPSRenderer::initialize(const scene::Scene& scene) {
    if (!pathTracer_.initialize(context_)) {
        core::Logger::warn("MPSRenderer", "Failed to initialize MPS path tracer device state");
        return false;
    }

    if (!uniformBuffer_.isValid()) {
        createUniformBuffer();
    }

    const MPSSceneData sceneData = buildSceneData(scene);
    if (sceneData.positions.empty() || sceneData.indices.empty()) {
        core::Logger::warn("MPSRenderer", "Scene conversion produced no geometry");
        return false;
    }

    const std::span<const vector_float3> positionSpan(sceneData.positions.data(), sceneData.positions.size());
    const std::span<const uint32_t> indexSpan(sceneData.indices.data(), sceneData.indices.size());

    if (!pathTracer_.uploadScene(positionSpan, indexSpan)) {
        core::Logger::error("MPSRenderer", "Failed to upload scene geometry to MPS path tracer");
        return false;
    }

    cpuScenePositions_ = sceneData.positions;
    cpuSceneIndices_ = sceneData.indices;
    cpuSceneColors_ = sceneData.colors;

    if (!initializeGPUResources()) {
        core::Logger::warn("MPSRenderer", "GPU shading resources unavailable; falling back to CPU shading");
    }

    core::Logger::info("MPSRenderer", "Loaded scene with %zu vertices and %zu triangles",
                       cpuScenePositions_.size(), cpuSceneIndices_.size() / 3);
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

    updateCameraUniforms(width, height);

    id<MTLBuffer> rayBuffer =
        [device newBufferWithLength:pixelCount * sizeof(MPSRayOriginMaskDirectionMaxDistance)
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> intersectionBuffer = [device newBufferWithLength:pixelCount * sizeof(MPSIntersectionDistancePrimitiveIndexCoordinates)
                                                           options:MTLResourceStorageModeShared];

    if (!rayBuffer || !intersectionBuffer) {
        core::Logger::error("MPSRenderer", "Failed to allocate ray or intersection buffers");
        return false;
    }

    if (kEnableGPUShading && gpuState_ && gpuState_->rayPipeline) {
        id<MTLCommandBuffer> setupCommandBuffer = [queue commandBuffer];
        if (!setupCommandBuffer) {
            core::Logger::error("MPSRenderer", "Failed to create command buffer for ray generation");
            return false;
        }

        id<MTLComputeCommandEncoder> computeEncoder = [setupCommandBuffer computeCommandEncoder];
        if (!computeEncoder) {
            core::Logger::error("MPSRenderer", "Failed to create compute encoder for ray generation");
            return false;
        }

        [computeEncoder setComputePipelineState:gpuState_->rayPipeline];
        [computeEncoder setBuffer:rayBuffer offset:0 atIndex:0];
        if (uniformBuffer_.isValid()) {
            id<MTLBuffer> uniforms = (__bridge id<MTLBuffer>)uniformBuffer_.nativeHandle();
            [computeEncoder setBuffer:uniforms offset:0 atIndex:1];
        }

        const NSUInteger threadWidth = gpuState_->rayPipeline.threadExecutionWidth;
        const NSUInteger threadHeight = gpuState_->rayPipeline.maxTotalThreadsPerThreadgroup / threadWidth;
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight > 0 ? threadHeight : 1, 1);
        MTLSize threadgroups = MTLSizeMake((width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                           (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                           1);
        [computeEncoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [computeEncoder endEncoding];
        [setupCommandBuffer commit];
        [setupCommandBuffer waitUntilCompleted];
    } else {
        auto* rays = reinterpret_cast<MPSRayOriginMaskDirectionMaxDistance*>([rayBuffer contents]);
        const vector_float3 cameraOrigin = cameraUniforms_.eye;
        const vector_float3 right = cameraUniforms_.right;
        const vector_float3 up = cameraUniforms_.up;
        const vector_float3 forward = cameraUniforms_.forward;
        const vector_float2 halfExtents = cameraUniforms_.imagePlaneHalfExtents;
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                const vector_float2 pixel = {static_cast<float>(x), static_cast<float>(y)};
                const vector_float2 screenSize = {static_cast<float>(width), static_cast<float>(height)};
                const vector_float2 ndc = (pixel / screenSize - 0.5f) * 2.0f;
                const vector_float3 targetPoint = cameraOrigin + forward + right * (ndc.x * halfExtents.x) +
                                                  up * (ndc.y * halfExtents.y);
                const vector_float3 direction = simd_normalize(targetPoint - cameraOrigin);

                const std::size_t idx = static_cast<std::size_t>(y) * width + x;
                rays[idx].origin = cameraOrigin;
                rays[idx].direction = direction;
                rays[idx].mask = 0xFFFFFFFF;
                rays[idx].maxDistance = FLT_MAX;
            }
        }
        [rayBuffer didModifyRange:NSMakeRange(0, rayBuffer.length)];
    }

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
    const auto& vertices = cpuScenePositions_;
    const auto& indices = cpuSceneIndices_;
    const auto& colors = cpuSceneColors_;

    std::vector<uint8_t> pixels(pixelCount * 3);
    std::vector<uint8_t> gpuPixels;
    bool usedGPUShading = false;

    if (kEnableGPUShading && gpuState_ && gpuState_->shadePipeline && gpuState_->positionsBuffer.isValid() &&
        gpuState_->indicesBuffer.isValid() && gpuState_->colorsBuffer.isValid()) {
        std::vector<MPSIntersectionData> gpuIntersections(pixelCount);
        for (std::size_t i = 0; i < pixelCount; ++i) {
            const auto& src = intersections[i];
            gpuIntersections[i].distance = src.distance;
            gpuIntersections[i].primitiveIndex = src.primitiveIndex;
            gpuIntersections[i].barycentric = {src.coordinates.x, src.coordinates.y};
            gpuIntersections[i].padding = 0.0f;
        }

        const std::size_t intersectionsByteLength = gpuIntersections.size() * sizeof(MPSIntersectionData);
        if (!gpuState_->intersectionsBuffer.isValid() ||
            gpuState_->intersectionsBuffer.length() < intersectionsByteLength) {
            gpuState_->intersectionsBuffer = bufferAllocator_.createBuffer(intersectionsByteLength,
                                                                          gpuIntersections.data(),
                                                                          "mps.intersections");
        } else {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)gpuState_->intersectionsBuffer.nativeHandle();
            if (buffer) {
                std::memcpy([buffer contents], gpuIntersections.data(), intersectionsByteLength);
                [buffer didModifyRange:NSMakeRange(0, intersectionsByteLength)];
            }
        }

        const std::size_t outputByteLength = pixelCount * sizeof(simd_float4);
        if (!gpuState_->shadingOutputBuffer.isValid() || gpuState_->shadingOutputBuffer.length() < outputByteLength) {
            gpuState_->shadingOutputBuffer = bufferAllocator_.createBuffer(outputByteLength, nullptr, "mps.shadingOutput");
        }

        if (gpuState_->intersectionsBuffer.isValid() && gpuState_->shadingOutputBuffer.isValid()) {
            id<MTLCommandBuffer> shadeCommandBuffer = [queue commandBuffer];
            if (shadeCommandBuffer) {
                id<MTLComputeCommandEncoder> shadeEncoder = [shadeCommandBuffer computeCommandEncoder];
                if (shadeEncoder) {
                    [shadeEncoder setComputePipelineState:gpuState_->shadePipeline];
                    [shadeEncoder setBuffer:(__bridge id<MTLBuffer>)gpuState_->intersectionsBuffer.nativeHandle()
                                   offset:0
                                  atIndex:0];
                    [shadeEncoder setBuffer:(__bridge id<MTLBuffer>)gpuState_->positionsBuffer.nativeHandle()
                                   offset:0
                                  atIndex:1];
                    [shadeEncoder setBuffer:(__bridge id<MTLBuffer>)gpuState_->indicesBuffer.nativeHandle()
                                   offset:0
                                  atIndex:2];
                    [shadeEncoder setBuffer:(__bridge id<MTLBuffer>)gpuState_->colorsBuffer.nativeHandle()
                                   offset:0
                                  atIndex:3];
                    [shadeEncoder setBuffer:(__bridge id<MTLBuffer>)gpuState_->shadingOutputBuffer.nativeHandle()
                                   offset:0
                                  atIndex:4];
                    if (uniformBuffer_.isValid()) {
                        [shadeEncoder setBuffer:(__bridge id<MTLBuffer>)uniformBuffer_.nativeHandle()
                                       offset:0
                                      atIndex:5];
                    }

                    const NSUInteger threadWidth = gpuState_->shadePipeline.threadExecutionWidth;
                    const NSUInteger threadsPerThreadgroup = gpuState_->shadePipeline.maxTotalThreadsPerThreadgroup;
                    const NSUInteger threadHeight = threadsPerThreadgroup / threadWidth;
                    const NSUInteger threadgroupSize = threadWidth * (threadHeight > 0 ? threadHeight : 1);
                    const NSUInteger totalThreads = static_cast<NSUInteger>(pixelCount);
                    MTLSize threadsPerGroup = MTLSizeMake(threadgroupSize, 1, 1);
                    MTLSize groups = MTLSizeMake((totalThreads + threadgroupSize - 1) / threadgroupSize, 1, 1);
                    [shadeEncoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
                    [shadeEncoder endEncoding];
                    [shadeCommandBuffer commit];
                    [shadeCommandBuffer waitUntilCompleted];

                    auto* gpuOutput = reinterpret_cast<simd_float4*>(
                        [(__bridge id<MTLBuffer>)gpuState_->shadingOutputBuffer.nativeHandle() contents]);
                    if (gpuOutput) {
                        gpuPixels.resize(pixelCount * 3);
                        for (std::size_t i = 0; i < pixelCount; ++i) {
                            const simd_float4 c = simd_clamp(gpuOutput[i], (simd_float4){0.0f}, (simd_float4){1.0f});
                            gpuPixels[i * 3 + 0] = static_cast<uint8_t>(c.x * 255.0f);
                            gpuPixels[i * 3 + 1] = static_cast<uint8_t>(c.y * 255.0f);
                            gpuPixels[i * 3 + 2] = static_cast<uint8_t>(c.z * 255.0f);
                        }
                        usedGPUShading = true;
                    }
                }
            }
        }
    }

    vector_float3 lightDir = simd_normalize((vector_float3){0.2f, 0.8f, 0.6f});
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

    if (usedGPUShading && gpuPixels.size() == pixels.size()) {
        float maxDifference = 0.0f;
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            maxDifference = std::max(maxDifference, std::fabs(static_cast<float>(pixels[i]) - gpuPixels[i]));
        }

        if (maxDifference <= 2.0f) {
            pixels = std::move(gpuPixels);
            core::Logger::info("MPSRenderer", "GPU shading matched CPU output (max diff %.2f)", maxDifference);
        } else {
            core::Logger::warn("MPSRenderer", "GPU shading diverged from CPU output (max diff %.2f)", maxDifference);
        }
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
