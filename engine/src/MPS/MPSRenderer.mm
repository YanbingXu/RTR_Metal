#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "RTRMetalEngine/MPS/MPSRenderer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
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

bool parseEnvBool(const char* value, bool defaultValue) {
    if (!value) {
        return defaultValue;
    }

    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lowered == "0" || lowered == "false" || lowered == "off") {
        return false;
    }
    if (lowered == "1" || lowered == "true" || lowered == "on") {
        return true;
    }
    return defaultValue;
}

uint8_t floatToSRGBByte(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    const long rounded = std::lroundf(clamped * 255.0f);
    return static_cast<uint8_t>(std::clamp<long>(rounded, 0, 255));
}

MPSRayOriginMaskDirectionMaxDistance makePrimaryRay(std::uint32_t x,
                                                    std::uint32_t y,
                                                    std::uint32_t width,
                                                    std::uint32_t height,
                                                    const MPSCameraUniforms& uniforms) {
    MPSRayOriginMaskDirectionMaxDistance ray{};
    const vector_float2 pixel = {static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
    const vector_float2 screenSize = {static_cast<float>(width), static_cast<float>(height)};
    const vector_float2 ndc = (pixel / screenSize - 0.5f) * 2.0f;

    const vector_float3 eye = simd_make_float3(uniforms.eye);
    const vector_float3 forward = simd_make_float3(uniforms.forward);
    const vector_float3 right = simd_make_float3(uniforms.right);
    const vector_float3 up = simd_make_float3(uniforms.up);
    const vector_float2 halfExtents = uniforms.imagePlaneHalfExtents;
    const vector_float3 targetPoint = eye + forward + right * (ndc.x * halfExtents.x) + up * (ndc.y * halfExtents.y);
    const vector_float3 direction = simd_normalize(targetPoint - eye);

    ray.origin = eye;
    ray.direction = direction;
    ray.mask = 0xFFFFFFFFu;
    ray.maxDistance = FLT_MAX;
    return ray;
}

bool shouldEnableGPUShadingByDefault() {
    const char* envValue = std::getenv("RTR_MPS_GPU_SHADING");
    return parseEnvBool(envValue, true);
}

std::vector<MPSPackedFloat3> packVectorFloat3(const std::vector<vector_float3>& input) {
    std::vector<MPSPackedFloat3> packed;
    packed.reserve(input.size());
    for (const auto& value : input) {
        packed.push_back({value.x, value.y, value.z});
    }
    return packed;
}

id<MTLLibrary> loadMetalLibrary(id<MTLDevice> device) {
    if (!device) {
        return nil;
    }

    @autoreleasepool {
        id<MTLLibrary> library = [device newDefaultLibrary];
        if (library) {
            return library;
        }

        auto tryLoadFromPath = [&](const std::filesystem::path& candidate) -> id<MTLLibrary> {
            if (candidate.empty()) {
                return nil;
            }
            std::error_code ec;
            std::filesystem::path resolved = candidate;
            if (!resolved.is_absolute()) {
                resolved = std::filesystem::current_path(ec) / candidate;
            }
            if (!std::filesystem::exists(resolved, ec)) {
                return nil;
            }

            const std::string pathString = resolved.string();
            NSString* nsPath = [[NSString alloc] initWithUTF8String:pathString.c_str()];
            if (!nsPath) {
                return nil;
            }

            NSError* loadError = nil;
            id<MTLLibrary> loaded = [device newLibraryWithFile:nsPath error:&loadError];
            if (!loaded && loadError) {
                core::Logger::warn("MPSRenderer", "Failed to load Metal library at %s: %s", pathString.c_str(),
                                   loadError.localizedDescription.UTF8String);
            }
            return loaded;
        };

        const char* envPath = std::getenv("MTL_NEW_DEFAULT_LIBRARY_FILE");
        if (envPath && envPath[0] != '\0') {
            library = tryLoadFromPath(std::filesystem::path(envPath));
            if (library) {
                core::Logger::info("MPSRenderer", "Loaded Metal library from %s", envPath);
                return library;
            }
        }

        static const std::filesystem::path kFallbackLibraries[] = {
            "shaders/RTRShaders.metallib",
            "cmake-build-debug/shaders/RTRShaders.metallib",
            "cmake-build-release/shaders/RTRShaders.metallib",
            "build/shaders/RTRShaders.metallib",
        };

        for (const auto& candidate : kFallbackLibraries) {
            library = tryLoadFromPath(candidate);
            if (library) {
                core::Logger::info("MPSRenderer", "Loaded Metal library from %s", candidate.string().c_str());
                return library;
            }
        }

        core::Logger::warn("MPSRenderer", "Unable to locate Metal library for GPU shading");
        return nil;
    }
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
    : context_(context), bufferAllocator_(context), geometryStore_(bufferAllocator_),
      gpuShadingEnabled_(shouldEnableGPUShadingByDefault()) {
    if (!gpuShadingEnabled_) {
        core::Logger::info("MPSRenderer",
                           "GPU shading disabled via RTR_MPS_GPU_SHADING environment override");
    }
}

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

    const vector_float3 forward = simd_normalize(target - cameraOrigin);
    vector_float3 right = simd_cross(forward, worldUp);
    if (simd_length(right) < 1e-5f) {
        right = {1.0f, 0.0f, 0.0f};
    }
    const vector_float3 normRight = simd_normalize(right);
    const vector_float3 up = simd_normalize(simd_cross(normRight, forward));

    cameraUniforms_.eye = simd_make_float4(cameraOrigin, 1.0f);
    cameraUniforms_.forward = simd_make_float4(forward, 0.0f);
    cameraUniforms_.right = simd_make_float4(normRight, 0.0f);
    cameraUniforms_.up = simd_make_float4(up, 0.0f);
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
    if (!gpuShadingEnabled_) {
        return true;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_.rawDeviceHandle();
    if (!device) {
        core::Logger::warn("MPSRenderer", "Cannot create GPU pipelines: Metal device unavailable");
        gpuShadingEnabled_ = false;
        return false;
    }

    if (!gpuState_) {
        gpuState_ = std::make_unique<GPUState>();
    }

    NSError* error = nil;
    id<MTLLibrary> library = loadMetalLibrary(device);
    if (!library) {
        gpuShadingEnabled_ = false;
        return false;
    }

    if (!gpuState_->rayPipeline) {
        id<MTLFunction> rayFunction = [library newFunctionWithName:@"mpsRayKernel"];
        if (!rayFunction) {
            core::Logger::warn("MPSRenderer", "Metal library missing mpsRayKernel; GPU shading disabled");
            gpuShadingEnabled_ = false;
            return false;
        }

        gpuState_->rayPipeline = [device newComputePipelineStateWithFunction:rayFunction error:&error];
        if (error || !gpuState_->rayPipeline) {
            core::Logger::warn("MPSRenderer", "Failed to create ray compute pipeline state: %s",
                               error.localizedDescription.UTF8String);
            gpuState_.reset();
            gpuShadingEnabled_ = false;
            return false;
        }
        core::Logger::info("MPSRenderer", "GPU ray generation pipeline initialised");
    }

    if (!gpuState_->shadePipeline) {
        id<MTLFunction> shadeFunction = [library newFunctionWithName:@"mpsShadeKernel"];
        if (!shadeFunction) {
            core::Logger::warn("MPSRenderer", "Metal library missing mpsShadeKernel; GPU shading disabled");
            gpuShadingEnabled_ = false;
            return false;
        }

        gpuState_->shadePipeline = [device newComputePipelineStateWithFunction:shadeFunction error:&error];
        if (error || !gpuState_->shadePipeline) {
            core::Logger::warn("MPSRenderer", "Failed to create shade compute pipeline state: %s",
                               error.localizedDescription.UTF8String);
            gpuState_->shadePipeline = nil;
            gpuShadingEnabled_ = false;
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

    const auto packedPositions = packVectorFloat3(cpuScenePositions_);
    if (!packedPositions.empty()) {
        uploadStaticBuffer(gpuState_->positionsBuffer, packedPositions.data(),
                           packedPositions.size() * sizeof(MPSPackedFloat3), "mps.positions");
    }

    uploadStaticBuffer(gpuState_->indicesBuffer, cpuSceneIndices_.data(),
                       cpuSceneIndices_.size() * sizeof(uint32_t), "mps.indices");

    const auto packedColors = packVectorFloat3(cpuSceneColors_);
    if (!packedColors.empty()) {
        uploadStaticBuffer(gpuState_->colorsBuffer, packedColors.data(),
                           packedColors.size() * sizeof(MPSPackedFloat3), "mps.colors");
    }

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
    FrameComparison comparison;
    if (!computeFrame(comparison, true)) {
        return false;
    }

    if (comparison.cpuPixels.empty()) {
        core::Logger::error("MPSRenderer", "CPU shading produced no pixels");
        return false;
    }

    const std::vector<uint8_t>* finalPixels = &comparison.cpuPixels;
    if (!comparison.gpuPixels.empty() && comparison.maxByteDifference <= 2.0) {
        finalPixels = &comparison.gpuPixels;
    }

    if (!outputPath) {
        return true;
    }

    if (!writePPM(outputPath, *finalPixels, comparison.width, comparison.height)) {
        core::Logger::error("MPSRenderer", "Failed to write %s", outputPath);
        return false;
    }

    core::Logger::info("MPSRenderer", "Wrote diagnostic image to %s", outputPath);
    return true;
}

bool MPSRenderer::renderFrameComparison(const char* cpuOutputPath,
                                        const char* gpuOutputPath,
                                        FrameComparison* outComparison) {
    FrameComparison comparison;
    if (!computeFrame(comparison, true)) {
        return false;
    }

    if (cpuOutputPath) {
        if (comparison.cpuPixels.empty()) {
            core::Logger::error("MPSRenderer", "CPU shading produced no pixels");
            return false;
        }
        if (!writePPM(cpuOutputPath, comparison.cpuPixels, comparison.width, comparison.height)) {
            core::Logger::error("MPSRenderer", "Failed to write %s", cpuOutputPath);
            return false;
        }
        core::Logger::info("MPSRenderer", "Wrote diagnostic image to %s", cpuOutputPath);
    }

    if (gpuOutputPath) {
        if (comparison.gpuPixels.empty()) {
            core::Logger::warn("MPSRenderer", "GPU shading data unavailable; skipping %s", gpuOutputPath);
        } else if (!writePPM(gpuOutputPath, comparison.gpuPixels, comparison.width, comparison.height)) {
            core::Logger::error("MPSRenderer", "Failed to write %s", gpuOutputPath);
            return false;
        } else {
            core::Logger::info("MPSRenderer", "Wrote diagnostic image to %s", gpuOutputPath);
        }
    }

    if (outComparison) {
        *outComparison = std::move(comparison);
    }

    return true;
}

bool MPSRenderer::computeFrame(FrameComparison& comparison, bool logDifferences) {
    comparison = {};

    if (!pathTracer_.isValid()) {
        core::Logger::warn("MPSRenderer", "Path tracer not initialized; skipping frame");
        return false;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)pathTracer_.deviceHandle();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)pathTracer_.commandQueueHandle();
    MPSTriangleAccelerationStructure* accelerationStructure =
        (__bridge MPSTriangleAccelerationStructure*)pathTracer_.accelerationStructureHandle();
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
    id<MTLBuffer> intersectionBuffer =
        [device newBufferWithLength:pixelCount * sizeof(MPSIntersectionDistancePrimitiveIndexCoordinates)
                             options:MTLResourceStorageModeShared];

    if (!rayBuffer || !intersectionBuffer) {
        core::Logger::error("MPSRenderer", "Failed to allocate ray or intersection buffers");
        return false;
    }

    auto dispatchRayKernel = [&]() -> bool {
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
        [computeEncoder setBytes:&cameraUniforms_ length:sizeof(cameraUniforms_) atIndex:1];

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
        return true;
    };

    bool raysGeneratedWithGPU = false;
    if (gpuShadingEnabled_ && gpuState_ && gpuState_->rayPipeline) {
        raysGeneratedWithGPU = dispatchRayKernel();
    }

    if (!raysGeneratedWithGPU) {
        auto* rays = reinterpret_cast<MPSRayOriginMaskDirectionMaxDistance*>([rayBuffer contents]);
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                const std::size_t idx = static_cast<std::size_t>(y) * width + x;
                rays[idx] = makePrimaryRay(x, y, width, height, cameraUniforms_);
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

    auto* rayData = reinterpret_cast<MPSRayOriginMaskDirectionMaxDistance*>([rayBuffer contents]);

    const auto dumpRay = [&](const char* label, std::uint32_t sampleIndex,
                             const MPSRayOriginMaskDirectionMaxDistance& ray) {
        core::Logger::info("MPSRenderer", "%s %u origin=(%.3f, %.3f, %.3f) dir=(%.3f, %.3f, %.3f) mask=%u",
                           label, sampleIndex, ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x,
                           ray.direction.y, ray.direction.z, ray.mask);
    };

    if (rayData) {
        for (std::uint32_t sampleIndex = 0; sampleIndex < 4; ++sampleIndex) {
            dumpRay(raysGeneratedWithGPU ? "GPU ray" : "CPU ray", sampleIndex, rayData[sampleIndex]);
            if (raysGeneratedWithGPU) {
                const std::uint32_t sampleX = sampleIndex % width;
                const std::uint32_t sampleY = sampleIndex / width;
                const auto referenceRay = makePrimaryRay(sampleX, sampleY, width, height, cameraUniforms_);
                const vector_float3 diffDir = rayData[sampleIndex].direction - referenceRay.direction;
                const float dirError = simd_length(diffDir);
                const vector_float3 diffOrigin = rayData[sampleIndex].origin - referenceRay.origin;
                const float originError = simd_length(diffOrigin);
                if (originError > 1e-4f || dirError > 1e-4f) {
                    core::Logger::warn("MPSRenderer",
                                       "Ray mismatch at sample %u: originError=%.6f dirError=%.6f",
                                       sampleIndex, originError, dirError);
                }
            }
        }
    }

    if (gpuShadingEnabled_) {
        for (std::size_t sampleIndex = 0; sampleIndex < 4 && sampleIndex < pixelCount; ++sampleIndex) {
            const auto& isect = intersections[sampleIndex];
            core::Logger::info("MPSRenderer",
                               "GPU intersection %zu distance=%.3f primitive=%u bary=(%.3f, %.3f)", sampleIndex,
                               isect.distance, isect.primitiveIndex, isect.coordinates.x, isect.coordinates.y);
        }
    }

    std::vector<uint8_t> cpuPixels(pixelCount * 3);
    std::vector<vector_float3> cpuFloatPixels(pixelCount, (vector_float3){0.0f, 0.0f, 0.0f});

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
        const std::uint32_t x = i % width;
        const std::uint32_t y = i / width;
        if (x == 153 && y == 225) {
            core::Logger::info("DEBUG_CPU", "-- Pixel (153, 225) --");
            core::Logger::info("DEBUG_CPU", "hit=%d, primitive=%u, distance=%.3f", hit, intersection.primitiveIndex,
                               intersection.distance);
            if (hit) {
                const uint32_t primitiveIndex = intersection.primitiveIndex;
                const std::size_t base = static_cast<std::size_t>(primitiveIndex) * 3;
                const uint32_t i0 = indices[base + 0];
                const uint32_t i1 = indices[base + 1];
                const uint32_t i2 = indices[base + 2];
                const float u = intersection.coordinates.x;
                const float v = intersection.coordinates.y;
                const float w = 1.0f - u - v;
                const vector_float3& v0 = vertices[i0];
                const vector_float3& v1 = vertices[i1];
                const vector_float3& v2 = vertices[i2];
                vector_float3 e1 = v1 - v0;
                vector_float3 e2 = v2 - v0;
                vector_float3 normal = simd_normalize(simd_cross(e1, e2));
                float intensity = fmaxf(0.0f, simd_dot(normal, lightDir));
                intensity = intensity * 0.8f + 0.2f;
                vector_float3 c0 = (i0 < colors.size()) ? colors[i0] : vector_float3{0.85f, 0.4f, 0.25f};
                vector_float3 c1 = (i1 < colors.size()) ? colors[i1] : vector_float3{0.25f, 0.85f, 0.4f};
                vector_float3 c2 = (i2 < colors.size()) ? colors[i2] : vector_float3{0.4f, 0.25f, 0.85f};
                core::Logger::info("DEBUG_CPU", "bary=(%.3f, %.3f, %.3f)", u, v, w);
                core::Logger::info("DEBUG_CPU", "indices=(%u, %u, %u)", i0, i1, i2);
                core::Logger::info("DEBUG_CPU", "c0=(%.2f, %.2f, %.2f)", c0.x, c0.y, c0.z);
                core::Logger::info("DEBUG_CPU", "c1=(%.2f, %.2f, %.2f)", c1.x, c1.y, c1.z);
                core::Logger::info("DEBUG_CPU", "c2=(%.2f, %.2f, %.2f)", c2.x, c2.y, c2.z);
                core::Logger::info("DEBUG_CPU", "normal=(%.3f, %.3f, %.3f)", normal.x, normal.y, normal.z);
                core::Logger::info("DEBUG_CPU", "intensity=%.3f", intensity);
            }
        }

        color = simd_clamp(color, (vector_float3){0.0f, 0.0f, 0.0f}, (vector_float3){1.0f, 1.0f, 1.0f});
        cpuFloatPixels[i] = color;
        cpuPixels[i * 3 + 0] = floatToSRGBByte(color.x);
        cpuPixels[i * 3 + 1] = floatToSRGBByte(color.y);
        cpuPixels[i * 3 + 2] = floatToSRGBByte(color.z);
    }

    std::vector<uint8_t> gpuPixels;
    double maxByteDifference = 0.0;
    float maxFloatDifference = 0.0f;
    std::size_t worstComponentIndex = 0;
    vector_float3 worstCpuColour{0.0f, 0.0f, 0.0f};
    vector_float3 worstGpuColour{0.0f, 0.0f, 0.0f};
    bool usedGPUShading = false;

    if (gpuShadingEnabled_ && gpuState_ && gpuState_->shadePipeline && gpuState_->positionsBuffer.isValid() &&
        gpuState_->indicesBuffer.isValid() && gpuState_->colorsBuffer.isValid()) {
        const std::size_t outputByteLength = pixelCount * sizeof(simd_float4);
        if (!gpuState_->shadingOutputBuffer.isValid() || gpuState_->shadingOutputBuffer.length() < outputByteLength) {
            gpuState_->shadingOutputBuffer =
                bufferAllocator_.createBuffer(outputByteLength, nullptr, "mps.shadingOutput");
        }

        if (gpuState_->shadingOutputBuffer.isValid()) {
            id<MTLBuffer> debugBuffer = [device newBufferWithLength:sizeof(simd_float4) * 8 options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> shadeCommandBuffer = [queue commandBuffer];
            if (shadeCommandBuffer) {
                id<MTLComputeCommandEncoder> shadeEncoder = [shadeCommandBuffer computeCommandEncoder];
                if (shadeEncoder) {
                    [shadeEncoder setComputePipelineState:gpuState_->shadePipeline];
                    [shadeEncoder setBuffer:intersectionBuffer offset:0 atIndex:0];
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
                    [shadeEncoder setBytes:&cameraUniforms_ length:sizeof(cameraUniforms_) atIndex:5];

                    MPSSceneLimits limits{};
                    limits.vertexCount = static_cast<std::uint32_t>(
                        std::min<std::size_t>(cpuScenePositions_.size(), std::numeric_limits<std::uint32_t>::max()));
                    limits.indexCount = static_cast<std::uint32_t>(
                        std::min<std::size_t>(cpuSceneIndices_.size(), std::numeric_limits<std::uint32_t>::max()));
                    limits.colorCount = static_cast<std::uint32_t>(
                        std::min<std::size_t>(cpuSceneColors_.size(), std::numeric_limits<std::uint32_t>::max()));
                    limits.primitiveCount = limits.indexCount / 3U;
                    [shadeEncoder setBytes:&limits length:sizeof(limits) atIndex:6];
                    [shadeEncoder setBuffer:debugBuffer offset:0 atIndex:7];

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

                    auto* gpuDebug = reinterpret_cast<simd_float4*>([debugBuffer contents]);
                    if (gpuDebug) {
                        core::Logger::info("DEBUG_GPU", "-- Pixel (153, 225) --");
                        core::Logger::info("DEBUG_GPU", "bary=(%.3f, %.3f, %.3f)", gpuDebug[0].x, gpuDebug[0].y, gpuDebug[0].z);
                        core::Logger::info("DEBUG_GPU", "indices=(%u, %u, %u)", (uint)gpuDebug[1].x, (uint)gpuDebug[1].y, (uint)gpuDebug[1].z);
                        core::Logger::info("DEBUG_GPU", "c0=(%.2f, %.2f, %.2f)", gpuDebug[2].x, gpuDebug[2].y, gpuDebug[2].z);
                        core::Logger::info("DEBUG_GPU", "c1=(%.2f, %.2f, %.2f)", gpuDebug[3].x, gpuDebug[3].y, gpuDebug[3].z);
                        core::Logger::info("DEBUG_GPU", "c2=(%.2f, %.2f, %.2f)", gpuDebug[4].x, gpuDebug[4].y, gpuDebug[4].z);
                        core::Logger::info("DEBUG_GPU", "normal=(%.3f, %.3f, %.3f)", gpuDebug[5].x, gpuDebug[5].y, gpuDebug[5].z);
                        core::Logger::info("DEBUG_GPU", "intensity=%.3f", gpuDebug[6].x);
                        core::Logger::info("DEBUG_GPU", "interpolatedColor=(%.2f, %.2f, %.2f)", gpuDebug[7].x, gpuDebug[7].y, gpuDebug[7].z);
                    }

                    auto* gpuOutput = reinterpret_cast<simd_float4*>(
                        [(__bridge id<MTLBuffer>)gpuState_->shadingOutputBuffer.nativeHandle() contents]);
                    if (gpuOutput) {
                        usedGPUShading = true;
                        gpuPixels.resize(pixelCount * 3);
                        for (std::size_t i = 0; i < pixelCount; ++i) {
                            const vector_float3 cpuColour = cpuFloatPixels[i];
                            const vector_float3 gpuColour = {gpuOutput[i].x, gpuOutput[i].y, gpuOutput[i].z};

                            const float diffR = std::fabs(cpuColour.x - gpuColour.x);
                            const float diffG = std::fabs(cpuColour.y - gpuColour.y);
                            const float diffB = std::fabs(cpuColour.z - gpuColour.z);
                            maxFloatDifference = std::max({maxFloatDifference, diffR, diffG, diffB});

                            const uint8_t gpuR = floatToSRGBByte(gpuColour.x);
                            const uint8_t gpuG = floatToSRGBByte(gpuColour.y);
                            const uint8_t gpuB = floatToSRGBByte(gpuColour.z);

                            gpuPixels[i * 3 + 0] = gpuR;
                            gpuPixels[i * 3 + 1] = gpuG;
                            gpuPixels[i * 3 + 2] = gpuB;

                            if (i == ((std::size_t)225 * width + 153)) {
                                core::Logger::info("DEBUG_GPU", "quantised (%.3f, %.3f, %.3f) -> (%u, %u, %u)",
                                                   gpuColour.x, gpuColour.y, gpuColour.z,
                                                   static_cast<unsigned>(gpuR), static_cast<unsigned>(gpuG), static_cast<unsigned>(gpuB));
                            }

                            const uint8_t cpuR = floatToSRGBByte(cpuColour.x);
                            const uint8_t cpuG = floatToSRGBByte(cpuColour.y);
                            const uint8_t cpuB = floatToSRGBByte(cpuColour.z);

                            const std::array<double, 3> channelDiffs = {
                                std::fabs(static_cast<double>(cpuR) - gpuR),
                                std::fabs(static_cast<double>(cpuG) - gpuG),
                                std::fabs(static_cast<double>(cpuB) - gpuB),
                            };
                            for (std::size_t channel = 0; channel < channelDiffs.size(); ++channel) {
                                if (channelDiffs[channel] > maxByteDifference) {
                                    maxByteDifference = channelDiffs[channel];
                                    worstComponentIndex = i * 3 + channel;
                                    worstCpuColour = cpuColour;
                                    worstGpuColour = gpuColour;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (logDifferences && gpuShadingEnabled_ && !usedGPUShading) {
        core::Logger::warn("MPSRenderer", "GPU shading unavailable; retaining CPU output");
    }

    if (logDifferences && usedGPUShading) {
        if (maxByteDifference <= 2.0) {
            core::Logger::info("MPSRenderer",
                               "GPU shading matched CPU output (max byte diff %.2f, max float diff %.6f)",
                               maxByteDifference, static_cast<double>(maxFloatDifference));
        } else {
            const std::size_t worstPixelIndex = worstComponentIndex / 3;
            const std::size_t worstX = worstPixelIndex % width;
            const std::size_t worstY = worstPixelIndex / width;
            const std::size_t channel = worstComponentIndex % 3;
            const char channelName = (channel == 0) ? 'R' : (channel == 1 ? 'G' : 'B');
            const auto& cpuIntersection = intersections[worstPixelIndex];
            const float gpuComponentFloat = (channel == 0) ? worstGpuColour.x : (channel == 1 ? worstGpuColour.y : worstGpuColour.z);
            const uint8_t gpuComponentByte = floatToSRGBByte(gpuComponentFloat);
            const vector_float3 cpuColourFloat = worstCpuColour;
            core::Logger::warn("MPSRenderer",
                               "GPU shading diverged from CPU output (max diff %.2f, float diff %.6f) at pixel (%zu, %zu) channel %c: cpu=%u gpuByte=%u gpuByteFromFloat=%u gpuFloat=%.3f, cpuFloat=(%.2f, %.2f, %.2f), distance=%.3f primitive=%u bary=(%.3f, %.3f), gpuColour=(%.2f, %.2f, %.2f)",
                               maxByteDifference, static_cast<double>(maxFloatDifference), worstX, worstY, channelName,
                               static_cast<unsigned>(cpuPixels[worstComponentIndex]),
                               static_cast<unsigned>(gpuPixels[worstComponentIndex]),
                               static_cast<unsigned>(gpuComponentByte), gpuComponentFloat,
                               cpuColourFloat.x, cpuColourFloat.y, cpuColourFloat.z, cpuIntersection.distance,
                               cpuIntersection.primitiveIndex,
                               cpuIntersection.coordinates.x, cpuIntersection.coordinates.y,
                               worstGpuColour.x, worstGpuColour.y, worstGpuColour.z);
        }
    }

    comparison.cpuPixels = std::move(cpuPixels);
    comparison.gpuPixels = usedGPUShading ? std::move(gpuPixels) : std::vector<uint8_t>{};
    comparison.maxByteDifference = usedGPUShading ? maxByteDifference : 0.0;
    comparison.maxFloatDifference = usedGPUShading ? maxFloatDifference : 0.0f;
    comparison.width = width;
    comparison.height = height;

    core::Logger::info("MPSRenderer", "Intersection hits: %zu / %zu", hitCount, pixelCount);
    return true;
}

bool MPSRenderer::writePPM(const char* path,
                           const std::vector<uint8_t>& data,
                           std::uint32_t width,
                           std::uint32_t height) {
    if (!path) {
        return true;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(file);
}

bool MPSRenderer::usesGPUShading() const noexcept {
    return gpuShadingEnabled_ && gpuState_ && gpuState_->shadePipeline != nil &&
           gpuState_->positionsBuffer.isValid() && gpuState_->indicesBuffer.isValid() &&
           gpuState_->colorsBuffer.isValid();
}
}  // namespace rtr::rendering

#pragma clang diagnostic pop
