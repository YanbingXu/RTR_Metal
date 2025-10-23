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

bool writePPM(const char* path, const std::vector<uint8_t>& data, std::uint32_t width, std::uint32_t height) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(file);
}

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

bool shouldEnableGPUShadingByDefault() {
    const char* envValue = std::getenv("RTR_MPS_GPU_SHADING");
    return parseEnvBool(envValue, true);
}

std::vector<MPSPackedFloat3> packVectorFloat3(const std::vector<vector_float3>& input) {
    std::vector<MPSPackedFloat3> packed;
    packed.reserve(input.size());
    for (const auto& value : input) {
        packed.emplace_back(value);
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

    if (gpuShadingEnabled_ && gpuState_ && gpuState_->rayPipeline) {
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

        auto* rayDebug = reinterpret_cast<MPSRayOriginMaskDirectionMaxDistance*>([rayBuffer contents]);
        if (rayDebug) {
            for (std::uint32_t sampleIndex = 0; sampleIndex < 4; ++sampleIndex) {
                const auto& ray = rayDebug[sampleIndex];
                core::Logger::info("MPSRenderer", "Post-ray-kernel %u origin=(%.3f, %.3f, %.3f) dir=(%.3f, %.3f, %.3f) mask=%u",
                                   sampleIndex, ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y,
                                   ray.direction.z, ray.mask);
            }
        }
    } else {
        auto* rays = reinterpret_cast<MPSRayOriginMaskDirectionMaxDistance*>([rayBuffer contents]);
        const vector_float3 cameraOrigin = simd_make_float3(cameraUniforms_.eye);
        const vector_float3 right = simd_make_float3(cameraUniforms_.right);
        const vector_float3 up = simd_make_float3(cameraUniforms_.up);
        const vector_float3 forward = simd_make_float3(cameraUniforms_.forward);
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

    if (gpuShadingEnabled_) {
        auto* rayData = reinterpret_cast<MPSRayOriginMaskDirectionMaxDistance*>([rayBuffer contents]);
        if (rayData) {
            for (std::uint32_t sampleIndex = 0; sampleIndex < 4; ++sampleIndex) {
                const auto& ray = rayData[sampleIndex];
                core::Logger::info("MPSRenderer", "GPU ray %u origin=(%.3f, %.3f, %.3f) dir=(%.3f, %.3f, %.3f) mask=%u",
                                   sampleIndex, ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y,
                                   ray.direction.z, ray.mask);
            }
        }

        for (std::size_t sampleIndex = 0; sampleIndex < 4 && sampleIndex < pixelCount; ++sampleIndex) {
            const auto& isect = intersections[sampleIndex];
            core::Logger::info("MPSRenderer",
                               "GPU intersection %zu distance=%.3f primitive=%u bary=(%.3f, %.3f)", sampleIndex,
                               isect.distance, isect.primitiveIndex, isect.coordinates.x, isect.coordinates.y);
        }
    }

    std::vector<uint8_t> pixels(pixelCount * 3);
    std::vector<vector_float3> cpuFloatPixels(pixelCount, (vector_float3){0.0f, 0.0f, 0.0f});
    std::vector<uint8_t> gpuPixels;
    simd_float4* shadeOutputPtr = nullptr;
    bool usedGPUShading = false;

    if (gpuShadingEnabled_ && gpuState_ && gpuState_->shadePipeline && gpuState_->positionsBuffer.isValid() &&
        gpuState_->indicesBuffer.isValid() && gpuState_->colorsBuffer.isValid()) {
        const std::size_t outputByteLength = pixelCount * sizeof(simd_float4);
        if (!gpuState_->shadingOutputBuffer.isValid() || gpuState_->shadingOutputBuffer.length() < outputByteLength) {
            gpuState_->shadingOutputBuffer =
                bufferAllocator_.createBuffer(outputByteLength, nullptr, "mps.shadingOutput");
        }

        if (gpuState_->shadingOutputBuffer.isValid()) {
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
                    limits.vertexCount = static_cast<std::uint32_t>(std::min<std::size_t>(cpuScenePositions_.size(),
                                                                                          std::numeric_limits<std::uint32_t>::max()));
                    limits.indexCount = static_cast<std::uint32_t>(std::min<std::size_t>(cpuSceneIndices_.size(),
                                                                                       std::numeric_limits<std::uint32_t>::max()));
                    limits.colorCount = static_cast<std::uint32_t>(std::min<std::size_t>(cpuSceneColors_.size(),
                                                                                       std::numeric_limits<std::uint32_t>::max()));
                    limits.primitiveCount = limits.indexCount / 3U;
                    [shadeEncoder setBytes:&limits length:sizeof(limits) atIndex:6];

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
                            gpuPixels[i * 3 + 0] = floatToSRGBByte(c.x);
                            gpuPixels[i * 3 + 1] = floatToSRGBByte(c.y);
                            gpuPixels[i * 3 + 2] = floatToSRGBByte(c.z);
                        }
                        shadeOutputPtr = gpuOutput;
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
        cpuFloatPixels[i] = color;
        pixels[i * 3 + 0] = floatToSRGBByte(color.x);
        pixels[i * 3 + 1] = floatToSRGBByte(color.y);
        pixels[i * 3 + 2] = floatToSRGBByte(color.z);
    }

    if (gpuShadingEnabled_ && usedGPUShading && gpuPixels.size() == pixels.size()) {
        float maxDifference = 0.0f;
        std::size_t worstComponentIndex = 0;
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const float diff = std::fabs(static_cast<float>(pixels[i]) - gpuPixels[i]);
            if (diff > maxDifference) {
                maxDifference = diff;
                worstComponentIndex = i;
            }
        }

        if (maxDifference <= 2.0f) {
            pixels = std::move(gpuPixels);
            core::Logger::info("MPSRenderer", "GPU shading matched CPU output (max diff %.2f)", maxDifference);
        } else {
            const std::size_t worstPixelIndex = worstComponentIndex / 3;
            const std::size_t worstX = worstPixelIndex % width;
            const std::size_t worstY = worstPixelIndex / width;
            const std::size_t channel = worstComponentIndex % 3;
            const char channelName = (channel == 0) ? 'R' : (channel == 1 ? 'G' : 'B');
            const auto& cpuIntersection = intersections[worstPixelIndex];
            const simd_float4 gpuColour = (shadeOutputPtr && worstPixelIndex < pixelCount)
                                              ? shadeOutputPtr[worstPixelIndex]
                                              : (simd_float4){0.0f, 0.0f, 0.0f, 0.0f};
            const float gpuComponentFloat = (channel == 0) ? gpuColour.x : (channel == 1 ? gpuColour.y : gpuColour.z);
            const uint8_t gpuComponentByte = static_cast<uint8_t>(
                std::clamp(gpuComponentFloat * 255.0f, 0.0f, 255.0f));
            const vector_float3 cpuColourFloat = (worstPixelIndex < cpuFloatPixels.size()) ? cpuFloatPixels[worstPixelIndex]
                                                                                          : (vector_float3){0.0f, 0.0f, 0.0f};
            core::Logger::warn("MPSRenderer",
                               "GPU shading diverged from CPU output (max diff %.2f) at pixel (%zu, %zu) channel %c: cpu=%u gpuByte=%u gpuFloat=%.3f, cpuFloat=(%.2f, %.2f, %.2f), distance=%.3f primitive=%u bary=(%.3f, %.3f), gpuColour=(%.2f, %.2f, %.2f)",
                               maxDifference, worstX, worstY, channelName,
                               static_cast<unsigned>(pixels[worstComponentIndex]),
                               static_cast<unsigned>(gpuPixels[worstComponentIndex]), gpuComponentFloat,
                               cpuColourFloat.x, cpuColourFloat.y, cpuColourFloat.z, cpuIntersection.distance,
                               cpuIntersection.primitiveIndex,
                               cpuIntersection.coordinates.x, cpuIntersection.coordinates.y, gpuColour.x,
                               gpuColour.y, gpuColour.z);
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
