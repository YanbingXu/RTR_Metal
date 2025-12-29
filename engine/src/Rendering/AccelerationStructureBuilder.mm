#import <Foundation/Foundation.h>
#ifndef MTL_ENABLE_RAYTRACING
#define MTL_ENABLE_RAYTRACING 1
#endif
#import <Metal/Metal.h>
#if __has_include(<Metal/MetalRayTracing.h>)
#import <Metal/MetalRayTracing.h>
#endif

#include "RTRMetalEngine/Rendering/AccelerationStructureBuilder.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"

#include <algorithm>
#include <cstring>
#include <simd/simd.h>
#include <vector>

namespace {

constexpr std::size_t kMaxTrianglesPerGeometry = 4096;

MTLPackedFloat3 makePackedColumn(const simd_float4& column) {
    return MTLPackedFloat3Make(column.x, column.y, column.z);
}

MTLPackedFloat4x3 makePackedTransform(const simd_float4x4& transform) {
    return MTLPackedFloat4x3(makePackedColumn(transform.columns[0]),
                             makePackedColumn(transform.columns[1]),
                             makePackedColumn(transform.columns[2]),
                             makePackedColumn(transform.columns[3]));
}

NSMutableArray<MTLAccelerationStructureTriangleGeometryDescriptor*>*
makeTriangleDescriptors(const rtr::rendering::MeshBuffers& meshBuffers,
                        id<MTLBuffer> vertexBuffer,
                        id<MTLBuffer> indexBuffer,
                        const std::string& label) {
    const NSUInteger totalTriangles = static_cast<NSUInteger>(meshBuffers.indexCount / 3);
    if (totalTriangles == 0) {
        rtr::core::Logger::warn("ASBuilder", "Mesh '%s' has no triangles", label.c_str());
        return nil;
    }

    NSMutableArray<MTLAccelerationStructureTriangleGeometryDescriptor*>* geometries =
        [NSMutableArray array];
    NSUInteger processed = 0;
    NSUInteger indexOffsetBytes = 0;
    while (processed < totalTriangles) {
        const NSUInteger remaining = totalTriangles - processed;
        const NSUInteger chunkTriangles = std::min<NSUInteger>(kMaxTrianglesPerGeometry, remaining);
        MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geometry.vertexBuffer = vertexBuffer;
        geometry.vertexStride = sizeof(float) * 3;
        geometry.vertexBufferOffset = 0;
#if defined(__MAC_13_0) || defined(__IPHONE_16_0)
        geometry.vertexFormat = MTLAttributeFormatFloat3;
#else
        geometry.vertexFormat = MTLVertexFormatFloat3;
#endif
        geometry.indexBuffer = indexBuffer;
        geometry.indexBufferOffset = indexOffsetBytes;
        geometry.indexType = MTLIndexTypeUInt32;
        geometry.triangleCount = chunkTriangles;
        geometry.opaque = YES;
        [geometries addObject:geometry];
        processed += chunkTriangles;
        indexOffsetBytes += chunkTriangles * 3 * sizeof(std::uint32_t);
    }

    rtr::core::Logger::info("ASBuilder",
                            "Mesh '%s' chunked into %lu geometry descriptors (total tris=%lu)",
                            label.c_str(),
                            static_cast<unsigned long>(geometries.count),
                            static_cast<unsigned long>(totalTriangles));
    return geometries;
}

bool populateInstanceDescriptors(std::span<const rtr::rendering::InstanceBuildInput> instances,
                                 NSMutableArray<id<MTLAccelerationStructure>>* blasArray,
                                 std::vector<MTLAccelerationStructureUserIDInstanceDescriptor>& descriptors,
                                 const std::string& label) {
    descriptors.resize(instances.size());
    for (std::size_t i = 0; i < instances.size(); ++i) {
        const auto& instance = instances[i];
        if (!instance.structure || !instance.structure->isValid()) {
            rtr::core::Logger::error("ASBuilder", "Invalid BLAS handle while preparing TLAS '%s'",
                                     label.c_str());
            return false;
        }

        id<MTLAccelerationStructure> blas =
            (__bridge id<MTLAccelerationStructure>)instance.structure->rawHandle();
        if (!blas) {
            rtr::core::Logger::error("ASBuilder",
                                     "Missing native BLAS handle while preparing TLAS '%s'",
                                     label.c_str());
            return false;
        }
        [blasArray addObject:blas];

        auto& descriptor = descriptors[i];
        descriptor.transformationMatrix = makePackedTransform(instance.transform);
        descriptor.options = MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        descriptor.mask = instance.mask;
        descriptor.intersectionFunctionTableOffset = instance.intersectionFunctionTableOffset;
        descriptor.accelerationStructureIndex = static_cast<uint32_t>(i);
        descriptor.userID = instance.userID;
    }
    return true;
}

}  // namespace

namespace rtr::rendering {

AccelerationStructureBuilder::AccelerationStructureBuilder(MetalContext& context)
    : context_(context) {}

bool AccelerationStructureBuilder::isRayTracingSupported() const noexcept {
    return context_.supportsRayTracing();
}

std::optional<BottomLevelBuildInfo> AccelerationStructureBuilder::queryBottomLevelSizes(
    const MeshBuffers& meshBuffers, const std::string& label) const {
    if (!context_.isValid()) {
        core::Logger::warn("ASBuilder", "Metal device unavailable; skipping BLAS sizing for '%s'", label.c_str());
        return std::nullopt;
    }

    if (!context_.supportsRayTracing()) {
        core::Logger::warn("ASBuilder", "Ray tracing unsupported on device; skipping BLAS sizing for '%s'",
                           label.c_str());
        return std::nullopt;
    }

    if (!meshBuffers.gpuVertexBuffer.isValid() || !meshBuffers.gpuIndexBuffer.isValid()) {
        core::Logger::error("ASBuilder", "Mesh buffers invalid for '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_.rawDeviceHandle();
    if (!device) {
        core::Logger::error("ASBuilder", "Failed to acquire Metal device for '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLBuffer> vertexBuffer = (__bridge id<MTLBuffer>)meshBuffers.gpuVertexBuffer.nativeHandle();
    id<MTLBuffer> indexBuffer = (__bridge id<MTLBuffer>)meshBuffers.gpuIndexBuffer.nativeHandle();
    if (!vertexBuffer || !indexBuffer) {
        core::Logger::error("ASBuilder", "Mesh buffers missing native handles for '%s'", label.c_str());
        return std::nullopt;
    }

    const NSUInteger indexCount = static_cast<NSUInteger>(meshBuffers.indexCount);
    if (indexCount < 3) {
        core::Logger::warn("ASBuilder", "Mesh '%s' has insufficient indices to form triangles", label.c_str());
        return std::nullopt;
    }

    NSMutableArray<MTLAccelerationStructureTriangleGeometryDescriptor*>* geometries =
        makeTriangleDescriptors(meshBuffers, vertexBuffer, indexBuffer, label);
    if (!geometries || geometries.count == 0) {
        return std::nullopt;
    }

    MTLPrimitiveAccelerationStructureDescriptor* descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = geometries;

    if (!descriptor) {
        core::Logger::error("ASBuilder", "Failed to create descriptor for '%s'", label.c_str());
        return std::nullopt;
    }

    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:descriptor];

    BottomLevelBuildInfo info{};
    info.accelerationStructureSize = static_cast<std::size_t>(sizes.accelerationStructureSize);
    info.buildScratchSize = static_cast<std::size_t>(sizes.buildScratchBufferSize);
    info.updateScratchSize = static_cast<std::size_t>(sizes.refitScratchBufferSize);

    return info;
}

std::optional<AccelerationStructure> AccelerationStructureBuilder::buildBottomLevel(
    const MeshBuffers& meshBuffers,
    const std::string& label,
    void* commandQueueHandle) const {
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)commandQueueHandle;
    if (!commandQueue) {
        core::Logger::error("ASBuilder", "Command queue unavailable for '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (!commandBuffer) {
        core::Logger::error("ASBuilder", "Failed to create command buffer for '%s'", label.c_str());
        return std::nullopt;
    }

    const auto sizes = queryBottomLevelSizes(meshBuffers, label);
    if (!sizes.has_value()) {
        return std::nullopt;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_.rawDeviceHandle();
    if (!device) {
        core::Logger::error("ASBuilder", "Metal device unavailable for '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLAccelerationStructure> accelerationStructure =
        [device newAccelerationStructureWithSize:sizes->accelerationStructureSize];
    if (!accelerationStructure) {
        core::Logger::error("ASBuilder", "Failed to allocate acceleration structure for '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLBuffer> scratchBuffer =
        [device newBufferWithLength:sizes->buildScratchSize options:MTLResourceStorageModePrivate];
    if (!scratchBuffer) {
        core::Logger::error("ASBuilder", "Failed to allocate scratch buffer for '%s'", label.c_str());
        return std::nullopt;
    }

    NSMutableArray<MTLAccelerationStructureTriangleGeometryDescriptor*>* geometries =
        makeTriangleDescriptors(meshBuffers,
                                (__bridge id<MTLBuffer>)meshBuffers.gpuVertexBuffer.nativeHandle(),
                                (__bridge id<MTLBuffer>)meshBuffers.gpuIndexBuffer.nativeHandle(),
                                label);
    if (!geometries || geometries.count == 0) {
        core::Logger::error("ASBuilder", "Failed to prepare geometry descriptors for '%s'", label.c_str());
        return std::nullopt;
    }
    MTLAccelerationStructureTriangleGeometryDescriptor* logGeometry = geometries.firstObject;
    core::Logger::info("ASBuilder",
                       "BLAS geometry chunks=%lu first tris=%lu stride=%lu vertexBuffer=%p indexBuffer=%p",
                       static_cast<unsigned long>(geometries.count),
                       static_cast<unsigned long>(logGeometry.triangleCount),
                       static_cast<unsigned long>(logGeometry.vertexStride),
                       logGeometry.vertexBuffer,
                       logGeometry.indexBuffer);

    MTLPrimitiveAccelerationStructureDescriptor* descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = geometries;

    id<MTLAccelerationStructureCommandEncoder> encoder = [commandBuffer accelerationStructureCommandEncoder];
    if (!encoder) {
        core::Logger::error("ASBuilder", "Failed to create AS encoder for '%s'", label.c_str());
        return std::nullopt;
    }

    [encoder buildAccelerationStructure:accelerationStructure
                       descriptor:descriptor
                    scratchBuffer:scratchBuffer
              scratchBufferOffset:0];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    const MTLCommandBufferStatus status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
        NSString* errorString = commandBuffer.error.localizedDescription;
        const char* message = errorString ? errorString.UTF8String : "unknown error";
        core::Logger::error("ASBuilder",
                            "BLAS command buffer status=%ld error=%s for '%s'",
                            static_cast<long>(status),
                            message,
                            label.c_str());
        return std::nullopt;
    }

    return makeAccelerationStructure(label,
                                     sizes->accelerationStructureSize,
                                     (__bridge_retained void*)accelerationStructure);
}

std::optional<TopLevelBuildInfo> AccelerationStructureBuilder::queryTopLevelSizes(
    std::span<const InstanceBuildInput> instances,
    const std::string& label) const {
    if (!context_.isValid()) {
        core::Logger::warn("ASBuilder", "Metal device unavailable; skipping TLAS sizing for '%s'", label.c_str());
        return std::nullopt;
    }

    if (!context_.supportsRayTracing()) {
        core::Logger::warn("ASBuilder", "Ray tracing unsupported on device; skipping TLAS sizing for '%s'",
                           label.c_str());
        return std::nullopt;
    }

    if (instances.empty()) {
        core::Logger::warn("ASBuilder", "No instances provided for TLAS sizing of '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_.rawDeviceHandle();
    if (!device) {
        core::Logger::error("ASBuilder", "Failed to acquire Metal device for '%s'", label.c_str());
        return std::nullopt;
    }

    const std::size_t descriptorStride = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
    const std::size_t descriptorBufferSize = descriptorStride * instances.size();
    if (descriptorBufferSize == 0) {
        core::Logger::warn("ASBuilder", "Descriptor stride invalid while sizing TLAS '%s'", label.c_str());
        return std::nullopt;
    }

    NSMutableArray<id<MTLAccelerationStructure>>* blasArray =
        [NSMutableArray arrayWithCapacity:instances.size()];
    std::vector<MTLAccelerationStructureUserIDInstanceDescriptor> cpuDescriptors;
    if (!populateInstanceDescriptors(instances, blasArray, cpuDescriptors, label)) {
        return std::nullopt;
    }

    MTLInstanceAccelerationStructureDescriptor* descriptor =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    descriptor.instanceCount = instances.size();
    descriptor.instancedAccelerationStructures = blasArray;
    descriptor.instanceDescriptorBuffer = nil;
    descriptor.instanceDescriptorBufferOffset = 0;
    descriptor.instanceDescriptorStride = descriptorStride;
    descriptor.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;

    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:descriptor];

    TopLevelBuildInfo info{};
    info.accelerationStructureSize = static_cast<std::size_t>(sizes.accelerationStructureSize);
    info.buildScratchSize = static_cast<std::size_t>(sizes.buildScratchBufferSize);
    info.updateScratchSize = static_cast<std::size_t>(sizes.refitScratchBufferSize);
    info.instanceDescriptorBufferSize = descriptorBufferSize;

    return info;
}

std::optional<AccelerationStructure> AccelerationStructureBuilder::buildTopLevel(
    std::span<const InstanceBuildInput> instances,
    const std::string& label,
    void* commandQueueHandle) const {
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)commandQueueHandle;
    if (!commandQueue) {
        core::Logger::error("ASBuilder", "Command queue unavailable for TLAS '%s'", label.c_str());
        return std::nullopt;
    }

    const auto sizes = queryTopLevelSizes(instances, label);
    if (!sizes.has_value()) {
        return std::nullopt;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_.rawDeviceHandle();
    if (!device) {
        core::Logger::error("ASBuilder", "Metal device unavailable for TLAS '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLAccelerationStructure> accelerationStructure =
        [device newAccelerationStructureWithSize:sizes->accelerationStructureSize];
    if (!accelerationStructure) {
        core::Logger::error("ASBuilder", "Failed to allocate TLAS '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLBuffer> scratchBuffer =
        [device newBufferWithLength:sizes->buildScratchSize options:MTLResourceStorageModePrivate];
    if (!scratchBuffer) {
        core::Logger::error("ASBuilder", "Failed to allocate TLAS scratch buffer for '%s'", label.c_str());
        return std::nullopt;
    }

    NSMutableArray<id<MTLAccelerationStructure>>* blasArray =
        [NSMutableArray arrayWithCapacity:instances.size()];
    std::vector<MTLAccelerationStructureUserIDInstanceDescriptor> cpuDescriptors;
    if (!populateInstanceDescriptors(instances, blasArray, cpuDescriptors, label)) {
        return std::nullopt;
    }

    if (cpuDescriptors.size() * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor) !=
        sizes->instanceDescriptorBufferSize) {
        core::Logger::warn("ASBuilder",
                           "Descriptor size mismatch while building TLAS '%s'", label.c_str());
    }

    id<MTLBuffer> instanceBuffer =
        [device newBufferWithBytes:cpuDescriptors.data()
                             length:cpuDescriptors.size() * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor)
                            options:MTLResourceStorageModeShared];
    if (!instanceBuffer) {
        core::Logger::error("ASBuilder", "Failed to upload TLAS descriptors for '%s'", label.c_str());
        return std::nullopt;
    }
    if ([instanceBuffer storageMode] == MTLStorageModeManaged) {
        [instanceBuffer didModifyRange:
                           NSMakeRange(0, cpuDescriptors.size() * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor))];
    }

    for (std::size_t i = 0; i < cpuDescriptors.size() && i < 8; ++i) {
        const auto& descriptor = cpuDescriptors[i];
        const MTLPackedFloat3 t0 = descriptor.transformationMatrix.columns[0];
        const MTLPackedFloat3 t1 = descriptor.transformationMatrix.columns[1];
        const MTLPackedFloat3 t2 = descriptor.transformationMatrix.columns[2];
        const MTLPackedFloat3 t3 = descriptor.transformationMatrix.columns[3];
        core::Logger::info("ASBuilder",
                           "Instance[%zu]: mask=0x%02X accelIndex=%u options=0x%X",
                           i,
                           descriptor.mask,
                           descriptor.accelerationStructureIndex,
                           descriptor.options);
        core::Logger::info("ASBuilder",
                           "TLAS instance[%zu]: mask=0x%02X options=0x%X translate=(%.3f, %.3f, %.3f)",
                           i,
                           descriptor.mask,
                           descriptor.options,
                           static_cast<double>(t3.x),
                           static_cast<double>(t3.y),
                           static_cast<double>(t3.z));
        core::Logger::info("ASBuilder",
                           "Transform rows: [%.3f %.3f %.3f] [%.3f %.3f %.3f] [%.3f %.3f %.3f] [%.3f %.3f %.3f]",
                           static_cast<double>(t0.x),
                           static_cast<double>(t0.y),
                           static_cast<double>(t0.z),
                           static_cast<double>(t1.x),
                           static_cast<double>(t1.y),
                           static_cast<double>(t1.z),
                           static_cast<double>(t2.x),
                           static_cast<double>(t2.y),
                           static_cast<double>(t2.z),
                           static_cast<double>(t3.x),
                           static_cast<double>(t3.y),
                           static_cast<double>(t3.z));
    }

    MTLInstanceAccelerationStructureDescriptor* descriptor =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    descriptor.instanceCount = instances.size();
    descriptor.instancedAccelerationStructures = blasArray;
    descriptor.instanceDescriptorBuffer = instanceBuffer;
    descriptor.instanceDescriptorBufferOffset = 0;
    descriptor.instanceDescriptorStride = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
    descriptor.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (!commandBuffer) {
        core::Logger::error("ASBuilder", "Failed to create command buffer for TLAS '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLAccelerationStructureCommandEncoder> encoder = [commandBuffer accelerationStructureCommandEncoder];
    if (!encoder) {
        core::Logger::error("ASBuilder", "Failed to create AS encoder for TLAS '%s'", label.c_str());
        return std::nullopt;
    }

    [encoder buildAccelerationStructure:accelerationStructure
                       descriptor:descriptor
                    scratchBuffer:scratchBuffer
              scratchBufferOffset:0];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    const MTLCommandBufferStatus status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
        NSString* errorString = commandBuffer.error.localizedDescription;
        const char* message = errorString ? errorString.UTF8String : "unknown error";
        core::Logger::error("ASBuilder",
                            "TLAS command buffer status=%ld error=%s for '%s'",
                            static_cast<long>(status),
                            message,
                            label.c_str());
        return std::nullopt;
    }

    return makeAccelerationStructure(label,
                                     sizes->accelerationStructureSize,
                                     (__bridge_retained void*)accelerationStructure);
}

}  // namespace rtr::rendering
