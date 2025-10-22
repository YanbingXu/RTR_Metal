#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "RTRMetalEngine/Rendering/AccelerationStructureBuilder.hpp"

#include "RTRMetalEngine/Core/Logger.hpp"

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

    if (!meshBuffers.vertexBuffer.isValid() || !meshBuffers.indexBuffer.isValid()) {
        core::Logger::error("ASBuilder", "Mesh buffers invalid for '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context_.rawDeviceHandle();
    if (!device) {
        core::Logger::error("ASBuilder", "Failed to acquire Metal device for '%s'", label.c_str());
        return std::nullopt;
    }

    id<MTLBuffer> vertexBuffer = (__bridge id<MTLBuffer>)meshBuffers.vertexBuffer.nativeHandle();
    id<MTLBuffer> indexBuffer = (__bridge id<MTLBuffer>)meshBuffers.indexBuffer.nativeHandle();
    if (!vertexBuffer || !indexBuffer) {
        core::Logger::error("ASBuilder", "Mesh buffers missing native handles for '%s'", label.c_str());
        return std::nullopt;
    }

    const NSUInteger indexCount = static_cast<NSUInteger>(meshBuffers.indexCount);
    if (indexCount < 3) {
        core::Logger::warn("ASBuilder", "Mesh '%s' has insufficient indices to form triangles", label.c_str());
        return std::nullopt;
    }

    MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geometry.vertexBuffer = vertexBuffer;
    geometry.vertexStride = static_cast<NSUInteger>(meshBuffers.vertexStride);
#if defined(__MAC_13_0) || defined(__IPHONE_16_0)
    geometry.vertexFormat = MTLAttributeFormatFloat3;
#else
    geometry.vertexFormat = MTLVertexFormatFloat3;
#endif
    geometry.indexBuffer = indexBuffer;
    geometry.indexType = MTLIndexTypeUInt32;
    geometry.triangleCount = indexCount / 3;
    geometry.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor* descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = @[ geometry ];

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

    MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geometry.vertexBuffer = (__bridge id<MTLBuffer>)meshBuffers.vertexBuffer.nativeHandle();
    geometry.vertexStride = static_cast<NSUInteger>(meshBuffers.vertexStride);
#if defined(__MAC_13_0) || defined(__IPHONE_16_0)
    geometry.vertexFormat = MTLAttributeFormatFloat3;
#else
    geometry.vertexFormat = MTLVertexFormatFloat3;
#endif
    geometry.indexBuffer = (__bridge id<MTLBuffer>)meshBuffers.indexBuffer.nativeHandle();
    geometry.indexType = MTLIndexTypeUInt32;
    geometry.triangleCount = static_cast<NSUInteger>(meshBuffers.indexCount / 3);
    geometry.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor* descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = @[ geometry ];

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

    return makeAccelerationStructure(label, sizes->accelerationStructureSize, (__bridge void*)accelerationStructure);
}

}  // namespace rtr::rendering
