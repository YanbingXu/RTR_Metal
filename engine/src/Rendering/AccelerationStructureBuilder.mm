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

}  // namespace rtr::rendering
