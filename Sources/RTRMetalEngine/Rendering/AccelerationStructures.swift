import Foundation
import Metal
import simd

struct GeometryGPUResources {
    let vertexBuffer: MTLBuffer
    let normalBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let blas: MTLAccelerationStructure
    let primitiveCount: Int
}

struct InstanceGPUResources {
    var transform: simd_float4x4
    var materialIndex: UInt32
    var geometry: GeometryGPUResources
}

struct GeometryInfoGPU {
    var vertexBufferAddress: UInt64
    var normalBufferAddress: UInt64
    var indexBufferAddress: UInt64
    var materialIndex: UInt32
    var primitiveCount: UInt32
    var padding: UInt32
}

struct InstanceInfoGPU {
    var transform: simd_float4x4
    var normalMatrix: simd_float3x3
    var materialIndex: UInt32
    var padding: simd_float3 = .zero
}

public final class AccelerationStructureBuilder {
    private let context: MetalContext
    private var geometryCache: [Mesh: GeometryGPUResources] = [:]

    init(context: MetalContext) {
        self.context = context
    }

    func buildScene(_ scene: Scene) throws -> (MTLAccelerationStructure, [InstanceGPUResources], MTLBuffer, MTLBuffer, MTLBuffer) {
        var instances: [InstanceGPUResources] = []
        var materials: [Material] = []
        for instance in scene.instances {
            let geometry = try geometryResources(for: instance.mesh)
            let transform = instance.transform
            let materialIndex = UInt32(materials.count)
            materials.append(instance.material)
            instances.append(InstanceGPUResources(transform: transform, materialIndex: materialIndex, geometry: geometry))
        }
        let materialBuffer = try makeMaterialBuffer(materials: materials)
        let geometryInfoBuffer = try makeGeometryInfoBuffer(instances: instances)
        let instanceInfoBuffer = try makeInstanceInfoBuffer(instances: instances)
        let tlas = try buildTLAS(instances: instances)
        return (tlas, instances, materialBuffer, geometryInfoBuffer, instanceInfoBuffer)
    }

    private func geometryResources(for mesh: Mesh) throws -> GeometryGPUResources {
        if let cached = geometryCache[mesh] {
            return cached
        }
        guard let vertexBuffer = context.device.makeBuffer(bytes: mesh.positions, length: MemoryLayout<simd_float3>.stride * mesh.positions.count, options: .storageModeManaged),
              let normalBuffer = context.device.makeBuffer(bytes: mesh.normals, length: MemoryLayout<simd_float3>.stride * mesh.normals.count, options: .storageModeManaged),
              let indexBuffer = context.device.makeBuffer(bytes: mesh.indices, length: MemoryLayout<UInt32>.stride * mesh.indices.count, options: .storageModeManaged)
        else {
            throw RendererError.resourceCreationFailed("Failed to allocate mesh buffers for \(mesh.name)")
        }

        let geometryDescriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
        geometryDescriptor.vertexBuffer = vertexBuffer
        geometryDescriptor.vertexStride = MemoryLayout<simd_float3>.stride
        geometryDescriptor.indexBuffer = indexBuffer
        geometryDescriptor.indexType = .uint32
        geometryDescriptor.triangleCount = mesh.indices.count / 3
        geometryDescriptor.intersectionFunctionTableOffset = 0

        let blasDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
        blasDescriptor.geometryDescriptors = [geometryDescriptor]
        blasDescriptor.usage = .preferFastBuild

        let sizes = context.device.accelerationStructureSizes(descriptor: blasDescriptor)
        guard let scratch = context.device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate),
              let blas = context.device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
        else {
            throw RendererError.resourceCreationFailed("Failed to allocate BLAS buffers for \(mesh.name)")
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else {
            throw RendererError.commandEncodingFailed("Failed to encode BLAS build for \(mesh.name)")
        }
        encoder.build(accelerationStructure: blas, descriptor: blasDescriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resources = GeometryGPUResources(vertexBuffer: vertexBuffer, normalBuffer: normalBuffer, indexBuffer: indexBuffer, blas: blas, primitiveCount: mesh.indices.count / 3)
        geometryCache[mesh] = resources
        return resources
    }

    private func buildTLAS(instances: [InstanceGPUResources]) throws -> MTLAccelerationStructure {
        let instanceDescriptor = MTLInstanceAccelerationStructureDescriptor()
        instanceDescriptor.instanceCount = instances.count
        instanceDescriptor.instancedAccelerationStructures = instances.map { $0.geometry.blas }
        instanceDescriptor.instanceDescriptorType = .userID

        let descriptorStride = MemoryLayout<MTLAccelerationStructureUserIDInstanceDescriptor>.stride
        guard let instanceBuffer = context.device.makeBuffer(length: descriptorStride * instances.count, options: .storageModeManaged) else {
            throw RendererError.resourceCreationFailed("Failed to allocate TLAS instance buffer")
        }
        instanceDescriptor.instanceDescriptorStride = descriptorStride

        let descriptorPointer = instanceBuffer.contents().bindMemory(to: MTLAccelerationStructureUserIDInstanceDescriptor.self, capacity: instances.count)
        for (index, instance) in instances.enumerated() {
            var desc = MTLAccelerationStructureUserIDInstanceDescriptor()
            let transform = instance.transform
            desc.transformationMatrix = makePackedTransform(from: transform)
            desc.options = .opaque
            desc.mask = 0xFF
            desc.intersectionFunctionTableOffset = 0
            desc.accelerationStructureIndex = UInt32(index)
            desc.userID = UInt32(index)
            descriptorPointer[index] = desc
        }

        instanceBuffer.didModifyRange(0..<instanceBuffer.length)
        instanceDescriptor.instanceDescriptorBuffer = instanceBuffer

        guard let tlas = context.device.makeAccelerationStructure(size: context.device.accelerationStructureSizes(descriptor: instanceDescriptor).accelerationStructureSize) else {
            throw RendererError.resourceCreationFailed("Failed to create TLAS")
        }
        guard let scratch = context.device.makeBuffer(length: context.device.accelerationStructureSizes(descriptor: instanceDescriptor).buildScratchBufferSize, options: .storageModePrivate) else {
            throw RendererError.resourceCreationFailed("Failed to allocate TLAS scratch buffer")
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else {
            throw RendererError.commandEncodingFailed("Failed to encode TLAS build")
        }
        encoder.build(accelerationStructure: tlas, descriptor: instanceDescriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return tlas
    }

    private func makeMaterialBuffer(materials: [Material]) throws -> MTLBuffer {
        var gpuMaterials: [MaterialGPU] = materials.map { MaterialGPU(albedo: $0.albedo, emission: $0.emission, roughness: $0.roughness, metallic: $0.metallic) }
        guard let buffer = context.device.makeBuffer(bytes: &gpuMaterials, length: MemoryLayout<MaterialGPU>.stride * gpuMaterials.count, options: .storageModeManaged) else {
            throw RendererError.resourceCreationFailed("Failed to allocate material buffer")
        }
        buffer.didModifyRange(0..<buffer.length)
        return buffer
    }

    private func makeGeometryInfoBuffer(instances: [InstanceGPUResources]) throws -> MTLBuffer {
        var info: [GeometryInfoGPU] = instances.map { instance in
            GeometryInfoGPU(
                vertexBufferAddress: instance.geometry.vertexBuffer.gpuAddress,
                normalBufferAddress: instance.geometry.normalBuffer.gpuAddress,
                indexBufferAddress: instance.geometry.indexBuffer.gpuAddress,
                materialIndex: instance.materialIndex,
                primitiveCount: UInt32(instance.geometry.primitiveCount),
                padding: 0
            )
        }
        guard let buffer = context.device.makeBuffer(bytes: &info, length: MemoryLayout<GeometryInfoGPU>.stride * info.count, options: .storageModeManaged) else {
            throw RendererError.resourceCreationFailed("Failed to allocate geometry info buffer")
        }
        buffer.didModifyRange(0..<buffer.length)
        return buffer
    }

    private func makeInstanceInfoBuffer(instances: [InstanceGPUResources]) throws -> MTLBuffer {
        var info: [InstanceInfoGPU] = instances.map { instance in
            let m = instance.transform
            let upperLeft = simd_float3x3(columns: (
                simd_float3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                simd_float3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                simd_float3(m.columns.2.x, m.columns.2.y, m.columns.2.z)
            ))
            let normalMatrix = upperLeft.inverse.transpose
            return InstanceInfoGPU(transform: instance.transform, normalMatrix: normalMatrix, materialIndex: instance.materialIndex)
        }
        guard let buffer = context.device.makeBuffer(bytes: &info, length: MemoryLayout<InstanceInfoGPU>.stride * info.count, options: .storageModeManaged) else {
            throw RendererError.resourceCreationFailed("Failed to allocate instance info buffer")
        }
        buffer.didModifyRange(0..<buffer.length)
        return buffer
    }

    private func makePackedTransform(from matrix: simd_float4x4) -> MTLPackedFloat4x3 {
        let column0 = MTLPackedFloat3Make(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z)
        let column1 = MTLPackedFloat3Make(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z)
        let column2 = MTLPackedFloat3Make(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z)
        let column3 = MTLPackedFloat3Make(matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z)
        return MTLPackedFloat4x3(columns: (column0, column1, column2, column3))
    }
}

struct MaterialGPU {
    var albedo: simd_float3
    var padding0: Float = 0
    var emission: simd_float3
    var padding1: Float = 0
    var roughness: Float
    var metallic: Float
    var padding2: simd_float2 = .zero
}
