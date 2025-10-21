import Foundation
import Metal

final class RayTracingPipeline {
    let rayGenPipelineState: MTLComputePipelineState
    let rayTracingPipelineState: MTLRayTracingPipelineState
    let visibleFunctionTable: MTLVisibleFunctionTable

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let rayGenFunction = library.makeFunction(name: "rayGenMain"),
              let missFunction = library.makeFunction(name: "missShader"),
              let closestHitFunction = library.makeFunction(name: "closestHitShader")
        else {
            throw RendererError.pipelineCreationFailed("Missing required shader functions")
        }

        let hitGroupDescriptor = MTLAccelerationStructureHitGroupDescriptor()
        hitGroupDescriptor.closestHitFunction = closestHitFunction
        hitGroupDescriptor.intersectionFunctionName = nil
        hitGroupDescriptor.anyHitFunction = nil

        let rtDescriptor = MTLRayTracingPipelineDescriptor()
        rtDescriptor.label = "RTRMetal RayTracing"
        rtDescriptor.rayGenerationFunction = rayGenFunction
        rtDescriptor.missFunctions = [missFunction]
        rtDescriptor.hitGroupDescriptors = [hitGroupDescriptor]
        rtDescriptor.maxRecursionDepth = 2
        rtDescriptor.payloadMemoryLength = MemoryLayout<RayPayload>.stride
        rtDescriptor.attributeMemoryLength = MemoryLayout<RayAttributes>.stride

        self.rayTracingPipelineState = try device.makeRayTracingPipelineState(descriptor: rtDescriptor)

        let linkedFunctions = MTLLLinkedFunctions()
        linkedFunctions.functions = [missFunction, closestHitFunction]
        linkedFunctions.rayTracingPipeline = rayTracingPipelineState

        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.computeFunction = rayGenFunction
        computeDescriptor.linkedFunctions = linkedFunctions
        computeDescriptor.maxCallStackDepth = 4

        self.rayGenPipelineState = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)

        let visibleDescriptor = MTLVisibleFunctionTableDescriptor()
        visibleDescriptor.functionCount = 1
        guard let visibleTable = rayTracingPipelineState.makeVisibleFunctionTable(descriptor: visibleDescriptor) else {
            throw RendererError.pipelineCreationFailed("Failed to allocate visible function table")
        }
        visibleTable.setFunction(missFunction, index: 0)
        self.visibleFunctionTable = visibleTable
    }
}
