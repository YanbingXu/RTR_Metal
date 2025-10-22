import Foundation
import Metal

final class RayTracingPipeline {
    let rayGenPipelineState: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let rayGenFunction = library.makeFunction(name: "rayGenMain") else {
            throw RendererError.pipelineCreationFailed("Missing ray generation shader function")
        }

        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.computeFunction = rayGenFunction

        self.rayGenPipelineState = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
    }
}
