import Foundation
import Metal
import MetalKit
import simd

public final class Renderer {
    public let context: MetalContext
    private let pipeline: RayTracingPipeline
    private let accelerationBuilder: AccelerationStructureBuilder
    private var frameUniformsBuffer: MTLBuffer
    private var scene: Scene?
    private var tlas: MTLAccelerationStructure?
    private var instanceResources: [InstanceGPUResources] = []
    private var materialBuffer: MTLBuffer?
    private var geometryInfoBuffer: MTLBuffer?
    private var instanceInfoBuffer: MTLBuffer?
    private var frameIndex: UInt32 = 0

    public var maxBounces: UInt32 = 1
    public var samplesPerPixel: UInt32 = 1

    public init?(context: MetalContext) {
        guard let frameBuffer = context.device.makeBuffer(length: MemoryLayout<FrameUniforms>.stride, options: .storageModeManaged) else {
            return nil
        }
        self.context = context
        do {
            self.pipeline = try RayTracingPipeline(device: context.device, library: context.defaultLibrary)
        } catch {
            return nil
        }
        self.accelerationBuilder = AccelerationStructureBuilder(context: context)
        self.frameUniformsBuffer = frameBuffer
    }

    public func upload(scene: Scene) throws {
        let (tlas, instances, materialBuffer, geometryInfoBuffer, instanceInfoBuffer) = try accelerationBuilder.buildScene(scene)
        self.tlas = tlas
        self.instanceResources = instances
        self.materialBuffer = materialBuffer
        self.geometryInfoBuffer = geometryInfoBuffer
        self.instanceInfoBuffer = instanceInfoBuffer
        self.scene = scene
        resetAccumulation()
    }

    public func draw(to view: MTKView, camera: Camera) throws {
        guard let drawable = view.currentDrawable,
              let tlas,
              let materialBuffer,
              let geometryInfoBuffer,
              let instanceInfoBuffer,
              let commandBuffer = context.commandQueue.makeCommandBuffer()
        else {
            return
        }

        guard let scene else {
            return
        }
        let lightUniform = DirectionalLightUniform(direction: scene.directionalLight.direction, intensity: scene.directionalLight.intensity, color: scene.directionalLight.color)
        let drawableWidth = Float(view.drawableSize.width)
        let drawableHeight = Float(view.drawableSize.height)
        if drawableWidth <= 0 || drawableHeight <= 0 {
            return
        }
        let resolution = simd_float2(drawableWidth, drawableHeight)
        var uniforms = FrameUniforms(camera: CameraUniforms(camera: camera), light: lightUniform, resolution: resolution, frameIndex: frameIndex, maxBounces: maxBounces, sampleCount: samplesPerPixel)
        memcpy(frameUniformsBuffer.contents(), &uniforms, MemoryLayout<FrameUniforms>.stride)
        frameUniformsBuffer.didModifyRange(0..<MemoryLayout<FrameUniforms>.stride)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RendererError.commandEncodingFailed("Failed to make compute encoder")
        }

        encoder.setComputePipelineState(pipeline.rayGenPipelineState)

        let threadGroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let threadGroups = MTLSize(width: (Int(view.drawableSize.width) + threadGroupSize.width - 1) / threadGroupSize.width,
                                   height: (Int(view.drawableSize.height) + threadGroupSize.height - 1) / threadGroupSize.height,
                                   depth: 1)

        encoder.setBuffer(frameUniformsBuffer, offset: 0, index: 0)
        encoder.setBuffer(materialBuffer, offset: 0, index: 1)
        encoder.setBuffer(geometryInfoBuffer, offset: 0, index: 2)
        encoder.setBuffer(instanceInfoBuffer, offset: 0, index: 3)
        encoder.setAccelerationStructure(tlas, bufferIndex: 0)
        encoder.setTexture(drawable.texture, index: 0)

        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
        frameIndex &+= 1
    }

    public func resetAccumulation() {
        frameIndex = 0
    }
}
