import Foundation
import Metal
import MetalKit
import os.log
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
    private var accumulationTexture: MTLTexture?
    private var readbackBuffer: MTLBuffer?
    private var readbackBytesPerRow: Int = 0

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
        if frameIndex == 0 {
            os_log("Drawable size: %{public}.1fx%{public}.1f", log: context.log, type: .info, drawableWidth, drawableHeight)
        }
        var uniforms = FrameUniforms(camera: CameraUniforms(camera: camera), light: lightUniform, resolution: resolution, frameIndex: frameIndex, maxBounces: maxBounces, sampleCount: samplesPerPixel)
        memcpy(frameUniformsBuffer.contents(), &uniforms, MemoryLayout<FrameUniforms>.stride)
        frameUniformsBuffer.didModifyRange(0..<MemoryLayout<FrameUniforms>.stride)

        let intWidth = Int(drawableWidth)
        let intHeight = Int(drawableHeight)

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: view.colorPixelFormat,
                                                                         width: intWidth,
                                                                         height: intHeight,
                                                                         mipmapped: false)
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        if accumulationTexture == nil || accumulationTexture?.width != intWidth || accumulationTexture?.height != intHeight {
            accumulationTexture = context.device.makeTexture(descriptor: textureDescriptor)
        }

        let bytesPerPixel = 4
        let alignedRow = max(256, ((intWidth * bytesPerPixel + 0xFF) & ~0xFF))
        readbackBytesPerRow = alignedRow
        let requiredSize = alignedRow * intHeight
        if readbackBuffer == nil || readbackBuffer!.length < requiredSize {
            readbackBuffer = context.device.makeBuffer(length: requiredSize, options: .storageModeShared)
        }

        guard let outputTexture = accumulationTexture,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RendererError.commandEncodingFailed("Failed to make compute encoder")
        }

        encoder.setComputePipelineState(pipeline.rayGenPipelineState)

        let threadGroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let threadCount = MTLSize(width: Int(drawableWidth), height: Int(drawableHeight), depth: 1)

        encoder.setBuffer(frameUniformsBuffer, offset: 0, index: 0)
        encoder.setBuffer(materialBuffer, offset: 0, index: 1)
        encoder.setBuffer(geometryInfoBuffer, offset: 0, index: 2)
        encoder.setBuffer(instanceInfoBuffer, offset: 0, index: 3)
        encoder.setAccelerationStructure(tlas, bufferIndex: 0)
        encoder.setTexture(outputTexture, index: 0)

        if frameIndex == 0 {
            os_log("dispatchThreads: %{public}ux%{public}u, threadsPerGroup: %{public}ux%{public}u",
                   log: context.log, type: .info,
                   UInt32(threadCount.width), UInt32(threadCount.height),
                   UInt32(threadGroupSize.width), UInt32(threadGroupSize.height))
        }
        encoder.dispatchThreads(threadCount, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        if let blit = commandBuffer.makeBlitCommandEncoder(), let accumulationTexture, let readbackBuffer {
            let size = MTLSize(width: intWidth, height: intHeight, depth: 1)
            blit.copy(from: accumulationTexture,
                      sourceSlice: 0,
                      sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: size,
                      to: drawable.texture,
                      destinationSlice: 0,
                      destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))

            blit.copy(from: accumulationTexture,
                      sourceSlice: 0,
                      sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: size,
                      to: readbackBuffer,
                      destinationOffset: 0,
                      destinationBytesPerRow: readbackBytesPerRow,
                      destinationBytesPerImage: readbackBytesPerRow * intHeight)
            blit.endEncoding()
        }

        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            if #available(macOS 11.0, *) {
                if let error = buffer.error {
                    os_log("Command buffer error: %{public}@", log: self.context.log, type: .error, String(describing: error))
                } else {
                    os_log("Command buffer completed", log: self.context.log, type: .info)
                    if let readback = self.readbackBuffer {
                        let ptr = readback.contents().bindMemory(to: UInt8.self, capacity: readback.length)
                        let r = ptr[0]
                        let g = ptr[1]
                        let b = ptr[2]
                        let a = ptr[3]
                        os_log("Top-left pixel RGBA: (%{public}u, %{public}u, %{public}u, %{public}u)",
                               log: self.context.log, type: .info, r, g, b, a)
                    }
                }
            } else if let error = buffer.error {
                print("Command buffer error: \(error)")
            }
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
        frameIndex &+= 1
    }

    public func resetAccumulation() {
        frameIndex = 0
    }
}
