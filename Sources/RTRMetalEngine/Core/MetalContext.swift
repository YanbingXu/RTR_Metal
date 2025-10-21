import Foundation
import Metal
import MetalKit
import os.log

public final class MetalContext {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let defaultLibrary: MTLLibrary
    public let allocator: MTKMeshBufferAllocator
    public let log: OSLog

    public init?(preferredDeviceName: String? = nil) {
        guard let device = MetalContext.makeDevice(preferredName: preferredDeviceName) else {
            return nil
        }
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            return nil
        }
        self.commandQueue = commandQueue
        do {
            self.defaultLibrary = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            return nil
        }
        self.allocator = MTKMeshBufferAllocator(device: device)
        self.log = OSLog(subsystem: "com.example.RTRMetalEngine", category: "MetalContext")
    }

    private static func makeDevice(preferredName: String?) -> MTLDevice? {
        if let preferredName {
            if let device = MTLCopyAllDevices().first(where: { $0.name.contains(preferredName) && $0.supportsRaytracing }) {
                return device
            }
        }
        if let best = MTLCopyAllDevices().first(where: { $0.supportsRaytracing }) {
            return best
        }
        return MTLCreateSystemDefaultDevice()?.supportsRaytracing == true ? MTLCreateSystemDefaultDevice() : nil
    }
}
