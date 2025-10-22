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
        self.log = OSLog(subsystem: "com.example.RTRMetalEngine", category: "MetalContext")

        guard let device = MetalContext.makeDevice(preferredName: preferredDeviceName, log: log) else {
            os_log("Unable to find a Metal device with ray tracing support", log: log, type: .error)
            return nil
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            os_log("Failed to create Metal command queue", log: log, type: .error)
            return nil
        }
        self.commandQueue = commandQueue

        do {
            self.defaultLibrary = try MetalContext.makeDefaultLibrary(for: device)
        } catch {
            os_log("Failed to load default Metal library: %{public}@", log: log, type: .error, String(describing: error))
            return nil
        }

        self.allocator = MTKMeshBufferAllocator(device: device)
    }

    private static func makeDevice(preferredName: String?, log: OSLog) -> MTLDevice? {
        let allDevices = MTLCopyAllDevices()

        func isRayTracingCapable(_ device: MTLDevice) -> Bool {
            if #available(macOS 11.0, *) {
                if device.supportsRaytracing { return true }
            }
            if #available(macOS 13.0, *) {
                if device.supportsFamily(.apple7) || device.supportsFamily(.apple8) || device.supportsFamily(.mac2) {
                    return true
                }
            }
            return false
        }

        let describe: (MTLDevice) -> String = { device in
            let supportsRT: String
            if #available(macOS 11.0, *) {
                supportsRT = device.supportsRaytracing ? "yes" : "no"
            } else {
                supportsRT = "n/a"
            }
            let families: [String]
            if #available(macOS 13.0, *) {
                var items: [String] = []
                for family in [MTLGPUFamily.apple7, .apple8, .mac2] {
                    if device.supportsFamily(family) {
                        items.append("\(family)")
                    }
                }
                families = items
            } else {
                families = []
            }
            return "\(device.name) (ray tracing: \(supportsRT), families: \(families.joined(separator: ",")))"
        }

        if !allDevices.isEmpty {
            os_log("Discovered Metal devices: %{public}@", log: log, type: .info, allDevices.map(describe).joined(separator: "; "))
        }

        if let preferredName, let match = allDevices.first(where: { $0.name.localizedCaseInsensitiveContains(preferredName) && isRayTracingCapable($0) }) {
            return match
        }

        if let capable = allDevices.first(where: { isRayTracingCapable($0) }) {
            return capable
        }

        guard let fallback = MTLCreateSystemDefaultDevice() else {
            return nil
        }

        os_log("Falling back to system default Metal device: %{public}@", log: log, type: .info, describe(fallback))
        guard isRayTracingCapable(fallback) else {
            os_log("System default device lacks required ray tracing capabilities", log: log, type: .error)
            return nil
        }
        return fallback
    }

    private static func makeDefaultLibrary(for device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: .module) {
            return library
        }
        if let url = Bundle.module.url(forResource: "default", withExtension: "metallib") {
            return try device.makeLibrary(URL: url)
        }
        if let library = device.makeDefaultLibrary() {
            return library
        }
        throw NSError(domain: "MetalContext", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to locate compiled Metal library in bundle"])
    }
}
