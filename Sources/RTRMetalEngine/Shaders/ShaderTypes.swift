import simd

public struct DirectionalLightUniform {
    public var direction: simd_float3
    public var intensity: Float
    public var color: simd_float3
    public var padding: Float

    public init(direction: simd_float3, intensity: Float, color: simd_float3) {
        self.direction = direction
        self.intensity = intensity
        self.color = color
        self.padding = 0
    }
}

public struct FrameUniforms {
    public var camera: CameraUniforms
    public var light: DirectionalLightUniform
    public var resolution: simd_float2
    public var paddingResolution: simd_float2 = .zero
    public var frameIndex: UInt32
    public var maxBounces: UInt32
    public var sampleCount: UInt32
    public var padding: UInt32

    public init(camera: CameraUniforms, light: DirectionalLightUniform, resolution: simd_float2, frameIndex: UInt32 = 0, maxBounces: UInt32 = 1, sampleCount: UInt32 = 1) {
        self.camera = camera
        self.light = light
        self.resolution = resolution
        self.frameIndex = frameIndex
        self.maxBounces = maxBounces
        self.sampleCount = sampleCount
        self.padding = 0
    }
}

public struct RayPayload {
    public var radiance: simd_float3
    public var attenuation: simd_float3
    public var depth: UInt32
    public var active: UInt32

    public init() {
        self.radiance = .zero
        self.attenuation = simd_float3(repeating: 1)
        self.depth = 0
        self.active = 1
    }
}

public struct RayAttributes {
    public var barycentricCoord: simd_float2
    public init() {
        self.barycentricCoord = .zero
    }
}
