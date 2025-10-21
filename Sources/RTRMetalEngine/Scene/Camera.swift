import simd

public struct Camera {
    public var position: simd_float3
    public var lookAt: simd_float3
    public var up: simd_float3
    public var verticalFov: Float
    public var aspectRatio: Float

    public init(position: simd_float3, lookAt: simd_float3, up: simd_float3 = simd_float3(0, 1, 0), verticalFov: Float = 45, aspectRatio: Float = 1.0) {
        self.position = position
        self.lookAt = lookAt
        self.up = simd_normalize(up)
        self.verticalFov = verticalFov
        self.aspectRatio = aspectRatio
    }

    public func viewMatrix() -> simd_float4x4 {
        let forward = simd_normalize(lookAt - position)
        let right = simd_normalize(simd_cross(forward, up))
        let camUp = simd_cross(right, forward)
        var matrix = simd_float4x4(
            simd_float4(right, 0),
            simd_float4(camUp, 0),
            simd_float4(-forward, 0),
            simd_float4(0, 0, 0, 1)
        )
        matrix.columns.3 = simd_float4(-simd_dot(right, position), -simd_dot(camUp, position), simd_dot(forward, position), 1)
        return matrix
    }

    public func projectionMatrix(near: Float = 0.01, far: Float = 1000.0) -> simd_float4x4 {
        let fov = verticalFov * (.pi / 180)
        let yScale = 1 / tan(fov / 2)
        let xScale = yScale / aspectRatio
        let zRange = far - near
        let zScale = -(far + near) / zRange
        let wzScale = -2 * far * near / zRange
        return simd_float4x4(
            simd_float4(xScale, 0, 0, 0),
            simd_float4(0, yScale, 0, 0),
            simd_float4(0, 0, zScale, -1),
            simd_float4(0, 0, wzScale, 0)
        )
    }
}

public struct CameraUniforms {
    public var viewMatrix: simd_float4x4
    public var projectionMatrix: simd_float4x4
    public var inverseViewMatrix: simd_float4x4
    public var inverseProjectionMatrix: simd_float4x4
    public var position: simd_float3
    public var padding: Float = 0

    public init(camera: Camera) {
        self.viewMatrix = camera.viewMatrix()
        self.projectionMatrix = camera.projectionMatrix()
        self.inverseViewMatrix = camera.viewMatrix().inverse
        self.inverseProjectionMatrix = camera.projectionMatrix().inverse
        self.position = camera.position
    }
}
