import Foundation
import Metal
import simd

public struct Material: Hashable {
    public var albedo: simd_float3
    public var emission: simd_float3
    public var roughness: Float
    public var metallic: Float

    public init(albedo: simd_float3, emission: simd_float3 = .zero, roughness: Float = 0.3, metallic: Float = 0.0) {
        self.albedo = albedo
        self.emission = emission
        self.roughness = roughness
        self.metallic = metallic
    }
}

public struct Mesh: Hashable {
    public var name: String
    public var positions: [simd_float3]
    public var normals: [simd_float3]
    public var texcoords: [simd_float2]
    public var indices: [UInt32]

    public init(name: String, positions: [simd_float3], normals: [simd_float3], texcoords: [simd_float2], indices: [UInt32]) {
        self.name = name
        self.positions = positions
        self.normals = normals
        self.texcoords = texcoords
        self.indices = indices
    }
    public func hash(into hasher: inout Hasher) {
        hasher.combine(name)
        hasher.combine(positions.count)
        hasher.combine(indices.count)
    }
}

public extension Mesh {
    static func quad(name: String, size: simd_float2, normal: simd_float3, origin: simd_float3, tangentAxis: simd_float3 = simd_float3(1, 0, 0)) -> Mesh {
        let bitangent = simd_normalize(simd_cross(normal, tangentAxis))
        let tangent = simd_normalize(simd_cross(bitangent, normal))
        let halfSize = size * 0.5
        let p0 = origin + (-halfSize.x) * tangent + (-halfSize.y) * bitangent
        let p1 = origin + (halfSize.x) * tangent + (-halfSize.y) * bitangent
        let p2 = origin + (halfSize.x) * tangent + (halfSize.y) * bitangent
        let p3 = origin + (-halfSize.x) * tangent + (halfSize.y) * bitangent
        let positions = [p0, p1, p2, p3]
        let normals = Array(repeating: simd_normalize(normal), count: 4)
        let texcoords: [simd_float2] = [simd_float2(0, 0), simd_float2(1, 0), simd_float2(1, 1), simd_float2(0, 1)]
        let indices: [UInt32] = [0, 1, 2, 0, 2, 3]
        return Mesh(name: name, positions: positions, normals: normals, texcoords: texcoords, indices: indices)
    }
}

public struct MeshInstance {
    public var mesh: Mesh
    public var material: Material
    public var transform: simd_float4x4

    public init(mesh: Mesh, material: Material, transform: simd_float4x4) {
        self.mesh = mesh
        self.material = material
        self.transform = transform
    }
}

public struct DirectionalLight {
    public var direction: simd_float3
    public var color: simd_float3
    public var intensity: Float

    public init(direction: simd_float3, color: simd_float3, intensity: Float) {
        self.direction = simd_normalize(direction)
        self.color = color
        self.intensity = intensity
    }
}

public struct Scene {
    public var meshes: [Mesh]
    public var instances: [MeshInstance]
    public var directionalLight: DirectionalLight

    public init(meshes: [Mesh], instances: [MeshInstance], directionalLight: DirectionalLight) {
        self.meshes = meshes
        self.instances = instances
        self.directionalLight = directionalLight
    }
}
