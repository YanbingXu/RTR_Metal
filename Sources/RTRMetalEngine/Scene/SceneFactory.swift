import Foundation
import simd

public enum SceneFactory {
    public static func cornellBox() -> Scene {
        let floor = Mesh.quad(name: "Floor", size: simd_float2(2, 2), normal: simd_float3(0, 1, 0), origin: simd_float3(0, -1, 0))
        let leftWall = Mesh.quad(name: "LeftWall", size: simd_float2(2, 2), normal: simd_float3(1, 0, 0), origin: simd_float3(-1, 0, 0), tangentAxis: simd_float3(0, 1, 0))
        let rightWall = Mesh.quad(name: "RightWall", size: simd_float2(2, 2), normal: simd_float3(-1, 0, 0), origin: simd_float3(1, 0, 0), tangentAxis: simd_float3(0, 1, 0))
        let backWall = Mesh.quad(name: "BackWall", size: simd_float2(2, 2), normal: simd_float3(0, 0, 1), origin: simd_float3(0, 0, -1))
        let ceiling = Mesh.quad(name: "Ceiling", size: simd_float2(2, 2), normal: simd_float3(0, -1, 0), origin: simd_float3(0, 1, 0))

        let red = Material(albedo: simd_float3(0.65, 0.05, 0.05))
        let green = Material(albedo: simd_float3(0.12, 0.45, 0.15))
        let white = Material(albedo: simd_float3(repeating: 0.73))

        let meshes = [floor, leftWall, rightWall, backWall, ceiling]

        let floorInstance = MeshInstance(mesh: floor, material: white, transform: matrix_identity_float4x4)
        let leftInstance = MeshInstance(mesh: leftWall, material: red, transform: matrix_identity_float4x4)
        let rightInstance = MeshInstance(mesh: rightWall, material: green, transform: matrix_identity_float4x4)
        let backInstance = MeshInstance(mesh: backWall, material: white, transform: matrix_identity_float4x4)
        let ceilingInstance = MeshInstance(mesh: ceiling, material: white, transform: matrix_identity_float4x4)

        let light = DirectionalLight(direction: simd_float3(-0.5, -1.0, -0.25), color: simd_float3(1.0, 0.95, 0.9), intensity: 8.0)

        return Scene(meshes: meshes,
                     instances: [floorInstance, leftInstance, rightInstance, backInstance, ceilingInstance],
                     directionalLight: light)
    }
}
