import XCTest
@testable import RTRMetalEngine

final class RTRMetalEngineTests: XCTestCase {
    func testCameraMatricesAreInvertible() {
        let camera = Camera(position: simd_float3(0, 0, 5), lookAt: simd_float3(0, 0, 0), verticalFov: 45, aspectRatio: 1.6)
        let uniforms = CameraUniforms(camera: camera)
        XCTAssertTrue(uniforms.viewMatrix.determinant != 0)
        XCTAssertTrue(uniforms.projectionMatrix.determinant != 0)
    }
}
