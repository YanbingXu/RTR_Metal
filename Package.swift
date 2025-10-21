// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "RTRMetal",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "RTRMetalEngine",
            targets: ["RTRMetalEngine"]
        ),
        .executable(
            name: "RTRMetalExample",
            targets: ["RTRMetalExample"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections", from: "1.0.4")
    ],
    targets: [
        .target(
            name: "RTRMetalEngine",
            dependencies: [
                .product(name: "Collections", package: "swift-collections")
            ],
            path: "Sources/RTRMetalEngine",
            resources: [
                .process("Shaders")
            ]
        ),
        .executableTarget(
            name: "RTRMetalExample",
            dependencies: ["RTRMetalEngine"],
            path: "Sources/RTRMetalExample",
            resources: []
        ),
        .testTarget(
            name: "RTRMetalEngineTests",
            dependencies: ["RTRMetalEngine"],
            path: "Tests/RTRMetalEngineTests"
        )
    ]
)
