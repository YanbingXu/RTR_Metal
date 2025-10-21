# Development Plan

## Milestone 1 – Project Bootstrap
1. Add Swift Package manifest with `RTRMetalEngine` library and `RTRMetalExample` executable targets.
2. Implement shared infrastructure: `MetalContext`, logging utilities, math helpers, and buffer alignment helpers.
3. Author architecture documentation (this document) and development plan.

## Milestone 2 – Core Engine
1. Define CPU-side scene representations (`Mesh`, `Material`, `Light`, `Scene`).
2. Implement GPU uploaders and acceleration-structure builders (`GeometryStore`, `AccelerationStructureBuilder`).
3. Create ray tracing pipeline setup, including shader binding table buffers.
4. Implement `Renderer` capable of producing a single-bounce ray traced image with a directional light.

## Milestone 3 – Example Application
1. Build a SwiftUI-based macOS viewer (`RTRMetalExample`) that embeds an `MTKView`.
2. Provide camera orbit controls and keyboard shortcuts for animation toggles.
3. Compose a Cornell-box inspired demo scene using engine APIs.
4. Add build and run instructions to the root `README.md`.

## Milestone 4 – Polish & Validation
1. Expose configurable rendering options (samples per pixel, max bounce count, tone mapping).
2. Add CPU-driven accumulation/reset logic for camera movement.
3. Validate rendering on Apple Silicon hardware, profiling GPU timings with Xcode GPU Frame Capture.
4. Document future work items (denoising, textures, animated meshes) for community contribution.

