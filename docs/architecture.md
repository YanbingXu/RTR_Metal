# RTR Metal Ray Tracing Engine Architecture

## Goals

- Deliver a reusable, real-time ray tracing engine targeting macOS/iOS devices with Metal ray tracing support.
- Support animated camera, multiple triangle meshes, physically based materials, and dynamic scene updates.
- Provide a minimal example viewer that demonstrates rendering a simple scene.

## High-Level Overview

The engine is organized into four layers:

1. **Platform Layer** – `MetalContext` encapsulates `MTLDevice`, `MTLCommandQueue`, default libraries, and resource loading.
2. **Scene Layer** – `Scene`, `Mesh`, `Material`, and `Camera` types describe world data in an engine-friendly format. A `SceneBuilder` converts raw vertex arrays into GPU buffers and builds bottom/top level acceleration structures.
3. **Rendering Layer** – `Renderer` manages ray tracing pipelines, shader binding tables, and per-frame command encoding. This layer owns `RayTracingPipeline`, `RayTracingResources`, and the per-frame uniform buffers.
4. **Application Layer** – Client applications (for example, `RTRMetalExample`) compose scenes, update camera transforms, and drive rendering through an `MTKViewDelegate` adapter.

## Data Flow

```
Application (MTKViewDelegate)
    ↓
SceneBuilder ← Asset loading (Model I/O)
    ↓
GPU resources (buffers/textures) + Acceleration Structures
    ↓
Renderer (RayTracingPipeline + RayTracingPass)
    ↓
Swapchain (MTKView currentDrawable)
```

## Key Components

### MetalContext
- Discovers a ray-tracing capable `MTLDevice`.
- Creates shared `MTLCommandQueue`, `MTLLibrary`, and manages function lookups.

### Mesh & GeometryStore
- `Mesh` holds CPU-side vertex/index data.
- `GeometryStore` uploads vertex/index buffers and BLAS descriptors per mesh.
- Uses `MTLPrimitiveAccelerationStructureDescriptor` to build a BLAS for triangle geometry.

### Scene / SceneGraph
- `Scene` aggregates meshes, materials, lights, and per-instance transforms.
- Supports multiple mesh instances referencing shared geometry buffers.
- Maintains a TLAS via `MTLInstanceAccelerationStructureDescriptor` with update-on-demand support.

### RayTracingPipeline
- Loads Metal shader functions from `MetalRayTracing.metal`.
- Configures `MTLRayTracingPipelineDescriptor`, hit groups, callable functions, and shader binding tables.

### Renderer
- Maintains per-frame uniform buffers (`CameraUniforms`, `FrameUniforms`).
- Encodes ray generation pass using `MTLComputeCommandEncoder` with `dispatchThreads` sized to the render target.
- Updates accumulation buffers for temporal anti-aliasing and path tracing iterations.

### Example Application
- SwiftUI + MetalKit based viewer.
- Implements `RayTracingView` which bridges the engine renderer to an `MTKView`.
- Provides an interactive camera controller and a demo scene containing spheres and a quad floor.

## Extensibility

- Additional material models via expanding the `Material` buffer and closest-hit shader.
- Support for textures by extending the resource binding table and sample fetch logic in shaders.
- Animation by recomputing TLAS instance transforms each frame.

## Files and Modules

- `Package.swift` – Swift Package definition with `RTRMetalEngine` library and `RTRMetalExample` executable.
- `Sources/RTRMetalEngine/` – Engine core types (`Context`, `Scene`, `Renderer`, `Shaders`).
- `Sources/RTRMetalExample/` – Example macOS app using SwiftUI + MetalKit.
- `Resources/Shaders/MetalRayTracing.metal` – Shader library used by engine.

## Testing & Validation

- Example viewer renders a Cornell-box-style scene to validate shading, TLAS updates, and accumulation.
- `swift test` placeholder for future unit tests; GPU validation is manual via the example app.

