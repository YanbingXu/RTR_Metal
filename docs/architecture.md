# RTR Metal Engine Architecture

## Goals

- Provide a cross-project C++20 foundation for Metal-based hardware ray tracing on Apple Silicon.
- Keep the runtime modular so renderer subsystems (context, resources, scene, shaders) can be exercised independently in tests.
- Ship a reference macOS sample that demonstrates camera control, resource uploads, and the ray tracing pipeline.

## Layered Structure

```
Application Layer (RTRMetalSample, future AppKit UI)
    ↓
Rendering Facade (Renderer, Frame graph)
    ↓
Resource & Scene Systems (BufferAllocator, future Geometry/Material stores)
    ↓
Core Platform Services (MetalContext, Logger, ConfigLoader, Math)
    ↓
Metal GPU + Shaders
```

### Core Platform Services (`engine/Core`)

- **Logger** – Minimal, thread-safe printf logger that standardizes severity + tag output. Used across all subsystems for diagnostic messages.
- **FileSystem** – Thin wrapper over `std::filesystem` for existence checks and text file ingestion.
- **ConfigLoader** – Parses simple `key=value` configuration files into `EngineConfig`, providing defaults for shader library locations.
- **Math** – SIMD helpers for projection/look-at matrices, unit conversions, and bounding boxes. These utilities back camera setup and acceleration structure bounds.

### Rendering Platform (`engine/Rendering`)

- **MetalContext** – Owns the `MTLDevice` and `MTLCommandQueue`, exposes capability queries, and centralizes device logging. Fails fast when no Metal device is available (as seen in headless CI environments).
- **BufferAllocator** – High-level GPU buffer creation entry point. It currently short-circuits when Metal is unavailable, while still providing the API surface (`BufferHandle`) required by upcoming resource managers.
- **Renderer** – Façade exposed to applications. Wires configuration, context creation, and staging for per-frame work. Presently emits stub frame logs; later stages will attach the full ray tracing loop here.

### Scene & Resource Systems (`engine/Scene`)

- **Mesh / Material / Scene** – Immutable CPU-side descriptions with bounding-box computation and instance tracking. These feed future GPU upload paths and TLAS assembly.
- **SceneBuilder** – Convenience façade for constructing meshes from raw position/index arrays and registering default materials; tests validate mesh insertion and error handling.
- **GeometryStore** – Uploads mesh vertex/index data to GPU buffers through `BufferAllocator`, preparing inputs for upcoming BLAS/TLAS builders.
- **AccelerationStructureBuilder** – Queries Metal for BLAS memory requirements and (when hardware allows) builds diagnostic BLAS instances to validate the ray-tracing command path end-to-end.
- **RayTracingPipeline** – Wraps `MTLRayTracingPipelineState` creation when the SDK exposes `MetalRayTracing.h`; otherwise falls back silently so the rest of the engine remains usable.
- **MPSPathTracer** – Initializes an MPS-based backend (`MPSSupportsMTLDevice`) to provide a portable fallback/existing sample parity when hardware ray tracing APIs are unavailable.
- Upcoming work will extend these types with GPU upload hooks, material textures, and acceleration-structure builders leveraging the math helpers already in place.

### Shaders (`shaders/`)

- Metal source files compiled via CMake custom commands. The build emits `build/shaders/RTRShaders.metallib`, ensuring shader compilation is integrated with standard `cmake --build` invocations.

## Testing Strategy

- **GoogleTest** backs deterministic unit tests for core utilities (config loader, math, logger) and rendering scaffolding (buffer allocator behaviour on invalid contexts).
- Headless environments typically lack a Metal device; tests assert graceful failure paths rather than GPU calls in that configuration.
- Future stages will add checksum-based offline render validations once we can generate frame outputs deterministically.

## Runtime Expectations

- The sample application (`RTRMetalSample`) initializes the renderer using the parsed configuration and logs device availability. On machines without a Metal device, the binary exits gracefully after reporting the issue.
- Once the ray tracer is implemented, the sample will expand into an AppKit/MetalKit front-end that presents rendered frames and camera controls.

## File Layout Snapshot

- `CMakeLists.txt` – Root build orchestration, shader compilation rules, GoogleTest integration.
- `engine/include/` – Public headers organized by module (`Core`, `Rendering`, `Scene`).
- `engine/src/` – Implementation files, including Objective-C++ sources for Metal interop (`*.mm`).
- `sample/` – Temporary CLI sample wiring up the renderer façade.
- `tests/src/` – GoogleTest suites mirrored by module (`core/`, `rendering/`).
- `shaders/` – Metal shader sources compiled into the runtime library.

## Next Architectural Steps

1. Flesh out scene data structures and host-side resource upload flows tied to the new `BufferAllocator`.
2. Implement BLAS/TLAS builders leveraging `Math::BoundingBox` and the forthcoming geometry store.
3. Integrate the ray tracing pipeline (shader binding tables, dispatch) inside `Renderer`, replacing the current stub output.
4. Build the interactive macOS sample that exercises renderer features end-to-end.
