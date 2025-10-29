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

## Milestone 3 – Rendering Pipeline Completion

### Short-Term (Stage 3B focus)
1. **MPS pipeline hardening**
   - Ingest Cornell Box assets (reuse `/Users/yanbing.xu/Desktop/MetalRayTracing`).
   - Add shadow-ray pass and per-frame sample controls (config + CLI) while keeping deterministic test modes.
   - Extend regression tests to hash-check CPU/GPU outputs and multi-frame accumulation where hardware allows.
2. **Demo & documentation**
   - Update CLI samples/README with shading mode, accumulation, and scene selection guidance.
   - Capture reference hashes/screenshots for continuous verification.

### Mid-Term (Stage 3C + Stage 3A prep)
1. **On-screen sample** – Build an `MTKView`/SwiftUI demo that streams GPU output, exposes runtime toggles (accum reset, sample count, shading mode) and displays hash statistics.
2. **Native Metal Ray Tracing groundwork** – Finish TLAS/SBT plumbing, feature flags, and shader dispatch so the engine can switch to `MTLRayTracingPipelineState` when the SDK/device supports it.

## Milestone 4 – Polish & Validation
1. Richer lighting/material features (textures, multi-bounce, tone mapping) shared across MPS/native paths.
2. Performance & QA tooling – GPU frame capture, profiling scripts, automated hash comparisons.
3. Asset/documentation pipeline – Cornell Box plus additional scenes, onboarding docs, contribution guidelines.
4. Future roadmap – Denoising, animated meshes, and community extension points.
