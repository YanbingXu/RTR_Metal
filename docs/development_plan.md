# Development Plan

## Milestone 1 – Project Bootstrap (✅)
1. Establish CMake build with `RTRMetalEngine` library, sample targets, and shader compilation.
2. Implement logging, math helpers, Metal context, buffer allocator, and configuration loader.
3. Draft architecture/development guidelines.

## Milestone 2 – Core Engine (✅)
1. Define CPU-side scene graph (`Mesh`, `Material`, `Scene`, builders).
2. Integrate `GeometryStore`, `BufferAllocator`, and acceleration-structure scaffolding.
3. Assemble renderer façade and diagnostic BLAS build.

## Milestone 3 – Ray Tracing Pipelines (🚧)
### Focus A – Hardware RT Pipeline
- Implement TLAS builder, shader binding table management, and ray dispatch writing into textures.
- Align shading inputs (materials, camera, lighting) with the MPS path.
- Detect capability at runtime; fallback to MPS when RT APIs unavailable.
- Target Effects: primary visibility, soft shadows, reflections; refraction once base path is stable.

### Focus B – MPS Compute Pipeline
- Finish GPU-only ray/shade/accumulate loop based on the MetalRayTracing reference sample.
- Support configurable resolution, samples per pixel, accumulation reset, and deterministic hashes.
- Keep CPU shading path as deterministic check; extend tests to cover scene switching (prism, Cornell).
- Introduce optional passes (shadow rays, refractive materials) to mirror hardware RT behaviour.

### Focus C – Examples
- Off-screen CLI renderer accepting Cornell Box assets, outputting PPM/PNG + hash metadata for both backends.
- On-screen demo (MetalKit/SwiftUI) with backend/sampling toggles, accumulation HUD, and screenshot capture.
- Shared scene loading/config handling between demos.

## Milestone 4 – Polish & Validation (🔒)
1. Expand material system (textures, multi-bounce, tone mapping) and keep feature parity across backends.
2. Add profiling & QA tooling (hash baselines, performance scripts, capture guides).
3. Document onboarding, backend requirements, regression workflow, and known limitations.
4. Explore roadmap extensions (denoising, animation support) once pipelines are stable.

## Reference
- `IMPLEMENTATION_PLAN.md` contains the stage statuses and acceptance tests.
- `/Users/yanbing.xu/Desktop/MetalRayTracing` remains the reference sample for the MPS compute pipeline.
