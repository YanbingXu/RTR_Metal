# RTR Metal Engine Vision

## Goal Snapshot
- Deliver a reusable hardware ray tracing engine targeting macOS 14+ on Apple Silicon.
- Support two interchangeable backends:
  - **Hardware-accelerated compute pipeline** built on Metal acceleration structures + compute kernels (TLAS/BLAS, linked/visible functions, `setAccelerationStructure:` dispatch).
  - **MPS Pipeline** that mirrors the reference `/Users/yanbing.xu/Desktop/MetalRayTracing` sample using compute shaders for ray tracing effects when native RT is unavailable or unstable.
- Ship both an off-screen CLI renderer and an on-screen demo that can ingest Cornell Box assets and showcase reflections, refraction, soft shadows, and other ray-traced lighting features.
- Maintain automated tests (hash/image checks, scene validation) to guard regression, plus documentation for onboarding and usage.

## Current Reality
- ~~Hardware RT work is blocked on inconsistent platform support (MacBook Pro M4 Pro); the renderer currently initialises the device and diagnostic BLAS but lacks TLAS/SBT/dispatch.~~
- Compute-based hardware traversal is planned using TLAS/BLAS + ray tracing kernels; diagnostic BLAS/TLAS builds already exist and are being extended to drive compute dispatch.
- MPS fallback path is active and under construction: GPU ray, shading, and accumulation kernels exist, with CPU shading retained for determinism and testing.
- Scene ingestion supports synthetic prism and Cornell Box scenes via `SceneBuilder`/`GeometryStore` with deterministic hash tests.

## Desired End State
1. **Hardware-Accelerated Compute Pipeline**
   - TLAS/BLAS builder, resource buffer layout, and compute dispatch writing to render targets.
   - Visual parity with the MPS path for shared shading models.
   - Capability detection and graceful fallback when `supportsRaytracing` is missing.
2. **MPS Pipeline**
   - Fully GPU-driven (ray → intersect → shade → accumulate) with optional CPU verification path.
   - Configurable sampling, accumulation reset, and deterministic hashes for regression.
3. **Examples**
   - CLI off-screen renderer accepting Cornell Box assets and emitting reference images/hashes.
   - MetalKit/SwiftUI on-screen viewer with runtime toggles (backend, samples, scene selection).
4. **Quality Infrastructure**
   - `ctest` coverage for deterministic behaviours, image/hash comparison, and capability probes.
   - Documentation describing build requirements, backend selection, and validation workflow.

## Guiding Principles
- Keep the hardware RT and MPS pipelines aligned in scene data, shading language, and testing to minimise divergence.
- Prioritise deterministic outputs for automated verification before extending to progressive/interactive modes.
- Incrementally layer features (e.g., start with reflections/shadows before refraction/complex materials) with clear doc and test updates at each milestone.
