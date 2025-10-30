## Stage 1: Project Shell & Tooling
**Goal**: Replace Swift package with CMake-based C++ project scaffold, shared engine/static library target, sample app target, and shader build integration stubs.
**Success Criteria**: CMake config generates build files; `metal` shader compilation hooks exist; placeholder engine library builds and links into sample.
**Tests**: Configure and build via `cmake --build` on macOS; run placeholder unit test binary.
**Status**: Complete

## Stage 2: Core Engine Infrastructure
**Goal**: Implement Metal device/context wrapper, memory/resource managers, logging, configuration loading, and math utilities needed by renderer.
**Success Criteria**: Engine initializes Metal device & command queues; can load mesh data and upload buffers; unit tests cover math/config modules.
**Tests**: New C++ unit tests for math/utilities; run sample in headless init mode validating context creation.
**Status**: Complete

## Stage 3: Ray Tracing Pipelines
**Goal**: Deliver both hardware RT and MPS pipelines capable of rendering Cornell Box/off-screen scenes with ray-tracing features (reflections, shadows, refraction), plus supporting infrastructure for deterministic validation.

### Stage 3A: Hardware RT Path (Metal Ray Tracing)
**Goal**: Build BLAS/TLAS constructors, shader binding tables, and dispatch a hardware RT frame to an off-screen target.
**Success Criteria**:
- Diagnostic Cornell Box renders through the hardware RT path on RT-capable devices, producing non-black frames with basic lighting.
- TLAS build + instance management validated via logs/tests; SBT layout documented.
- Renderer toggles between hardware RT and MPS backends at runtime with graceful fallback when headers/devices are missing.
**Tests**: Capability-gated integration tests for TLAS/SBT creation, hash comparison on RT hardware, smoke test ensuring fallback when unavailable.
**Status**: In Progress

### Stage 3B: MPS Compute Path
**Goal**: Mirror the reference MPS-based path tracer using compute shaders for ray generation, shading, and accumulation to cover machines without stable hardware RT.
**Success Criteria**:
- GPU shading kernels (ray, shade, accumulate, optional shadow/refraction passes) render Cornell/prism scenes with deterministic hashes.
- Resolution/sample-count configurable; accumulation reset exposed (CLI + API).
- CPU fallback retained for testing with unified comparison and tolerance thresholds.
**Tests**: Frame hash/diff tests, multi-frame accumulation checks, scene-switch validation, capability probe skip logic.
**Status**: In Progress

### Stage 3C: Examples & Scene Assets
**Goal**: Provide off-screen and on-screen demos consuming shared renderer infrastructure and showcasing ray-traced effects.
**Success Criteria**:
- CLI tool renders Cornell Box and additional assets to disk (PNG/PPM) with recorded hashes and backend selection flags.
- On-screen demo (MetalKit/SwiftUI) displays progressive rendering with runtime toggles (backend, scene, sample count, accumulation reset, screenshot capture).
- Documentation and scripts describe running both demos, backend requirements, and regression expectations.
**Tests**: Automated CLI hash comparisons; manual checklist for on-screen demo until CI coverage is feasible.
**Status**: Not Started

## Stage 4: Polish & Documentation
**Goal**: Round out material models (reflections/refraction, textures), ensure parity between hardware RT and MPS shading, and publish developer guidance.
**Success Criteria**: Extended shading features shared across backends; performance/QA tooling (hash baselines, profiling); README/docs capture usage and troubleshooting.
**Tests**: Expanded hash/image diff suite, profiling scripts, documentation linting.
**Status**: Not Started
