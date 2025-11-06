> Status checkpoint (2025-11-06): Stage 1–3A complete; Stage 3B active; Stage 3C and Stage 4 queued.
> Refer to [AGENTS.md](AGENTS.md) for contributor workflow guidance and entry points into the codebase.

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
**Goal**: Track the Apple reference implementation step by step—finish the hardware RT path (matching the official Cornell result) before touching the MPS fallback; only once hardware parity is achieved do we mirror the fallback pipeline.

### Stage 3A: Hardware-Accelerated Compute Path (Metal Ray Tracing)
**Goal**: Build BLAS/TLAS constructors and dispatch a compute-based ray tracing kernel that leverages hardware traversal to render diagnostic frames.
**Success Criteria**:
- Diagnostic Cornell Box renders through the compute ray tracing pipeline on RT-capable devices, producing non-black frames with basic lighting.
- TLAS/BLAS build + instance management validated via logs/tests; resource buffer layout and intersection/visible function usage documented.
- Renderer toggles between compute RT and fallback backends at runtime with graceful degradation when `supportsRaytracing` is false.
**Tests**: Capability-gated integration tests for TLAS creation and compute dispatch, hash comparison on RT hardware, smoke test ensuring fallback when unavailable.
**Status**: Complete

### Stage 3B: Hardware Rendering Polish
**Goal**: Rebuild the hardware kernel to mirror the Apple sample (Cornell lighting, reflections, refraction) so we ship a complete hardware RT frame before considering fallbacks.
**Success Criteria**:
- Ray-gen kernel produces the Cornell reference image from `~/Desktop/metal_RTR_official_example`.
- Material/enclosure data in `RTRRayTracing.metal` stays aligned with the reference shading code.
- Renderer writes frame dumps via `writeRayTracingOutput`.
- Docs outline hardware-only requirements and troubleshooting steps.
**Tests**: Basic integration run on hardware RT devices (smoke) and unit coverage for resource packing.
**Status**: In Progress — Cornell shading ported; capture hashes + doc updates pending

### Stage 3C: MPS Compute Path
**Goal**: After hardware parity, mirror the open-source `~/Desktop/MetalRayTracing` fallback sample using MPS compute shaders so non-hardware devices render the same Cornell frame.
**Success Criteria**:
- GPU shading kernels (ray, shade, accumulate, optional shadow/refraction passes) render Cornell/prism scenes with deterministic hashes.
- Resolution/sample-count configurable; accumulation reset exposed (CLI + API).
- CPU fallback retained for testing with unified comparison and tolerance thresholds.
**Tests**: Frame hash/diff tests, multi-frame accumulation checks, scene-switch validation, capability probe skip logic.
**Status**: Not Started

### Stage 3D: Examples & Scene Assets
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
