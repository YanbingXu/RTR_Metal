> Status checkpoint (2025-11-06): Stage 1–3C complete; Stage 3D (hardware shading polish) active; Stage 4 (software/fallback return) queued.
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
**Goal**: Track the Apple reference implementation step by step—Stage 3 is now hardware-only. Hardware parity against Apple's reference scenes must be achieved before we bring back any software/MPS fallback (punted to Stage 4).

### Stage 3A: Hardware-Accelerated Compute Path (Metal Ray Tracing)
**Goal**: Build BLAS/TLAS constructors and dispatch a compute-based ray tracing kernel that leverages hardware traversal to render diagnostic frames.
**Success Criteria**:
- Diagnostic Cornell Box renders through the compute ray tracing pipeline on RT-capable devices, producing non-black frames with basic lighting.
- TLAS/BLAS build + instance management validated via logs/tests; resource buffer layout and intersection/visible function usage documented.
- Hardware renderer reports clear errors on non-RT devices (no more silent fallback).
**Tests**: Capability-gated integration tests for TLAS creation and compute dispatch, hash comparison on RT hardware.
**Status**: Complete

### Stage 3B: Hardware Rendering Polish
**Goal**: Rebuild the hardware kernel to mirror the Apple sample (Cornell lighting, reflections, refraction) so we ship a complete hardware RT frame before considering any future fallback work.
**Success Criteria**:
- Ray-gen kernel produces the Cornell reference image from `~/Desktop/metal_RTR_official_example`.
- Material/enclosure data in `RTRRayTracing.metal` stays aligned with the reference shading code.
- Renderer writes frame dumps via `writeRayTracingOutput`.
- Docs outline hardware-only requirements and troubleshooting steps.
**Tests**: Basic integration run on hardware RT devices (smoke) and unit coverage for resource packing.
**Status**: Complete — Cornell shading path committed; hashes/docs scheduled with on-screen demo work

### Stage 3C: On-Screen Examples & Scene Assets
**Goal**: Prioritize an on-screen sample that exercises the engine from a simple app shell, matching Apple’s UI patterns. Hardware-only toggles remain.
**Success Criteria**:
- Engine exposes the minimal swapchain/display loop so the sample app remains thin (UI logic in app, rendering control in engine).
- MetalKit/SwiftUI sample renders the Cornell frame interactively with backend toggles, resolution controls, and screenshot capture.
- Documentation updated with run instructions and troubleshooting (shader paths, device requirements) plus frame/hash expectations.
**Tests**: Manual checklist for on-screen demo, smoke run capturing output hash, optional unit hooks for window lifecycle.
**Status**: Complete — MetalKit UI with mode/resolution/screenshot controls shipped; docs + 1024×768 hash recorded

### Stage 3D: Extended Shading & Effects
**Goal**: Build on the hardware kernel to introduce polished effects (reflections, refraction, soft shadows, motion blur, etc.). The fallback/software renderer is paused until Stage 4, so this milestone is hardware-only.
**Current Status**:
The Stage 3D shader port is partially in place: primary shading, BRDF sampling, and TLAS setup now mirror the Apple reference, and the hardware ray kernel consumes true per-instance TLAS data (instance IDs map back to packed scene buffers so Mario/Cornell transforms and per-triangle materials survive the upload). However, the render graph still copies `shadeTexture` to the drawable before the shadow/accumulate passes run, so queued lighting never reaches the screen. Secondary bounces also fall back to the CPU `traceScene` helper instead of issuing real hardware rays, which keeps reflection/refraction black. The next iteration must address both shortcomings (render graph ordering + hardware-driven secondary rays) before we can claim Stage 3D parity.
**Success Criteria**:
- Material system supports reflective/refractive parameters with documented presets.
- Shadowing, indirect bounce approximations, and motion blur toggles wired through CLI/on-screen samples.
- Regression assets include updated frame hashes per effect.
**Tests**: Expanded image-diff suite, stress tests for parameter toggles, targeted unit tests for material packing.
**Status**: In Progress — TLAS build now flattens the scene into a single hardware mesh, but the on-screen frame still diverges from Apple's reference (Mario scale/placement and per-triangle materials/UVs). Upcoming work:

- Mirror the CPU reference scene (`~/Desktop/MetalRayTracing`) triangle-for-triangle so our Cornell builder uses the exact same transforms/materials.
- Rebuild the hardware acceleration structures the way the Metal RT sample does (`~/Desktop/metal_RTR_official_example`): BLAS per mesh plus TLAS instances with correct masks/material bindings.
- Prepare the same material/texture buffers as the reference and rewrite `RTRRayTracing.metal` so all ray queries (primary/shadow/reflection) use the hardware intersection path, matching Apple's shading code.
- Validate frame dumps + hashes for each shading tweak; the next fallback work remains blocked until this is green.

## Stage 4: Software RT & Fallback Return
**Goal**: Reintroduce the software/MPS fallback once the hardware renderer reaches parity. Mirror the resource layouts from Stage 3 so both backends share scene packing and produce deterministic outputs for regression tests.
**Success Criteria**: Software path renders the Cornell reference image, hashes recorded for both backends, CLI/UI toggles expose hardware vs fallback, docs outline hardware requirements plus fallback caveats.
**Tests**: Expanded hash/image diff suite running both backends, profiling scripts, documentation linting.
**Status**: Not Started — blocked on Stage 3D completion.
