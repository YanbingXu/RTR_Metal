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

## Stage 3: Ray Tracing Pipeline
**Goal**: Build acceleration structure builders, shader binding tables, ray generation/miss/closest-hit shaders, and per-frame rendering loop.
**Success Criteria**: Engine traces rays against test geometry producing rendered image to texture; validation frame dump matches tolerance; tests cover AS builders.
**Tests**: Offline render test comparing checksum; unit tests for TLAS/BLAS builders.
**Status**: In Progress

### Stage 3 Sub-plan: Dual Ray Tracing Backends

#### Stage 3A: Native Metal Ray Tracing Path
**Goal**: Complete TLAS build, ray tracing pipeline state, SBT, and render dispatch using `MTLRayTracingPipelineState`.
**Success Criteria**: On SDKs exposing `MetalRayTracing.h`, diagnostic scene renders via hardware RT; surface logs confirm TLAS/dispatch.
**Tests**: Feature-flagged integration test that verifies BLAS/TLAS creation under ray-tracing-capable SDK; renderer smoke test logs.
**Status**: In Progress

#### Stage 3B: MPS Path Tracing Path
**Goal**: Integrate Metal Performance Shaders path tracer (e.g., `MPSRayIntersector`/`MPSPathTracingSample`) sharing scene/resource infrastructure.
**Success Criteria**: MPS backend renders diagnostic scene deterministically; API surface allows switching between native + MPS backends.
**Tests**: Unit tests for backend selection/initialization; offline render comparison for MPS output.
**Status**: In Progress

#### Stage 3C: Demo Applications
**Goal**: Provide two runnable demos—one using native RT (when available), one using MPS fallback—with CLI or simple UI to choose backend.
**Success Criteria**: Both demos build/run via CMake; README documents requirements and selection; logs/screenshots captured.
**Tests**: Manual verification on hardware; scripted build/run checks for each demo target.
**Status**: Not Started

## Stage 4: Sample Application & Documentation
**Goal**: Deliver macOS sample app demonstrating engine with interactive controls, plus developer docs and usage instructions.
**Success Criteria**: Sample window renders scene with camera controls; README updated; docs explain architecture and build steps.
**Tests**: Manual run of sample; doc lint/checklist; integration test verifying sample executable launches.
**Status**: Not Started
