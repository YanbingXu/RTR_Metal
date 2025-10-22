## Stage 1: Project Shell & Tooling
**Goal**: Replace Swift package with CMake-based C++ project scaffold, shared engine/static library target, sample app target, and shader build integration stubs.
**Success Criteria**: CMake config generates build files; `metal` shader compilation hooks exist; placeholder engine library builds and links into sample.
**Tests**: Configure and build via `cmake --build` on macOS; run placeholder unit test binary.
**Status**: Complete

## Stage 2: Core Engine Infrastructure
**Goal**: Implement Metal device/context wrapper, memory/resource managers, logging, configuration loading, and math utilities needed by renderer.
**Success Criteria**: Engine initializes Metal device & command queues; can load mesh data and upload buffers; unit tests cover math/config modules.
**Tests**: New C++ unit tests for math/utilities; run sample in headless init mode validating context creation.
**Status**: In Progress

## Stage 3: Ray Tracing Pipeline
**Goal**: Build acceleration structure builders, shader binding tables, ray generation/miss/closest-hit shaders, and per-frame rendering loop.
**Success Criteria**: Engine traces rays against test geometry producing rendered image to texture; validation frame dump matches tolerance; tests cover AS builders.
**Tests**: Offline render test comparing checksum; unit tests for TLAS/BLAS builders.
**Status**: Not Started

## Stage 4: Sample Application & Documentation
**Goal**: Deliver macOS sample app demonstrating engine with interactive controls, plus developer docs and usage instructions.
**Success Criteria**: Sample window renders scene with camera controls; README updated; docs explain architecture and build steps.
**Tests**: Manual run of sample; doc lint/checklist; integration test verifying sample executable launches.
**Status**: Not Started
