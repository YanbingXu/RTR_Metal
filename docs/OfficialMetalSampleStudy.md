# Study Notes: Metal Ray Tracing Samples

This document captures how each of the official Metal samples in `~/Desktop/metal_RTR_official_example` implements ray tracing or reflection effects, and outlines takeaways for the `RTR_Metal` project.

## Accelerating Ray Tracing Using Metal
- **Goal & Flow**: Builds a minimal path tracer using GPU parallelism. Acceleration structures are assembled on the CPU, then a compute kernel (`raytracingKernel`) performs traversal, shading, and accumulation. Optional custom intersection functions cover non-triangular geometry.
- **Pipeline Type**: `MTLComputePipelineState` with linked functions. Dispatch runs via a compute command encoder.
- **Ray Tracing Mode**: Compute-based ray tracing; relies on intersection function tables and acceleration structures, but not on a dedicated ray tracing command encoder.

## Accelerating Ray Tracing and Motion Blur Using Metal
- **Goal & Flow**: Extends the path tracer to handle per-instance and per-primitive motion blur. It rebuilds TLAS/BLAS with motion transforms and feeds them into a compute ray tracing kernel (either `raytracingInstanceMotionKernel` or `raytracingInstanceAndPrimitiveMotionKernel`).
- **Pipeline Type**: Compute pipeline; MLAA-style kernels handle motion data.
- **Ray Tracing Mode**: Compute-based ray tracing with motion blur data encoded in acceleration structures and uniform buffers.

## Control the Ray Tracing Process Using Intersection Queries
- **Goal & Flow**: Demonstrates intersection query control by using visible function tables instead of intersection function tables, exposing explicit management of procedural geometry hits. Traversal and shading still happen inside a compute kernel.
- **Pipeline Type**: Compute pipeline with linked functions; uses `MTLVisibleFunctionTable` for procedural control.
- **Ray Tracing Mode**: Compute-based ray tracing with manual intersection query handling.

## Rendering a Curve Primitive in a Ray Tracing Scene
- **Goal & Flow**: Shows how to mix triangle and curve primitives. Custom intersection functions (for triangles and Catmull–Rom curves) are linked into the main compute ray tracing kernel, and their resources are bound via an intersection function table.
- **Pipeline Type**: Compute pipeline with intersection function table.
- **Ray Tracing Mode**: Compute-based ray tracing; procedural curve hits are handled entirely in linked intersection functions.

## Rendering Reflections in Real Time Using Ray Tracing
- **Goal & Flow**: Implements a hybrid renderer. It rasterizes a thin G-buffer to capture position and direction, then runs a compute kernel to shoot reflection rays through an acceleration structure. The reflections feed back into the forward render pass and post-processing.
- **Pipeline Type**: Combination of raster render pipelines plus a compute pipeline (`_rtReflectionPipeline`) for ray traced reflections. Uses `setAccelerationStructure:` on the compute encoder.
- **Ray Tracing Mode**: Compute-based ray tracing for reflections; no dedicated ray tracing command encoder is used.

## Key Takeaways for `RTR_Metal`
1. **Expect Compute Pipelines Today**: All official samples route ray tracing work through `MTLComputePipelineState` plus acceleration-structure-aware encoders. Public SDKs currently expose traversal/intersection via compute, not via standalone `MTLRayTracingCommandEncoder` APIs.
2. **Leverage Linked/Visible Function Tables**: Intersection and procedural handling is achieved by linking additional Metal functions and binding them through intersection or visible function tables, not through shader binding tables in the DXR sense.
3. **Structure Resources Like the Samples**:
   - Maintain resource buffers that pack per-geometry pointers/strides, so intersection functions can fetch data directly.
   - Use per-frame uniform ring buffers, accumulation targets, and random textures for stochastic integration.
   - Bind TLAS/BLAS with `setAccelerationStructure:` when dispatching compute kernels.
4. **Fallback Paths Remain Valuable**: Samples always retain raster-based or simplified modes (clear fallback, reflections-only, no RT). Our project should keep a non-RT or simplified mode for unsupported hardware.

## Suggested Modifications for `RTR_Metal`
- **Adopt Compute-Based Ray Tracing Pipeline**: Replace the placeholder hardware ray tracing dispatch with a compute kernel approach mirroring these samples—build BLAS/TLAS, create a specialized ray tracing compute function (with `MTLLinkedFunctions` as needed), and dispatch via `MTLComputeCommandEncoder`.
- **Use Intersection/Visible Function Tables**: Introduce intersection tables for procedural geometry and visible function tables for dynamic control, matching the official patterns.
- **Resource Layout Refactor**: Implement resource buffers (geometry/material pointers) and per-frame uniform buffers. Accumulation targets and random textures should reside alongside TLAS to enable iterative path tracing.
- **Maintain Graceful Fallbacks**: Keep the gradient or raster fallback for environments lacking acceleration structure support, but gate feature probing on `MTLDevice.supportsRaytracing` instead of non-existent headers.
- **Plan for Hybrid Rendering**: When integrating with the example application, consider the hybrid flow used in the reflections sample: render a thin G-buffer, dispatch compute for ray traced effects, then composite in the main render pass.

Together, these changes align `RTR_Metal` with the currently documented, publicly available Metal ray tracing workflow and provide a concrete path forward even before Apple exposes dedicated ray tracing command encoders.

---

## Reference Notes & Discussion Summary

### Official Guidance
- Apple documentation: [Accelerating ray tracing using Metal](https://developer.apple.com/documentation/metal/accelerating-ray-tracing-using-metal?language=objc)
  - Describes how to build `MTLAccelerationStructure` objects and issue ray traversal through compute kernels.
  - Highlights that BLAS/TLAS construction and traversal run on dedicated hardware once you wire acceleration structures into a compute pipeline via `setAccelerationStructure:`.

### Summary of Our Findings
1. **Public SDK Workflow**
   - The current Xcode 16.1 + macOS 15.5 SDK exposes acceleration-structure APIs but not a dedicated `MTLRayTracingCommandEncoder`.
   - Ray traversal is performed in compute kernels; the GPU’s hardware units handle BVH traversal/intersection once you supply TLAS/BLAS.
   - Compilers require `#define MTL_ENABLE_RAYTRACING 1` before importing `<Metal/Metal.h>`.
2. **Sample Code Insights**
   - All five official samples (WWDC20/22) are Objective‑C++/C++ + Metal shader projects. None use Swift or a specialized ray tracing command encoder.
   - Each sample builds TLAS/BLAS, then dispatches work through `MTLComputePipelineState` with linked functions/intersection tables.
   - Hybrid scenarios (e.g., “Rendering Reflections in Real Time Using Ray Tracing”) render thin G-buffers via rasterization, then fire reflection rays in a compute pass, bind TLAS with `setAccelerationStructure:`, and composite results back into the forward render pipeline.
3. **Implications for RTR_Metal**
   - Transition our placeholder ray tracing path to match the compute-pipeline pattern: build acceleration structures, create a specialized ray tracing kernel, dispatch via `MTLComputeCommandEncoder`.
   - Introduce resource buffers, intersection/visible function tables, and per-frame accumulation textures mirroring Apple’s structure.
   - Keep the fallback pipeline (e.g., gradient or raster mode) gated on `MTLDevice.supportsRaytracing`.
   - Plan for hybrid rendering (thin G-buffer + compute ray tracing) when integrating reflections or other effects.

### Environment Observations
- macOS 15.5 + Xcode 16.1 + Apple Silicon (M4 Pro) still reports `__has_include(<Metal/MetalRayTracing.h>) == false`; the SDK folds ray tracing extensions into existing headers.
- Objective‑C runtime inspection (`NSClassFromString`) shows ray tracing classes like `MTLRayTracingPassDescriptor` aren’t publicly visible. This aligns with the compute-only flow.
- `strings` on `Metal.tbd` reveals private symbols (`_MTLSWRaytracing…`), further suggesting the public API layer is centered on compute kernels + acceleration structures.

### Next Steps
- Replace `dispatchRayTracingPass()` stub with compute-based implementation using `setAccelerationStructure:` and a ray tracing kernel (refer to “Accelerating Ray Tracing Using Metal”).
- Refactor renderer resource management (uniform ring buffers, accumulation textures, resource buffer strides) to align with the sample architecture.
- Maintain compute fallback paths for unsupported devices while instrumenting `supportsRaytracing` checks.

## Final Goals & Execution Plan (Quick Reference)
- **Engine Goals**
  - Reusable Metal ray tracing engine on macOS 14+ / Apple Silicon using TLAS/BLAS + compute kernels (hardware traversal via `setAccelerationStructure:`) with MPS fallback.
  - Off-screen CLI + on-screen demo rendering Cornell Box–class scenes with reflections/shadows/refraction and automated validation (hash/logs).
  - Unified resource/shading architecture across backends (scene upload, geometry/material buffers, intersection/visible function tables, accumulation/random textures).

- **Development Plan**
  1. *Compute Ray Tracing Pipeline* – implement `raytracingKernel` (with linked/visible functions) and bind TLAS/BLAS on the compute encoder.
  2. *Resources & Buffers* – add per-frame uniform ring buffer, resource pointer buffers, accumulation/random textures; refactor renderer to consume them.
  3. *Procedural Geometry & Extensions* – support custom primitives through intersection/visible function tables; align MPS/compute resource layout.
  4. *Hybrid Rendering & Demos* – build thin G-buffer + compute RT flow for CLI & MetalKit/SwiftUI demos; expose toggles, screenshot/PPM, accumulation controls.
  5. *Testing & Docs* – extend `ctest` (TLAS, resources, image hashes) and update README/docs with hardware requirements, fallback behaviour, validation guidance.

- **Immediate Focus**
  - Implement compute-based ray tracing pipeline & resource structures and replace `dispatchRayTracingPass()` stub before layering additional features.

## Updated Strategy for `RTR_Metal`

### Current Understanding
- Public Metal SDKs (macOS 15+/Xcode 16+) expose hardware-accelerated ray tracing through acceleration-structure APIs combined with compute kernels; there is no separate ray tracing command encoder.
- Official Apple samples all employ `MTLComputePipelineState` plus BLAS/TLAS, intersection/visible function tables, resource buffers, and accumulation targets to realize hardware ray tracing.
- Our environment (macOS 15.5 + Xcode 16.1 + M4 Pro) supports `MTLDevice.supportsRaytracing`, so compute-based ray tracing with hardware traversal is available immediately.

### Revised Project Goals (Stage 3 Focus)
1. **Compute Ray Tracing Pipeline**: Implement the hardware-accelerated path using a dedicated compute kernel (`raytracingKernel`), linked intersection functions, and TLAS/BLAS integration.
2. **Resource & Buffer Architecture**: Mirror the sample structure—per-frame uniform ring buffers, resource pointer buffers, accumulation textures, random sequences, and reusable staging buffers.
3. **Procedural Geometry Support**: Introduce intersection or visible function tables to handle custom primitives and intersection queries.
4. **Hybrid Rendering & UI Integration**: Provide a thin G-buffer + compute ray tracing workflow for reflections/path tracing, then composite into the renderer/view model (keeping fallback/raster mode).
5. **Robust Fallback Mode**: When `supportsRaytracing` is false, fall back to simplified compute shading (current gradient or future CPU-only path) without touching acceleration structures.

### Implementation Plan
1. **Pipeline Foundation**
   - Define ray tracing shader entry points (`raytracingKernel`, optional intersection functions) and specialize constants.
   - Create compute pipeline with `MTLLinkedFunctions` when procedural geometry is present.
   - Implement TLAS/BLAS upload/build flows that feed the kernel (`setAccelerationStructure:`).
2. **Resource Layout**
   - Introduce per-frame uniform buffers, resource pointer buffers, accumulation textures, and random textures (aligned with sample code patterns).
   - Encapsulate scene geometry/material uploads similar to Apple examples for predictable shader access.
3. **Dispatch Path**
   - Replace `dispatchRayTracingPass()` stub with a compute command encoder that binds TLAS, resources, uniforms, and accumulation targets; dispatch threadgroups matching output resolution.
   - Integrate fallback compute gradient path as a secondary branch, selected when capabilities/pipelines are unavailable.
4. **Hybrid Rendering (Optional Step after core path)**
   - Prepare a thin G-buffer render pass followed by compute ray tracing for reflections/path tracing, then composite results in the main render pass.
5. **Testing & Validation**
   - Add unit/integration tests for TLAS build, resource buffer packing, and compute dispatch readiness.
   - Provide manual validation steps (captured images, logs) until automated image diffs are in place.

### Immediate Next Actions
1. Refactor renderer setup to create a ray tracing compute pipeline and ray tracing kernel shaders.
2. Restructure acceleration structure build/upload to match the sample resource layout and expose resource buffers to shaders.
3. Replace `dispatchRayTracingPass()` with compute dispatch logic (including accumulation target management).
4. Preserve and polish the fallback gradient path for unsupported devices (post-refactor).
