# MPS GPU Compute Pipeline Plan

> **Status (2025-11-06)**: Deferred. The hardware pipeline owns Stage 3D; the software/MPS roadmap below (and its former targets such as `RTRMetalMPSSample`) is archived until Stage 4 resumes fallback work.

## Goals
- Replace the current CPU shading loop in `MPSRenderer` with Metal compute kernels so the fallback path mirrors the production renderer architecture.
- Share buffers between ray generation, intersection, shading, and accumulation to minimise copies and ease future integration with the native RT path.
- Preserve a deterministic off-screen path for regression tests while allowing the same kernels to drive the upcoming on-screen sample.

## Proposed Pipeline

1. **Ray Generation Kernel (`rayKernel`)**
   - Compute per-pixel primary rays given camera parameters (origin, orientation, projection matrices).
   - Populate an `MPSRayOriginMaskDirectionMaxDistance` buffer (shared with `MPSRayIntersector`).
   - Inputs: camera uniforms, resolution, random seeds.
   - Outputs: ray buffer, optional payload buffer (e.g. sample index).

2. **Intersection Pass (MPS)**
   - Reuse `MPSRayIntersector` to populate the intersection buffer.
   - No kernel work; ensure command buffer sequencing lets shading kernel read results directly (use `MTLBlitCommandEncoder` barriers if required).

3. **Shading Kernel (`shadeKernel`)**
   - Consume intersection data, triangle buffers (positions, normals, materials), and camera/light uniforms.
   - Perform direct lighting (Lambert + optional specular), barycentric colour interpolation, and shadow-ray spawning when enabled.
   - Write results to a 16-bit float RGBA render target, plus accumulate emittance and throughput for multi-sample scenarios.

4. **Shadow Ray Pass (optional)**
   - If soft shadows are desired, launch a second intersector dispatch with shadow rays generated in `shadeKernel`.
   - Encode results into a visibility buffer consumed by a subsequent shading kernel or in-kernel loop.

5. **Accumulation Kernel (`accumulateKernel`)**
   - Combine the current frame with the accumulation history (tonemapping + exposure) when progressive rendering is enabled.
   - For off-screen tests, allow a single-sample path that writes directly to the PPM buffer after conversion to 8-bit RGB.

6. **Presentation / Read-back**
   - Off-screen: blit the final texture into a staging buffer, map to CPU, and write PNG/PPM.
   - On-screen: bind the texture to an `MTKView` render pass with a full-screen quad or texture-to-drawable blit.

## Buffer & Resource Layout

| Resource | Usage | Notes |
| --- | --- | --- |
| `RayBuffer` (`MTLStorageModePrivate`) | Ray gen → intersector | Structured as `MPSRayOriginMaskDirectionMaxDistance`.
| `IntersectionBuffer` (`MTLStorageModePrivate`) | Intersector → shading | `MPSIntersectionDistancePrimitiveIndexCoordinates`.
| `VertexBuffer`, `IndexBuffer` | Geometry | Reuse `GeometryStore` Metal buffers; keep CPU copies only for diagnostic unit tests.
| `MaterialBuffer` | Shading params | Array of packed material structs (color, roughness, flags).
| `UniformBuffer` | Camera + frame data | Triple-buffered for on-screen path.
| `OutputTexture` | Float RGBA | Source for accumulation and tone mapping.
| `AccumulationTexture` (optional) | Progressive rendering | Alternate between ping/pong targets.
| `RandomBuffer` (optional) | RNG seeds | Per-pixel random state for anti-aliasing or soft shadows.

## Integration Steps

1. **Uniform Struct Definition**
   - Add a shared Metal/Swift/C++ header describing camera and lighting uniforms.
   - Extend `Renderer` to update the buffer each frame.

2. **Metal Kernel Authoring**
   - Port `rayKernel`, `shadeKernel`, `shadowKernel`, `accumulateKernel` from the reference sample, adjusting bindings to our buffer IDs.
   - Place kernels in `shaders/RTRRayTracing.metal` under conditional compilation so they can compile without native RT support.

3. **Command Encoding Sequence**
   - `rayEncoder.dispatchThreads`
   - `intersector.encodeIntersection`
   - `shadeEncoder.dispatchThreads`
   - Optional shadow/intersection loop
   - `accumulateEncoder.dispatchThreads`
   - `blitEncoder.copyFromTexture` (off-screen) or present (on-screen)

4. **Resource Lifetime**
   - Promote buffers/textures to members of `MPSRenderer` to avoid per-frame recreation.
   - Add resize handling for the on-screen sample (respond to `MTKView` size changes).

5. **Testing Strategy**
   - Headless checksum test: render a deterministic frame with fixed seeds and compare against a stored hash.
   - Shader unit tests via Metal function constants (if feasible) or by executing kernels on a small buffer and verifying memory contents.

6. **Fallback Path**
   - Retain the CPU shading routine under a compile-time or runtime flag for CI environments lacking MPS support.
   - Share the scene conversion output (CPU vectors) between both paths to minimise drift.

## Milestone Breakdown

1. **Milestone A – Kernel bootstrap**
   - Implement uniforms, create ray and shading pipelines, hard-code a single bounce without accumulation.
   - Validate on off-screen sample and log energy statistics.

2. **Milestone B – Shadows & accumulation**
   - Introduce shadow rays, accumulation textures, and simple tone mapping.
   - Add CLI flags to toggle features for regression tests.

3. **Milestone C – On-screen integration**
   - Hook the pipelines into an `MTKView`, add frame timing HUD/logging, and ensure progressive accumulation resets on camera changes.
   - Deliver screenshots for documentation.

4. **Milestone D – QA & Tooling**
   - Add checksum/image-diff tests.
   - Update docs/README with instructions, capabilities, and GPU requirements.

## Open Questions
- How much of the reference sample’s material system (textures, reflection parameters) should be ported before Stage 3 completion?
- Do we need a generic buffer abstraction so the native RT path and MPS path share the same shader-binding layout?
- What tolerance should we use for checksum/image-diff tests to account for platform FP variance?
