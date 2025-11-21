# MPSFallback Engine Progress (Plain-English Walkthrough)

> **Status (2025-11-06)**: The software/MPS fallback (including the `RTRMetalMPSSample` target) has been archived until Stage 4 so that Stage 3D can focus entirely on hardware ray tracing. The notes below remain for historical reference only.
> **Audience**: Someone who knows a little C++ and has heard of ray tracing, but hasn’t built a renderer and doesn’t know Metal.

## 1. Project Context
- We’re building a ray-tracing engine for macOS that normally talks to Apple’s hardware ray-tracing APIs (Metal+RT).
- Because not every Mac exposes those shiny APIs, we also ship a *fallback* using **Metal Performance Shaders (MPS)**. Think of MPS as Apple’s library of GPU helpers. We call this fallback the “MPS path.”
- Everything we add here lives inside the `RTRMetalEngine` library, so both the CLI sample and future app UI reuse the same code.

## 2. What We Have So Far
### 2.1 Scene plumbing (CPU side)
1. **Scene data**: Meshes, materials, and instances live in pure C++ structs (`scene::Scene`, `scene::Mesh`, etc.). The SceneBuilder helper turns raw vertex arrays into nice meshes.
2. **GeometryStore**: Because the GPU can’t read C++ vectors directly, we upload mesh data into Metal buffers (`MTLBuffer`). GeometryStore owns those GPU buffers and keeps them alive for the whole renderer.
3. **MPSSceneConverter**: Converts the Scene into simple arrays (positions, indices, colours). We still need those arrays because the CPU fallback shades pixels and because MPS requires “plain” data to build its acceleration structures.

### 2.2 MPSPathTracer (GPU scaffolding)
- Metal needs a “device” (the GPU), a “command queue,” and a ray-tracing acceleration structure (basically a fast way to query triangles). `MPSPathTracer` wraps those bits:
  1. grabs the system Metal device;
  2. creates a command queue;
  3. asks MPS for an intersector (the thing that will shoot rays at triangles);
  4. builds an acceleration structure from our mesh data.
- Important: Until now, `uploadScene` accepted CPU arrays, created Metal buffers, and stored copies. This week we simplified it so it **only creates Metal buffers**. The renderer keeps the CPU copies it needs separately.

### 2.3 MPSRenderer (orchestrator)
1. During `initialize(scene)`, we call the PathTracer and SceneConverter so the GPU and CPU are ready.
2. For the CLI sample we also build a small two-mesh scene (a floor and a prism). This is just for demonstration.
3. For each frame, we:
   - Create a ray buffer and intersection buffer.
   - Manually fill the ray buffer with rays firing from `{0,0,1.5}` toward a virtual screen.
   - Ask MPS to intersect those rays with our triangles.
   - Pull down the intersection data and run a CPU loop that shades each pixel (Lambert lighting + barycentric colour).
   - Write out a PPM image.

**Why CPU shading?** It’s the quickest way to verify the geometry pipeline without diving into Metal compute shaders. But it’s slow and doesn’t use the GPU beyond the intersection step.

### 2.4 Tests + Docs
- Added GoogleTests to check that SceneConverter handles multiple meshes and index offsets.
- Wrote `docs/mps_stage3_status.md` for progress tracking and `docs/mps_gpu_pipeline_plan.md` as our GPU shading roadmap.

## 3. The Implementation Method
1. **Match data flow**: Everything still starts from the Scene. The GeometryStore uploads the same data the future hardware path will use. This keeps our fallback and primary renderer in sync.
2. **Wrap platform APIs**: Metal context creation, buffers, and intersectors live behind C++ classes. Think RAII wrappers that keep ownership clear.
3. **Fallback shading**: We intentionally wrote a boring CPU shading loop. Yes, it’s slow, but it’s also transparent and testable. Perfect for the early milestone.
4. **Incremental upgrades**: After the basic frame worked, we refactored to avoid duplicate CPU copies and added camera uniform buffers. These small steps lead us toward the GPU compute pipeline.

## 4. Roadmap (Where We’re Going)
1. **Uniform buffers**: Already added `MPSCameraUniforms` and a Metal buffer to hold camera data. This buffer will feed the upcoming ray-generation kernel.
2. **GPU compute pipeline**: We’ll introduce Metal compute shaders (kernels) that run these stages:
   - ray generation
   - shading
   - (optional) shadow rays
   - accumulation (averaging multiple samples)
3. **On-screen sample**: After compute kernels work off-screen, we’ll hook the same pipeline into an `MTKView` window so you can watch the image converge live.
4. **Off-screen regression tools**: Keep a CLI path that writes images, plus add hash tests so CI can catch regressions.

## 5. Big Picture Analogy
- Think of this like building a remote-control car (the hardware path) while also assembling a cardboard prototype (the fallback). The cardboard version must share the same steering mechanism so the drivers learn once and switch easily.
- We’re in the stage where the cardboard car can roll downhill (CPU shading). Next we’ll add a small motor (GPU compute kernels) using the same steering (shared buffers) so both cars behave the same.

## 6. Glossary
- **GPU buffer (`MTLBuffer`)**: Metal’s way of storing data on the graphics card.
- **Acceleration structure**: A data structure that speeds up ray/triangle intersection tests. In MPS we use `MPSTriangleAccelerationStructure`.
- **MPSRayIntersector**: The MPS helper that fires rays at the acceleration structure and returns intersections.
- **Lambert shading**: Shading based on the angle between the surface normal and the light direction.
- **Barycentric coordinates**: The weights that tell us how much each vertex of a triangle contributes to a point inside that triangle. Perfect for colour interpolation.
- **Metal compute kernel**: A GPU function (written in Metal Shading Language) that runs in parallel on many threads.

## 7. How to Verify Things Yourself
1. Build: `cmake --build build`
2. Tests: `ctest --test-dir build`
3. Run sample: `build/RTRMetalMPSSample`
   - Output image: `mps_output.ppm` (view with Preview app or convert to PNG).
4. Where to look in the code:
   - Scene setup: `engine/src/MPS/MPSRenderer.mm`
   - Path tracer core: `engine/src/MPS/MPSPathTracer.mm`
   - CPU shader: bottom half of `MPSRenderer::renderFrame`

## 8. What’s Next (Concrete Action Items)
- Implement the ray-generation Metal kernel and pipeline bindings.
- Feed our new uniform buffer into the kernel.
- Keep the CPU shading path as a fallback/test mode until GPU shading is battle-tested.

Remember: you don’t need to be a Metal wizard to follow along. We’re isolating Metal-specific bits so you can focus on the data flow and high-level renderer architecture.
