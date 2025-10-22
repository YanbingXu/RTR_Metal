# Stage 3B – MPS Renderer Progress Report

## Current Status
- **MPSPathTracer refactor** – Device and intersector setup now happens independently from scene upload, letting us validate hardware support and log failures clearly. `uploadTriangleScene` handles buffer creation, acceleration-structure rebuilds, and stores CPU copies for shading.
- **Scene conversion pipeline** – `MPSSceneConverter` walks the engine `Scene`, validates geometry, and generates position/index/color arrays for the tracer. When materials are monochrome, a simple RGB palette is assigned per-vertex so diagnostic renders show obvious barycentric shading.
- **Renderer shading pass** – `MPSRenderer` dispatches the `MPSRayIntersector` for a camera facing the X/Y plane, shades hits on the CPU (Lambert + barycentric mix), and writes a 512×512 PPM. Rays now originate from `{0,0,1.5}` and march toward each pixel’s projection, yielding a correctly framed triangle.
- **Sample app** – `RTRMetalMPSSample` initialises the tracer, runs conversions, and emits `mps_output.ppm`. On RT-capable hardware the output is a coloured triangle on a dark background, matching the diagnostic goal.
- **Test coverage** – Added `MPSSceneConverterTests` to check colour assignment and index packing. Full suite passes locally (21 tests, 1 intentional skip for unsupported hardware).

## How the Triangle Is Rendered
1. **Scene build** – The sample scene constructs one triangle with per-vertex colours supplied by `MPSSceneConverter` (either from material albedo or the fallback palette).
2. **Ray generation** – `MPSRenderer` creates a shared ray buffer of type `MPSRayOriginMaskDirectionMaxDistance`. Each pixel computes its `(u,v)` location on the image plane, sets the origin to `{0,0,1.5}`, and normalises the `(targetPoint - origin)` direction.
3. **Intersection** – `MPSRayIntersector` consumes the ray buffer and the triangle acceleration structure; hits populate an intersection buffer with distance, primitive index, and barycentric coordinates.
4. **CPU shading** – For each pixel we fetch the intersection record. If the primitive index is valid, we reconstruct the triangle normal, apply a directional light, and mix the three vertex colours using the barycentric weights `(w,u,v)`.
5. **Image write-back** – Colour values are clamped to `[0,1]`, converted to 8-bit RGB, and the frame is saved as `mps_output.ppm`. Pixels without hits fall back to the ambient background colour.

## CPU Shading vs. GPU Shading
- **Current CPU shading** – All shading work happens on the CPU after intersections are computed. This keeps the diagnostic path simple: no GPU compute pipelines are required, and we can inspect buffers directly. The trade-off is performance; every pixel is processed serially on the host.
- **Target GPU shading** – The reference sample pipelines (`rayKernel`, `shadeKernel`, `shadowKernel`, `accumulateKernel`) execute ray generation, shading, and accumulation entirely on the GPU. Porting to that model will:
  - Remove CPU-side loops and buffer read-backs.
  - Enable multi-bounce lighting, accumulation, and shadows without large host cost.
  - Align the fallback path with the ultimate renderer architecture where per-frame work happens on the GPU.

## Next Steps
1. **Integrate real engine geometry** – Feed meshes uploaded via `GeometryStore` into `MPSSceneConverter` so the fallback path can render arbitrary scenes, not just the built-in triangle.
2. **Port shading to GPU** – Recreate the reference compute pipeline within `RTRMetalEngine`: generate rays, shade, cast shadows, and accumulate entirely on the GPU. Reuse `MPSRayIntersector` buffers for interoperability.
3. **Image verification & tooling** – Add lightweight image-difference tooling (hash or checksum) under `tests/` to catch regressions once deterministic outputs are available.
4. **Documentation & sample UX** – Update the README with MPS usage instructions, include rendered output samples, and expose simple CLI options (resolution, camera distance) for debugging.
