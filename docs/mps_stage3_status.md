# Stage 3B – MPS Renderer Progress Report

## Current Status
- **MPSPathTracer refactor** – Device and intersector setup now happens independently from scene upload, letting us validate hardware support and log failures clearly. `uploadTriangleScene` handles buffer creation, acceleration-structure rebuilds, and stores CPU copies for shading.
- **Scene conversion pipeline** – `MPSSceneConverter` walks the engine `Scene`, validates geometry, and generates position/index/color arrays for the tracer. When materials are monochrome, a simple RGB palette is assigned per-vertex so diagnostic renders show obvious barycentric shading.
- **Renderer shading pass** – `MPSRenderer` dispatches the `MPSRayIntersector` for a camera facing the X/Y plane, shades hits on the CPU (Lambert + barycentric mix), and writes a 512×512 PPM. Rays now originate from `{0,0,1.5}` and march toward each pixel’s projection, yielding a correctly framed triangle that now sits above a lit floor.
- **GeometryStore integration** – The renderer uploads every mesh in the input scene through `GeometryStore`, keeping GPU-side buffers alive alongside the diagnostic float3 copies. This mirrors the core engine’s resource flow and unlocks rendering scenes built elsewhere in the engine without rewriting geometry ingestion.
- **Sample app** – `RTRMetalMPSSample` initialises the tracer with a simple scene composed of a ground plane and a raised prism, then emits `mps_output.ppm`. On RT-capable hardware the output shows the prism’s coloured face against the floor, matching the diagnostic goal.
- **Sample app** – `RTRMetalMPSSample` now accepts CLI flags to choose shading mode (`--cpu`, `--gpu`) and can emit a CPU/GPU comparison pair (`--compare`),报告最大差异；同时 `config/engine.ini` 可以通过 `shadingMode=auto|cpu|gpu` 设置默认偏好，CLI 参数会覆盖该值；`--reset-accum` 会清空 GPU 累积缓冲。
- **Test coverage** – Added `MPSSceneConverterTests` to check colour assignment and index packing. Full suite passes locally (21 tests, 1 intentional skip for unsupported hardware).
- **GPU compute plan** – See `docs/mps_gpu_pipeline_plan.md` for the staged strategy that transitions the fallback path from CPU shading to Metal compute kernels while preserving deterministic off-screen outputs.

## How the Triangle Is Rendered
1. **Scene build** – The sample scene constructs a ground plane and a prism face; per-vertex colours come from the scene materials or the fallback palette supplied by `MPSSceneConverter`.
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
1. **Integrate real engine geometry** – Ingest Cornell Box assets, upload via `GeometryStore`, and allow scene switching in the samples.
2. **Port shading to GPU** – Extend the compute pipeline with shadow rays and multi-sample accumulation（现已具备基础平均流程，下一步加入阴影与多样本控制）。
3. **On-screen vs. off-screen samples** – 保留 CLI 回归通道，同时引入 MetalKit/SwiftUI 上屏 Demo，统一 shading/accumulation 控制并记录参考截图。
4. **Image verification & tooling** – 通过 hash/截图建立回归基线，在支持设备上验证多帧累积输出。
5. **Documentation & sample UX** – README/docs 补充 shadingMode、accumulation、scene selector 说明，并记录真机回归数据。

## Stage 3C Deliverables

### Off-screen Sample (CLI)
- Produce deterministic image output (`ppm` or `png`) via the GPU shading path, including optional accumulation controls.
- Provide flags for resolution, sample count, output format, and backend selection (`mps` vs. future native RT).
- Emit checksums/hashes for automation and integrate with the planned image-diff test harness.

### On-screen Sample (AppKit/MetalKit)
- Present progressive frames in an `MTKView`, with camera orbit controls and frame statistics overlay.
- Allow runtime toggles for accumulation reset, sample count, and screenshot capture to the off-screen pipeline.
- Share renderer infrastructure and resource ownership with the CLI path; both should reuse `MPSRenderer` setup with minimal divergence.

### Acceptance Criteria
- Both samples run from the same build (flags or runtime switches) and report meaningful errors on hardware lacking MPS support.
- Rendering artifacts checked into documentation (screenshots + explanation of expected output).
- Automated validation covers the off-screen path; manual verification checklist established for the on-screen demo until automation is feasible.
