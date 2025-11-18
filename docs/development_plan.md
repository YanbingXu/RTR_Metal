# Development Plan

## Milestone 1 – Project Bootstrap (✅)
1. Establish CMake build with `RTRMetalEngine` library, sample targets, and shader compilation.
2. Implement logging, math helpers, Metal context, buffer allocator, and configuration loader.
3. Draft architecture/development guidelines.

## Milestone 2 – Core Engine (✅)
1. Define CPU-side scene graph (`Mesh`, `Material`, `Scene`, builders).
2. Integrate `GeometryStore`, `BufferAllocator`, and acceleration-structure scaffolding.
3. Assemble renderer façade and diagnostic BLAS build.

## Milestone 3 – Ray Tracing Pipelines (🚧)

| Focus | Deliverables | Acceptance |
| --- | --- | --- |
| **3A – Hardware-Accelerated Compute Pipeline** | • TLAS/BLAS construction via `MTLAccelerationStructureDescriptor`<br>• `raytracingKernel` compute pipeline（含 linked/visible functions）<br>• 统一的资源缓冲：per-frame uniform、geometry/material 指针、累积/random 纹理<br>• Renderer dispatch 绑定 TLAS（`setAccelerationStructure:`）写入渲染目标<br>• `supportsRaytracing == false` 时的 compute fallback（梯度/CPU 模式） | • RT 设备上渲染诊断 Cornell 场景得到非黑输出<br>• 日志/测试确认 TLAS、资源缓冲、dispatch 顺序正确<br>• Fallback 模式在不支持设备上输出确定性图像 |
| **3B – MPS Compute Pipeline** | • GPU shading kernel 覆盖射线生成、求交、着色、累积<br>• 与 `MPSRayIntersector` 共享资源缓冲<br>• 分辨率/SPP/累积控制可配置，保留 CPU 着色用于确定性校验 | • MPS 路径输出场景图像并生成稳定 hash<br>• CLI 支持 GPU/CPU 切换与累积参数，测试覆盖多场景 |
| **3C – Examples & Tooling** | • Off-screen CLI 生成 PPM/PNG + hash<br>• MetalKit/SwiftUI Demo：后端/采样/场景切换、累积 HUD、截图导出<br>• README/Docs 更新运行说明、硬件/回退要求<br>• `ctest` 脚本覆盖 TLAS 构建、资源缓冲、图像 hash | • CLI & GUI 在支持/不支持 RT 的设备上均能运行并输出结果<br>• 自动化测试验证核心路径（TLAS、资源、图像 hash） |

### Immediate Sprint Backlog
1. 实现 compute 光追管线：`raytracingKernel` + TLAS 绑定，替换 `dispatchRayTracingPass()` stub。
2. 引入 per-frame uniform ring buffer、资源指针缓冲、累积/随机纹理，并在 Renderer 中串联调度逻辑。
3. ~~让 MPS GPU 着色路径使用相同资源布局，同时保留 CPU 着色作为确定性回退。~~ ✅ GPU/CPU 均使用 `RTRRayTracingMaterial` 与纹理缓冲，差异比较仍以 CPU 结果为基准。
4. 搭建 CLI 图像导出与 hash 校验流程，更新 README/Docs 的硬件要求与运行步骤。

## Milestone 4 – Polish & Validation (🔒)
1. 扩展材质系统（纹理、多次弹射、Tone Mapping），保持后端一致性。
2. 增加 Profiling/QA 工具（hash 基线、性能脚本、捕获指南）。
3. 完善文档：开发者入门、硬件要求、回归流程、常见问题。
4. 在核心管线稳定后探索扩展功能（降噪、动画支持等）。

## Reference
- `IMPLEMENTATION_PLAN.md` contains the stage statuses and acceptance tests.
- `/Users/yanbing.xu/Desktop/MetalRayTracing` remains the reference sample for the MPS compute pipeline.
