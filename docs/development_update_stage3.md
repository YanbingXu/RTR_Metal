# 阶段 3 进展与下一步计划

## 当前状态概览

- **Metal 原生硬件 RT 管线**
  - 新增 `RayTracingPipeline` 封装，当 SDK 暴露 `MetalRayTracing.h` 时自动加载 `rayGenMain`、`missMain`、`closestHitMain` 并创建 `MTLRayTracingPipelineState`。
  - 在不存在头文件的环境下保持降级：输出警告，但允许后续构建、运行及其他模块开发。
  - Renderer 引导过程同时尝试构建诊断 BLAS，便于确认底层资源流正常工作。

- **MPS 后端（兼容路径）**
  - 新增 `MPSPathTracer`，通过 `MPSSupportsMTLDevice` 检测设备能力。
  - 新 `MPSRenderer` 目前生成占位梯度图（后续替换为真实 MPS 渲染管线），可用于端到端测试文件输出流程。
  - 新增 `RTR_BUILD_MPS_SAMPLE` 选项和 `RTRMetalMPSSample` CLI Demo，便于快速验证 MPS 代码。

- **公共基础设施**
  - Shader 构建流程支持多 `.metal` 源文件，输出统一 metallib。
  - BLAS 构建封装在 `AccelerationStructure`/`AccelerationStructureBuilder`，新增命令队列访问函数。
  - README、架构文档、实施计划都已更新，明确 Stage 3 子阶段（Native / MPS / Demo）。

## 新划分的工作任务

1. **Stage 3A：Metal 原生 RT**
   - TLAS 构建、SBT 绑定、Ray Dispatch（等待 SDK 支持时启用编译）。
   - 将当前诊断 BLAS 与 TLAS 关联，并输出到纹理。
2. **Stage 3B：MPS Path Tracing**
   - 整合 `MPSRayIntersector`，实现真实的光线生成、交点计算与着色。
   - 利用共享 `GeometryStore` 上传场景数据，复用诊断三角形或导入 Cornell Box。
   - 输出可验证的图像（可写入 PPM），在本地具备 GPU 的环境运行回归。
3. **Stage 3C：双 Demo / 文档**
   - 提供原生 RT Demo（在 SDK 支持时自动启用）与 MPS Demo 的选择机制。
   - 在 README 中补充使用说明、依赖要求、差异点。
   - 添加简单的合规测试（如图像 hash、pipeline 启动日志检查）。

## 下一步详细计划

1. **完善 MPS 渲染管线（Stage 3B）**
   - 基于 Apple 示例流水线整理：射线生成 → `MPSRayIntersector` → 阴影/间接光处理 → 累积。
   - 使用当前的 `GeometryStore` 输出顶点/索引，在 MPS 中构建 `MPSTriangleAccelerationStructure`。
   - 用 `MPSRenderer` 将结果写入纹理再导出（先生成 PPM，后续可考虑 MetalKit 窗口显示）。
   - 在本地真机验证生成结果，并保留一份测试截图/图像。

2. **原生 RT（Stage 3A）预备作业**
   - 在没有头文件的情况下，继续搭建 TLAS 数据结构、SBT 配置接口（编译时可通过 feature flag 屏蔽）。
   - 规划 Shader Binding Table 数据布局、hit group 命名约定，为未来启用做好准备。

3. **Demo & 文档同步（Stage 3C）**
   - 在两个 Demo（Native/MPS）之间建立统一入口或 CLI 参数，便于用户选择。
   - 整理 README 的设备/SDK 要求，明确目前的降级模式。
   - 添加执行脚本或 CTest 目标帮助验证 Demo 能成功输出图像/日志。

---

请先查阅上述内容，若有建议再告知。接下来我会按计划继续推进 MPS 渲染管线的实施。
