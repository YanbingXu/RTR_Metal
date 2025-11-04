# ~~调试总结与关键修改~~

> **Deprecated:** 此文记录的“Ray Tracing Pipeline + SBT”调试步骤已被 compute-based 工作流取代，信息仅作存档参考。

## 当前状态
- 引擎在 Apple M4 Pro 上成功识别到 `supportsRaytracing` 的 GPU，并能完成加速结构构建与命令调度。
- 示例应用仍显示全黑，说明 shader 输出或资源绑定存在兼容性差异；需要参考官方示例进一步验证。

## 遇到的主要问题
1. **Runtime 编译失败**：初始版本尝试 `device.makeLibrary(source:)` 会报 “no member named 'trace_ray'” 等错误。
2. **Ray Tracing API 不匹配**：Metal Ray Tracing pipeline 的 miss/closest-hit 函数与现有 SDK 不兼容。
3. **资源绑定问题**：GPU 地址 (`gpuAddress`) 在当前 SDK 下不可用，导致几何数据无法通过指针访问。
4. **输出纹理未更新**：虽然命令缓冲执行完成，但屏幕仍然是黑色，需要进一步确认 compute 写入与 drawable 访问权限。

## 关键修改
- **MetalContext**：移除 runtime 编译 fallback，改为加载预编译 `default.metallib`，并记录详细日志。
- **build_metallib.sh**：新增脚本离线编译 Metal shader，并将 clang 模块缓存放入仓库内部目录。
- **RayTracingPipeline**：改为仅创建 ray-gen compute 管线，避免依赖缺失的 miss/closest-hit 函数。
- **MetalRayTracing.metal**：重写为 `intersection_query` 流程，并在无命中时写入渐变背景以验证可见输出。
- **Renderer**：添加日志记录线程组、drawable 大小；为 command buffer 增加完成回调，用于确认命令执行完毕。
- **示例应用**：确保 `MTKView` 在首帧打印调度信息，并捕获 `upload(scene:)` 的错误，以便 UI 提示。

## 后续建议
- 对照 Apple 官方文档与开源示例，调整 shader 与管线配置，尤其是 `MTLVisibleFunctionTable`、payload 与属性结构。
- 验证引擎模块中的 PROJ/和 GPU 地址获取逻辑，必要时改用 `MTLResource.gpuResourceID` 或显式传递缓冲索引。
- 分步调试：先实现在 compute shader 中写固定颜色，再逐步引入 TLAS/BLAS 与 shading。

## 2025-10-22 调试记录补充
- 将 shader 简化为纯红输出，并通过额外的中间纹理 + blit 复制到 drawable，确认 compute pass 仍未显示颜色。
- 核心问题定位到 “compute 写入 drawable 未生效”，下一步需对比 Apple 官方示例，复用其 pipeline 配置。
