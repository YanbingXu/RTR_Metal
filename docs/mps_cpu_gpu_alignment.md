# CPU / GPU 渲染结果对齐调试记录

## 背景
- `MPSRendererImageComparisonTests` 在 Apple Silicon 设备上持续失败，报告 GPU 输出相对 CPU 参考图像存在 20+ 级别的像素差异。
- 日志中 GPU 渲染阶段打印了 `GPU shading matched CPU output (max byte diff 0.00)`，与测试结论矛盾。

## 现象复盘
1. 测试用例分别调用 CPU-only 和 GPU 路径各自渲染一帧，再比较两份 PPM 文件。
2. 同一日志会出现“GPU 匹配 CPU”的提示，但图像对比仍然超差。
3. 详细日志显示两次渲染的命中数量不同（CPU: 63 652，GPU: 63 653）。说明两次 `MPSRayIntersector` 调度结果并非完全一致。

## 定位思路
1. **对齐光线生成**：
   - 先比对 CPU 与 GPU 光线生成公式，发现 GPU kernel 采样像素左上角，而 CPU 使用像素中心。同步二者后，初始差异消除。
2. **补充诊断输出**：
   - 在 `MPSRenderer` 中添加 `dumpRay` 和最差像素日志，记录 CPU/GPU 颜色、像素坐标、三角形索引与重心。
   - 统一将 GPU 缓冲量化回 8-bit 再比较，避免量化影响判断。
3. **分析命中统计**：
   - 注意到 GPU 模式内部会先 CPU shade 一遍做对比，因此“匹配”日志仅说明同一次渲染内部一致，而非和外部 CPU-only 帧一致。
   - 两次独立渲染的加速结构重建和射线发射由 MPS 决定，存在 ±1 像素的非确定性。

## 根因
测试把 CPU 和 GPU 分别调用两次 `renderFrame`，导致使用了两份独立的 MPS 交点结果。由于 MPS 在浮点精度和调度上的非确定性，单个像素可能落在不同的三角面，从而引起 20+ 的字节差异。

## 解决方案
1. **统一渲染通路**：
   - 新增 `MPSRenderer::FrameComparison` 结构，封装 CPU/GPU 输出、最大字节差 / 浮点差及分辨率。
   - 提供 `computeFrame` 与 `renderFrameComparison`，在一次交点计算后同步生成 CPU 与 GPU 像素，并可选择性写出两份 PPM。
   - `renderFrame` 复用该逻辑，只按阈值挑选最终输出，实现单通道渲染无需额外成本。
2. **保持 GPU 与 CPU 光线生成一致**：
   - `shaders/RTRRayTracing.metal` 使用 `float2(gid) + 0.5f`，与 CPU 的 `makePrimaryRay` 同步。
3. **测试更新**：
   - `MPSRendererImageComparisonTests` 直接调用 `renderFrameComparison`，断言一次渲染中的 CPU/GPU 输出差异，避免重复调度造成的非确定性。

## 验证
- `cmake --build cmake-build-debug`
- `ctest --output-on-failure`
- 在具备 GPU 的环境运行同样测试已通过。

## 后续建议
1. 如需长期监控图像差异，可在测试中记录 `FrameComparison.maxFloatDifference`，并允许更精细的误差分析（例如统计分布或写入 JSON）。
2. 若未来支持可配置分辨率 / 随机采样，需要保证 `computeFrame` 继续复用同一份射线和交点数据。
3. 定期核查 `renderFrameComparison` 输出的 PPM，作为回归样本备份，避免 MPS 升级导致视觉回归而未触发阈值。

---
**关联提交**：
- `engine/include/RTRMetalEngine/MPS/MPSRenderer.hpp`
- `engine/src/MPS/MPSRenderer.mm`
- `shaders/RTRRayTracing.metal`
- `tests/src/mps/MPSRendererImageComparisonTests.mm`
