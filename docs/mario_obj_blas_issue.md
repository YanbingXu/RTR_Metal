# Mario OBJ 与 BLAS 可交性问题（阶段记录）

## 文档状态
- 阶段记录（问题已收敛，保留追溯）

## 历史问题
- 早期曾出现 Mario OBJ（mesh 8）“可构建但不可交”现象。
- 当时通过 `--debug-isolate-mesh=8` + `instance-trace` 观测为 `<none>`。

## 当前结论（2026-02）
- Cornell 中 Mario 默认已恢复 OBJ 实网格路径，不再是默认占位几何。
- `--debug-isolate-mesh=8 --debug-visualization=instance-trace` 已可稳定得到非空命中统计（示例：`[8:40047]`）。

## 复现命令
```bash
./build/RTRMetalSample \
  --scene=cornell \
  --resolution=512x512 \
  --frames=1 \
  --mode=hardware \
  --asset-root=assets \
  --config=config/engine.ini \
  --debug-isolate-extras \
  --debug-isolate-mesh=8 \
  --debug-visualization=instance-trace \
  --output=iso_m8_trace.ppm
```

## 后续修复路径（专项任务）
1. 增加自动化回归：`mesh8 isolate + instance-trace` 必须出现非空命中统计。
2. 对 Mario OBJ 路径补视觉质量验收（边缘锯齿、法线一致性、材质细节）。
3. 对 Mario OBJ 路径补性能验收（与占位几何对比开销区间）。

## 对阶段目标的影响
- 该问题不再阻断 Stage 3D 主线。
- 当前风险从“可交性”转为“回归自动化与质量门禁”。
