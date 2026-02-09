# Mario OBJ 与 BLAS 可交性问题（阶段记录）

## 现状结论
- 当前硬件 RT 主线中，Cornell 场景的两球（mesh 6/7）已恢复可见并参与反射/折射。
- Mario 原始 OBJ（mesh 8）在进入 BLAS 后出现“可构建但不可交”的现象：
  - TLAS 实例存在，实例矩阵与材质索引正确。
  - `instance-trace` 在 `--debug-isolate-mesh=8` 下命中统计为 `<none>`。
  - 将 mesh 8 临时替换为简单盒体后可立即命中，说明 TLAS/实例链路正常，问题集中在 OBJ 几何本体与当前 BLAS 输入路径的兼容性。

## 临时策略（当前默认）
- 为避免阻塞阶段目标，Cornell 场景中的 Mario 先使用“占位几何”（盒体）+ 纹理材质链路：
  - 若 `assets/mario.png` 存在，则继续作为占位体贴图。
  - 若纹理缺失，使用显式占位颜色并打日志提醒。
- 该策略保证场景中第三个几何体稳定可见，便于继续验证光照、反射、折射与累积收敛。

## 复现命令
```bash
./cmake-build-debug/RTRMetalSample \
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
1. 新建最小 OBJ 回归集（仅保留 1~2 个已知三角簇）并比对 BLAS 命中。
2. 在独立分支中重做 OBJ->RT 几何上传路径，对齐 Apple 官方示例的顶点/索引布局与 descriptor 配置。
3. 增加自动化回归：
   - `mesh8 isolate + instance-trace` 必须出现非空命中统计。
   - 正常 Cornell 渲染图 hash 与基线对齐（允许噪声窗口内偏差）。
4. 专项通过后再恢复 Mario OBJ 实网格，移除占位体路径。

## 对阶段目标的影响
- 不影响当前阶段对硬件 RT 主路径的推进。
- 对视觉验收的影响：Mario 细节暂时不可用，但场景构图、材质模型、反射折射验证可继续。
