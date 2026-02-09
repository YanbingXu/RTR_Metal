# Stage 3C On-Screen 示例记录

## 文档状态
- 当前有效（基线记录）

## 目标
记录 `RTRMetalOnScreenSample` 的可复现实验基线，便于回归对照。

## 启动方式
```bash
cmake --build build --target RTRMetalOnScreenSample
open build/RTRMetalOnScreenSample.app
```

## 当前可用控制项
- 模式：`Auto`、`Hardware`（当前两者都走硬件路径）
- 分辨率：预设 + 窗口动态分辨率
- 截图：输出到 `~/Pictures/RTR_<timestamp>.ppm`
- 调试可视化：`none/albedo/instance-colors/instance-trace/primitive-trace`

## 参考输出（CLI 对照）
```bash
./build/RTRMetalSample --scene=cornell --frames=1 --mode=hardware \
  --resolution=1024x768 --output=onscreen_reference.ppm --hash
```

- 参考 hash（历史记录）：`0x72FDA1309C1E4FB1`
- 说明：在 shader 或采样策略变更后，需更新此基线并同步到文档与测试。
