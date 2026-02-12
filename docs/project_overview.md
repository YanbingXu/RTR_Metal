# 项目总览

## 文档状态
- 当前有效（中文主文档）

## 目标
1. 打造可复用的 macOS/Apple Silicon 硬件 RT 引擎。
2. 以 compute + TLAS/BLAS 为主路径，确保可验证、可扩展。
3. 提供 CLI 与 On-Screen 两种示例形态。
4. 建立稳定回归手段（hash/image diff + 文档基线）。

## 当前状态
- Stage 1~3C 已完成。
- Stage 3D 进行中：重点在折射、累积与交互性能。
- Stage 4（软件/MPS 回退）尚未开始。

## 关键问题
- 图像回归仅有 Cornell F1 自动门禁，F4/F16 仍缺稳定门禁。
- 文档状态与代码实现需持续同步（避免阶段判断漂移）。
- Stage 4 软件/MPS 回退尚未启动。

## 近期优先级
1. 维持交互路径无强制 CPU 读回阻塞，并补稳定性回归。
2. 扩展 Cornell/资产场景 hash 基线与自动门禁覆盖。
3. 在 Stage 3D 收口后启动 Stage 4 回退路径设计。
