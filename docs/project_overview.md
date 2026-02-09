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
- 交互帧仍存在每帧 CPU 读回与同步阻塞。
- 折射参数存在于材质数据，但 shader 未形成闭环。
- 测试默认可能关闭，且图像回归尚未制度化。

## 近期优先级
1. 去掉交互路径的强制 CPU 读回与 `waitUntilCompleted`。
2. 完成折射/材质标记 shader 路径与参数开关。
3. 固化 Cornell/资产场景 hash 基线并纳入测试流程。
