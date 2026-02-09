# Stage 3 变更记录（归档）

## 文档状态
- 历史归档（仅用于追溯）

本文记录了早期 Stage 3 拆分讨论与过渡方案。
当前请以以下文档为准：
- [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)
- [`README.md`](../README.md)
- [`docs/project_overview.md`](project_overview.md)

## 归档说明
- 旧版内容中包含已暂停的 MPS 路线阶段划分，不再作为执行依据。
- 如需恢复回退路径，请直接在 Stage 4 下建立新任务，而不是复用本归档计划。

## 最新同步说明（2026-02-06）
- 本页仅做状态同步，不替代主计划文档。
- 与 `IMPLEMENTATION_PLAN.md` 当前口径一致：
  - Cornell 扩展几何中两球（镜面/折射）已稳定可见，可用于 Stage 3D 主线验证。
  - Mario 当前采用占位几何（保留材质与纹理链路），避免阻塞 Stage 3D 目标推进。
  - Mario OBJ 网格的 BLAS 可交性问题作为专项后续处理，详见 `docs/mario_obj_blas_issue.md`。

## Cornell 基线同步（2026-02-09）
- 运行条件：`--scene=cornell --resolution=1024x768 --mode=hardware --asset-root=assets --config=config/engine.ini`
- 结果 hash（FNV-1a）：
  - `frames=1`：`0x9A6AD96130FF3506`
  - `frames=4`：`0x0E0D4150478BDFEE`
  - `frames=16`：`0xEA655D1AB536C88C`
