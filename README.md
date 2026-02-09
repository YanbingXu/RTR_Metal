# RTR Metal

## 文档状态
- 当前有效（中文主文档）
- 最后更新：2026-02-06

## 项目简介
`RTR Metal` 正在重构为面向 Apple Silicon 的 C++20 + Metal 硬件光线追踪引擎。
当前仓库提供：
- 可复用静态库 `RTRMetalEngine`
- 命令行示例 `RTRMetalSample`
- 桌面窗口示例 `RTRMetalOnScreenSample`
- 与构建流程集成的 Metal shader 编译

## 阶段状态（唯一口径）
- `Stage 1`：完成（工程骨架与工具链）
- `Stage 2`：完成（核心模块、资源上传、场景基础）
- `Stage 3A`：完成（硬件 RT 计算路径打通）
- `Stage 3B`：完成（Cornell 基础着色）
- `Stage 3C`：完成（On-Screen 示例与截图流程）
- `Stage 3D`：进行中（反射/折射/累积/性能完善）
- `Stage 4`：未开始（软件/MPS 回退恢复，受 Stage 3D 阻塞）

详细阶段定义见：[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

## 目录结构
- `engine/`：引擎代码（`Core` / `Rendering` / `Scene`）
- `sample/`：CLI 与 On-Screen 示例
- `shaders/`：Metal shader 源码
- `tests/`：GoogleTest + CTest
- `config/`：配置文件
- `assets/`：运行资源
- `docs/`：设计、计划、归档文档

## 依赖环境
- macOS 14+
- Xcode 15+（含 `xcrun`、`metal`、`metallib`）
- 支持 Metal Ray Tracing 的 Apple Silicon GPU
- CMake 3.21+

## 构建与运行
```bash
cmake -S . -B build
cmake --build build
```

CLI 示例：
```bash
./build/RTRMetalSample --scene=cornell --frames=1 --mode=hardware --hash
```

On-Screen 示例：
```bash
cmake --build build --target RTRMetalOnScreenSample
open build/RTRMetalOnScreenSample.app
```

## 测试
默认 `build` 目录可能关闭测试（`RTR_BUILD_TESTS=OFF`）。建议使用独立目录显式开启：

```bash
cmake -S . -B build-tests -DRTR_BUILD_TESTS=ON
cmake --build build-tests
cd build-tests && ctest --output-on-failure
```

## Cornell 图像回归基线（2026-02-09）
以下 hash 基于当前主线：`--scene=cornell` + `--resolution=1024x768` + `--mode=hardware` +
Mario 临时占位几何策略。

命令模板：
```bash
./cmake-build-debug/RTRMetalSample \
  --scene=cornell \
  --resolution=1024x768 \
  --frames=<N> \
  --mode=hardware \
  --asset-root=assets \
  --config=config/engine.ini \
  --output=/tmp/cornell_baseline_f<N>.ppm \
  --hash
```

当前基线：
- `frames=1`：`0x4707C26E88759C9D`
- `frames=4`：`0x8E53D49E6CDCB3CB`
- `frames=16`：`0xB543187723A93D47`

## 当前已知事实
- 当前仅启用硬件 RT 路径；`auto` 与 `hardware` 行为一致。
- 旧软件/MPS 路径已归档，计划在 `Stage 4` 恢复。
- Cornell 场景中的 Mario 当前使用临时占位几何（保留材质/纹理链路），OBJ 网格专项见 [`docs/mario_obj_blas_issue.md`](docs/mario_obj_blas_issue.md)。
- 部分历史文档仍保留用于追溯，但均已标注“历史归档”。

## 关键文档
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- [`AGENTS.md`](AGENTS.md)
- [`docs/Documentation_Index.md`](docs/Documentation_Index.md)
- [`docs/project_overview.md`](docs/project_overview.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/Development_Guidelines.md`](docs/Development_Guidelines.md)

## 许可证
MIT，见 [`LICENSE`](LICENSE)。
