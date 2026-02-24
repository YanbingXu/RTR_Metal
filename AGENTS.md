# 仓库协作指南

## 文档状态
- 当前有效（中文主文档）

## 项目结构
- 根目录 `CMakeLists.txt`：组织静态库、示例、shader 构建和测试。
- 引擎代码在 `engine/`：头文件在 `engine/include/RTRMetalEngine`，实现在 `engine/src`（`Core` / `Rendering` / `Scene`）。
- shader 源码在 `shaders/`，产物默认为 `build/shaders/RTRShaders.metallib`。
- 配置在 `config/`，运行资源在 `assets/`，文档在 `docs/`。
- 测试在 `tests/`，由 CTest 注册。

## 常用命令
- 配置：`cmake -S . -B build`
- 构建：`cmake --build build`
- CLI 示例：`./build/RTRMetalSample --scene=cornell --frames=16 --mode=hardware`
- On-Screen 示例：`cmake --build build --target RTRMetalOnScreenSample && open build/RTRMetalOnScreenSample.app`
- 测试（推荐）：
  - `cmake -S . -B build-tests -DRTR_BUILD_TESTS=ON`
  - `cmake --build build-tests`
  - `cd build-tests && ctest --output-on-failure`

## 编码规范
- C++20，4 空格缩进，单行尽量不超过 120 列。
- 类型使用 `UpperCamelCase`，函数/变量使用 `lowerCamelCase`，常量使用 `kPascalCase`。
- 维持现有命名空间和模块边界，优先组合而非继承。
- shader 统一通过 CMake 调 `metal/metallib` 构建，不使用临时脚本分叉流程。

## 开发原则
- 以 Apple 官方 Metal RT 示例和既有参考项目为对照，但优先保证硬件主线闭环。
- 先稳定后扩展，不在主线引入未经验证的试验路径。
- 对资源绑定、缓冲上传、命令编码流程优先复用官方样式。

## 会话协作约束（用户明确要求，优先级高于默认流程）
- 不默认采用 workaround。若必须 workaround：先说明原因、影响范围、可回退方案；获得用户同意后再实施。
- 每次代码或关键参数修改后，必须运行实际渲染并展示结果图（至少提供输出路径；建议同时给出 hash 与渲染参数）。
- 每次准备提交前，先由用户确认；仅在用户同意后执行 `git commit`，并提供清晰提交描述（修改内容与目的）。
- 开发过程中不得引入已验证行为的回退；如发现回退，立即停止继续改动并先与用户确认处理方式。
- 若工作区存在用户本地未提交改动：先判断必要性。必要则单独提交；不必要则先说明理由并在用户同意后再丢弃。

## 测试要求
- 新功能要有对应测试（优先 `tests/` 下同模块文件）。
- 保持确定性；避免在单元测试中依赖真实 GPU 行为。
- 提交前应完成可编译与测试通过（若测试关闭，需说明原因与配置）。

## 提交与 PR
- 提交标题使用祈使句，建议不超过 72 字符。
- 涉及阶段变更时同步更新 `IMPLEMENTATION_PLAN.md`。
- PR 需说明影响范围，并附构建/测试结果。
