# 架构说明

## 文档状态
- 当前有效（中文主文档）

## 架构目标
- 提供可测试、可复用的 C++20 + Metal 硬件 RT 基础。
- 将平台层、渲染层、场景层解耦，便于独立演进。

## 分层
```
应用层（CLI / On-Screen）
  ↓
渲染门面（Renderer）
  ↓
资源与场景（GeometryStore / ASBuilder / SceneData）
  ↓
平台服务（MetalContext / Config / Logger / Math）
  ↓
Metal GPU + Shaders
```

## 模块职责
- `Core`
  - `Logger`：统一日志输出
  - `ConfigLoader`：加载 `engine.ini`
  - `Math`：矩阵、包围盒等基础数学
- `Rendering`
  - `MetalContext`：设备与命令队列
  - `BufferAllocator`：缓冲创建与更新
  - `GeometryStore`：网格缓冲上传
  - `AccelerationStructureBuilder`：BLAS/TLAS 构建
  - `RayTracingPipeline`：RT compute 管线初始化
  - `Renderer`：场景加载、资源绑定、帧调度、输出写回
- `Scene`
  - `Scene/Mesh/Material`：CPU 侧场景描述
  - `SceneBuilder`：示例与测试场景构造

## 当前约束
- 当前只启用硬件 RT 主路径。
- 软件/MPS 回退在 Stage 4 恢复，当前不参与运行路径。
