# RTRMetal 引擎设计与实现笔记

## 设计目标
- 构建一个可在 macOS 14+、Apple Silicon GPU 上运行的实时光线追踪引擎。
- 通过 Swift Package 将引擎与示例应用解耦，便于复用。
- 采用 Metal Ray Tracing API 组织 BLAS/TLAS、着色器管线与渲染循环。

## 模块划分
1. **Core**：`MetalContext` 负责设备选择、命令队列与默认库加载。
2. **Rendering**：`Renderer`、`RayTracingPipeline`、`AccelerationStructureBuilder` 负责渲染调度及加速结构生成。
3. **Scene**：定义 `Scene`、`Mesh`、`Material` 及实例信息，作为渲染输入。
4. **Shaders**：`MetalRayTracing.metal` 包含 ray-generation 与 shading 逻辑。

## 实现思路
1. **设备初始化**：扫描所有 Metal 设备，优先选择 `supportsRaytracing` 的 GPU，加载离线编译的 `default.metallib`。
2. **加速结构**：
   - 为每个 `Mesh` 缓存几何缓冲并创建 `MTLPrimitiveAccelerationStructureDescriptor`。
   - 组装实例信息，生成 TLAS。
3. **渲染循环**：
   - 根据 `MTKView` 尺寸更新帧 Uniform。
   - 设置 compute 管线与资源（材质、几何、实例、TLAS、输出纹理）。
   - 调度线程组并提交命令缓冲。
4. **着色流程**：
   - ray-generation kernel 计算视线方向，使用 `intersection_query` 遍历命中。
   - 通过 `InstanceInfoGPU`、`GeometryInfoGPU` 还原世界坐标与法线，执行简单的光照模型。

## 数据调用流
```
SceneFactory --> Renderer.upload(scene:)
                |-> AccelerationStructureBuilder.buildScene
                          |-> geometryResources(mesh)
                          |-> buildTLAS(instances)
                |-> Renderer.store buffers + TLAS

MTKView.draw -> Renderer.draw(to:camera:)
                |-> 更新 FrameUniforms -> MTLBuffer
                |-> setBuffer(材质/几何/实例)
                |-> setAccelerationStructure(TLAS)
                |-> dispatchThreadgroups -> MetalRayTracing.metal
```

## 依赖与打包
- `build_metallib.sh` 负责离线编译 `default.metallib`。
- SwiftPM 构建时会将 `Shaders/` 目录作为资源打包到 `RTRMetal_RTRMetalEngine.bundle`。
