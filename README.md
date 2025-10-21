# RTR Metal

RTR Metal is a real-time ray tracing engine written in Swift and Metal for Apple platforms that support the Metal ray tracing pipeline. The repository contains both the reusable engine (`RTRMetalEngine`) and a SwiftUI-based example viewer (`RTRMetalExample`).

## Features

- Metal ray tracing pipeline with TLAS/BLAS acceleration structures
- Simple scene representation with reusable meshes and materials
- Directional lighting and physically inspired shading
- Camera utilities for interactive applications
- Example viewer demonstrating the engine with a Cornell-box scene

## Project Layout

- `Package.swift` – Swift Package definition for the engine, example app, and tests.
- `Sources/RTRMetalEngine/` – Engine implementation including context management, scene types, renderer, and shaders.
- `Sources/RTRMetalExample/` – SwiftUI application that renders a demo scene using the engine.
- `docs/` – Architecture and development plan documents.
- `Tests/` – Lightweight unit tests for math utilities.

## Requirements

- macOS 14 (Sonoma) or newer
- Xcode 15 or newer with Metal ray tracing support
- Apple Silicon GPU (tested on M-series hardware)

## Building the Example App

1. Ensure the Swift toolchain/Xcode command line tools are installed.
2. From the project root, resolve dependencies:
   ```bash
   swift package resolve
   ```
3. Build and run the example using Xcode:
   ```bash
   open Package.swift
   ```
   Select the `RTRMetalExample` scheme and press **Run**.

Alternatively, you can build the example via the command line:

```bash
swift build --product RTRMetalExample
open .build/debug/RTRMetalExample.app
```

> **Note:** Command-line builds require running on macOS with Metal ray tracing capable hardware. The Linux CI environment cannot execute Metal code, so GPU validation must be performed on a local Mac.

## Usage

The `Renderer` class accepts a `Scene` description and draws into an `MTKView` each frame. The example app demonstrates how to:

1. Instantiate `MetalContext` and `Renderer`.
2. Build a Cornell-box scene via `SceneFactory`.
3. Drive rendering inside an `MTKViewDelegate` implementation.

## Documentation

Detailed architecture and roadmap information lives in:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/development_plan.md`](docs/development_plan.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
