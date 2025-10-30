# RTR Metal

RTR Metal is being rebuilt as a C++20 + Metal hardware ray tracing engine that targets Apple Silicon Macs. The repository now exposes a reusable static library (`RTRMetalEngine`), a small command-line sample (`RTRMetalSample`), buildable Metal shaders, and an executable smoke test to validate the toolchain.

## Stage Status

- ✅ **Stage 1** – CMake scaffold, shader build integration, sample + test binaries
- ✅ **Stage 2** – Core math/utilities, configuration & logging, Metal context, buffer allocator, scene + geometry upload
- 🚧 **Stage 3** – Acceleration structure sizing/building foundations and ray tracing shader stubs underway

Remaining stages cover the ray tracing pipeline and AppKit sample per [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Project Layout

- `CMakeLists.txt` – Root build definition (library, sample, shaders, tests).
- `engine/` – Engine headers (`include/RTRMetalEngine/...`) and sources (`src/...`) split into `Core`, `Rendering`, and `Scene` bundles.
- `config/engine.ini` – Sample configuration loaded by the CLI demo via `ConfigLoader`.
- `sample/` – Temporary console sample that exercises the renderer facade.
- `sample/src/mps_main.mm` – MPS demo entry point (build with `RTR_BUILD_MPS_SAMPLE=ON`).
- `tests/` – Executables registered with CTest for deterministic regression coverage.
- `shaders/` – Metal shader sources compiled into `RTRShaders.metallib` at build time.
- `docs/` – Architecture notes and development guidelines.

## Requirements

- macOS 14 (Sonoma) or newer
- Xcode 15+ command line tools (for `xcrun`, `metal`, `metallib`)
- Apple Silicon GPU with Metal ray tracing capability
- CMake ≥ 3.21
- Initial CMake configure must reach GitHub once to fetch GoogleTest (cached afterward)

## Building

Configure and build from the repository root:

```bash
cmake -S . -B build
cmake --build build
```

This flow compiles the engine library, sample executable, unit test binary, and generates `build/shaders/RTRShaders.metallib` automatically.

## Running

- Sample: `./build/RTRMetalSample`
- MPS Sample: `./build/RTRMetalMPSSample [--cpu|--gpu] [--compare] [--reset-accum] [--no-accum|--accum] [--accum-frames=N] [--resolution=WxH] [--output=<file>] [--cpu-output=<file>] [--gpu-output=<file>]`
- Tests: `cd build && ctest --output-on-failure`

Both binaries currently emit console output only; rendering integration arrives in later stages.

The MPS sample defaults to the shading mode configured in `config/engine.ini` (`shadingMode = auto|cpu|gpu`).
CLI switches override that default. `--compare` writes both CPU and GPU frames while reporting pixel hash statistics, and
`--reset-accum` clears the GPU accumulation buffer before rendering a new frame. `--no-accum` disables GPU accumulation (single-sample), `--accum` re-enables it, `--accum-frames=N` caps accumulation at `N` frames (0 keeps accumulating indefinitely), and `--resolution=WxH` overrides the default 512x512 frame size.

Scenes available via `--scene=` include `prism`, `cornell`, `reflective`, and `glass`. The reflective/glass demos expect OBJ assets under `assets/` (for example the bundled `assets/mario.obj` sourced from the reference project).

> `RTRMetalMPSSample` 会在当前工作目录输出 `mps_output.ppm`（若设备支持 MPS ray tracing），便于快速检查渲染结果。

> Tip: Adjust `config/engine.ini` to point at custom shader libraries or change the reported application name when embedding the engine elsewhere.

## Documentation

Project direction, architecture, and working agreements live in:

- [`docs/Development_Guidelines.md`](docs/Development_Guidelines.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

The optional keys `accumulation = on|off`, `accumulationFrames = <n>`, `samplesPerPixel = <n>`, and `sampleSeed = <n>` can be added to `config/engine.ini` to provide defaults for the sample apps, and the CLI flags above override those values when present.

## License

This project remains licensed under the MIT License. See [LICENSE](LICENSE) for details.
