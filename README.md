# RTR Metal

RTR Metal is being rebuilt as a C++20 + Metal hardware ray tracing engine that targets Apple Silicon Macs. The repository now exposes a reusable static library (`RTRMetalEngine`), a small command-line sample (`RTRMetalSample`), buildable Metal shaders, and an executable smoke test to validate the toolchain.

## Stage Status

- ✅ **Stage 1** – CMake scaffold, shader build integration, sample + test binaries
- ✅ **Stage 2** – Core math/utilities, configuration & logging, Metal context, buffer allocator, scene + geometry upload
- 🚧 **Stage 3** – Stage&nbsp;3D hardware shading polish in progress; software RT/fallback work is paused until Stage 4

Remaining stages focus exclusively on the hardware ray tracing pipeline per [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). Software RT milestones were pushed to Stage 4.

## Project Layout

- `CMakeLists.txt` – Root build definition (library, sample, shaders, tests).
- `engine/` – Engine headers (`include/RTRMetalEngine/...`) and sources (`src/...`) split into `Core`, `Rendering`, and `Scene` bundles.
- `config/engine.ini` – Sample configuration loaded by the CLI demo via `ConfigLoader`.
- `sample/` – Console and on-screen samples that exercise the renderer facade.
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

- Sample: `./build/RTRMetalSample [--output=FILE] [--scene=cornell|reflective|glass] [--resolution=WxH] [--frames=N] [--mode=auto|hardware] [--max-bounces=N] [--hash] [--debug-albedo]`
- `--expect-hash=0xHASH` 会在渲染后比对 FNV-1a 结果，方便在有 RT GPU 的机器上做回归验证。
- `reflective` 和 `glass` 场景需要在 `assets/` 下提供 `mario.obj`（可从官方 MetalRayTracing 示例拷贝），否则会退回简易几何体。
- 调试可使用 `--debug-albedo` 直接输出材质反照率，便于验证资源管线。
- `--mode=hardware` 会强制尝试硬件 RT；默认 `auto` 与硬件模式一致，保留未来引入备用管线的选择。
- On-Screen Sample: build with `cmake --build build --target RTRMetalOnScreenSample` (or `cmake-build-debug` when using CLion) and run `open build/RTRMetalOnScreenSample.app`. The overlay toolbar provides mode selection (`auto|hardware`), resolution presets (plus a dynamic entry when resizing the window), and a screenshot button that writes `~/Pictures/RTR_<timestamp>.ppm`. Reference hash for the Cornell default is `0x72FDA1309C1E4FB1` (1024×768 single-sample).
- Tests: `cd build && ctest --output-on-failure`

Only the hardware ray tracing backend is active. Former software/MPS paths have been removed until the hardware feature set is complete.

Scenes available via `--scene=` include `prism`, `cornell`, `reflective`, and `glass`. The reflective/glass demos expect OBJ assets under `assets/` (for example the bundled `assets/mario.obj` sourced from the reference project).

> Tip: Adjust `config/engine.ini` to point at custom shader libraries or change the reported application name when embedding the engine elsewhere.

### Software RT Status

The previous software/MPS fallback renderer, CLI sample, and docs remain in `docs/mps_*.md` for historical context but are not part of the active build.

## Documentation

Project direction, architecture, and working agreements live in:

- [`docs/Development_Guidelines.md`](docs/Development_Guidelines.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- [`AGENTS.md`](AGENTS.md) – Contributor quick-start covering structure, build/test flow, and review expectations
- [`docs/Stage3C_OnScreenDemo.md`](docs/Stage3C_OnScreenDemo.md) – Notes covering the interactive sample and current reference hashes

The optional key `maxBounces = <n>` can be added to `config/engine.ini` to provide defaults for the sample apps, and the CLI flags above override those values when present.

## License

This project remains licensed under the MIT License. See [LICENSE](LICENSE) for details.
