# RTR Metal

RTR Metal is being rebuilt as a C++20 + Metal hardware ray tracing engine that targets Apple Silicon Macs. The repository now exposes a reusable static library (`RTRMetalEngine`), a small command-line sample (`RTRMetalSample`), buildable Metal shaders, and an executable smoke test to validate the toolchain.

## Stage Status

- ✅ **Stage 1** – CMake scaffold, shader build integration, sample + test binaries
- 🚧 **Stage 2** – Core math utilities, Metal context bootstrap, and logging system (initial slices landed)

Remaining stages cover the ray tracing pipeline and AppKit sample per [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Project Layout

- `CMakeLists.txt` – Root build definition (library, sample, shaders, tests).
- `engine/` – Engine headers (`include/RTRMetalEngine/...`) and sources (`src/...`) split into `Core`, `Rendering`, and `Scene` bundles.
- `sample/` – Temporary console sample that exercises the renderer facade.
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
- Tests: `cd build && ctest --output-on-failure`

Both binaries currently emit console output only; rendering integration arrives in later stages.

## Documentation

Project direction, architecture, and working agreements live in:

- [`docs/Development_Guidelines.md`](docs/Development_Guidelines.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

## License

This project remains licensed under the MIT License. See [LICENSE](LICENSE) for details.
