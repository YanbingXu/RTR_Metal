# Repository Guidelines

## Project Structure & Module Organization
- Swift Package with primary targets `RTRMetalEngine` (library) and `RTRMetalExample` (macOS app).
- Engine source lives in `Sources/RTRMetalEngine`, grouped into `Core`, `Rendering`, `Scene`, and `Shaders` bundles; Metal shaders are under `Sources/RTRMetalEngine/Shaders`.
- Example application code resides in `Sources/RTRMetalExample`; UI scaffolding and renderer view model are defined in `RTRMetalExampleApp.swift`.
- Unit test stubs are under `Tests/RTRMetalEngineTests`; add new tests here mirroring the engine module layout.

## Build, Test, and Development Commands
- `swift build --disable-sandbox` — compile in debug mode inside the repo sandbox.
- `swift build --configuration release --disable-sandbox` — produce optimized artifacts for profiling or distribution.
- `swift run --disable-sandbox RTRMetalExample` — launch the sample macOS app (requires GUI session and a ray-tracing capable GPU).
- `swift test --disable-sandbox` — execute XCTest suites in `Tests/`.

## Coding Style & Naming Conventions
- Follow Swift API Design Guidelines: 4-space indentation, `UpperCamelCase` types, `lowerCamelCase` members, and explicit ACLs when deviating from defaults.
- Prefer protocol-oriented designs and immutable structs for scene data; engines classes (`MetalContext`, `Renderer`) remain `final`.
- Metal shader function names mirror their Swift callers (e.g., `rayGenMain`, `missShader`) to ease linkage.
- Run `swift format` if available; otherwise, keep line lengths under ~120 columns and group related imports.

## Testing Guidelines
- Use XCTest cases under `Tests/RTRMetalEngineTests`; name files `*Tests.swift` and methods `test_*`.
- Target deterministic logic (e.g., geometry builders, buffer packing) while skipping GPU-specific assertions unless you can mock devices.
- Ensure new tests run via `swift test --disable-sandbox` before pushing.

## Commit & Pull Request Guidelines
- Craft commit messages in the imperative mood (`Add TLAS builder helper`) and keep subject lines ≤72 characters.
- For PRs, describe functional impact, include `swift build`/`swift test` results, and attach screenshots or screen recordings of the example app when UI-facing changes occur.
- Link to open issues or ADRs when applicable and call out required GPU capabilities for features that depend on Metal Ray Tracing.

## Hardware & Configuration Notes
- Development and runtime assume macOS 14+ with an Apple Silicon GPU that reports `supportsRaytracing`.
- When testing headlessly, expect `swift run` to hang because the GUI cannot surface; validate builds via `swift build` instead.
