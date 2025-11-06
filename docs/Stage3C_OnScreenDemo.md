# Stage 3C – On-Screen Sample Field Notes

The `RTRMetalOnScreenSample` bundle mirrors Apple’s Metal ray tracing viewer while staying on top of the shared renderer. This note captures the current behaviour so Stage 3C has a reproducible baseline.

## Launching

```bash
cmake --build cmake-build-debug --target RTRMetalOnScreenSample
open cmake-build-debug/RTRMetalOnScreenSample.app
```

The app sets the activation policy to `NSApplicationActivationPolicyRegular` and brings the window forward when it launches. If macOS still leaves the window in the dock, click the icon once—no additional parameters are required.

## Runtime Controls

The overlay in the top-left corner exposes three immediate controls:

- **Mode** – `Auto`, `Hardware`, `Gradient`. This calls `Renderer::setShadingMode` so the UI and CLI share the same switch.
- **Resolution** – 512×512, 1024×768, 1280×720, 1920×1080 plus a dynamic entry that tracks manual window resizing. Selecting a preset updates `MTKView::drawableSize` and `Renderer::setRenderSize`.
- **Screenshot** – saves the current frame to `~/Pictures/RTR_<timestamp>.ppm`. The handler pipes the request back through `Renderer::renderFrame()` so we reuse the existing PPM writer.

## Reference Output

Running the CLI sample with the on-screen defaults produces the current hardware baseline:

```bash
cd cmake-build-debug
./RTRMetalSample --scene=cornell --frames=1 --mode=hardware \
    --asset-root=.. --config=../config/engine.ini \
    --resolution=1024x768 --output=onscreen_reference.ppm --hash
```

- Output path: `cmake-build-debug/onscreen_reference.ppm`
- Dimensions: 1024×768 (single-sample Cornell box)
- FNV-1a hash: `0x72FDA1309C1E4FB1`

Use this hash for manual verification until we wire the on-screen path into automated checks.

## Follow-Up

- Hook the same hash check into docs/tests once Stage 3C closes.
- Add UI affordances for accumulation reset and backend toggling once we mirror Apple’s full toolbar.
