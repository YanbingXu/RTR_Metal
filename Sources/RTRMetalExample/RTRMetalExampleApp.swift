import AppKit
import MetalKit
import SwiftUI
import RTRMetalEngine
import QuartzCore
import simd

@main
struct RTRMetalExampleApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 960, minHeight: 540)
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = RendererViewModel()

    var body: some View {
        MetalView(renderer: viewModel.renderer)
            .onAppear {
                viewModel.start()
            }
            .onDisappear {
                viewModel.stop()
            }
            .overlay(alignment: .topLeading) {
                VStack(alignment: .leading) {
                    Text("RTR Metal Ray Tracing")
                        .font(.title2)
                        .padding(8)
                    Text("FPS: \(String(format: "%.1f", viewModel.fps))")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.bottom, 8)
                }
                .background(Color.black.opacity(0.4))
                .foregroundColor(.white)
                .cornerRadius(8)
                .padding()
            }
    }
}

final class RendererViewModel: NSObject, ObservableObject {
    private let context: MetalContext
    let renderer: Renderer
    private var displayLink: CVDisplayLink?
    @Published var fps: Double = 0
    private var lastTimestamp: CFTimeInterval = 0

    override init() {
        guard let context = MetalContext() else {
            fatalError("Metal ray tracing is not supported on this device")
        }
        self.context = context
        guard let renderer = Renderer(context: context) else {
            fatalError("Unable to initialize Renderer")
        }
        self.renderer = renderer
        super.init()
        setupScene()
    }

    func setupScene() {
        let scene = SceneFactory.cornellBox()
        try? renderer.upload(scene: scene)
    }

    func start() {
        guard displayLink == nil else { return }
        var link: CVDisplayLink?
        CVDisplayLinkCreateWithActiveCGDisplays(&link)
        guard let displayLink = link else { return }
        CVDisplayLinkSetOutputCallback(displayLink, { (_, _, _, _, _, userData) -> CVReturn in
            let unmanaged = Unmanaged<RendererViewModel>.fromOpaque(userData!)
            unmanaged.takeUnretainedValue().displayLinkDidFire()
            return kCVReturnSuccess
        }, UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()))
        CVDisplayLinkStart(displayLink)
        self.displayLink = displayLink
    }

    func stop() {
        if let displayLink {
            CVDisplayLinkStop(displayLink)
            self.displayLink = nil
        }
    }

    private func displayLinkDidFire() {
        DispatchQueue.main.async {
            let now = CACurrentMediaTime()
            let delta = now - self.lastTimestamp
            self.lastTimestamp = now
            if delta > 0 {
                self.fps = 1.0 / delta
            }
        }
    }
}

struct MetalView: NSViewRepresentable {
    let renderer: Renderer

    func makeCoordinator() -> Coordinator {
        Coordinator(renderer: renderer)
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: renderer.context.device)
        view.clearColor = MTLClearColorMake(0.1, 0.1, 0.1, 1.0)
        view.colorPixelFormat = .rgba16Float
        view.delegate = context.coordinator
        view.framebufferOnly = false
        view.isPaused = false
        view.preferredFramesPerSecond = 60
        context.coordinator.view = view
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.updateCameraAspect(aspect: Float(nsView.bounds.width / nsView.bounds.height))
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        let renderer: Renderer
        var view: MTKView?
        private var camera = Camera(position: simd_float3(0, 0, 5), lookAt: simd_float3(0, 0, 0))

        init(renderer: Renderer) {
            self.renderer = renderer
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            updateCameraAspect(aspect: Float(size.width / size.height))
        }

        func draw(in view: MTKView) {
            camera.aspectRatio = Float(view.drawableSize.width / view.drawableSize.height)
            camera.lookAt = simd_float3(0, 0, -0.5)
            do {
                try renderer.draw(to: view, camera: camera)
            } catch {
                print("Renderer error: \(error)")
            }
        }

        func updateCameraAspect(aspect: Float) {
            camera.aspectRatio = aspect
        }
    }
}
