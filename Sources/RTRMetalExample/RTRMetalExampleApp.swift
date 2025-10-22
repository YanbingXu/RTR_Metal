import AppKit
import MetalKit
import SwiftUI
import RTRMetalEngine
import QuartzCore
import os.log
import simd

@main
struct RTRMetalExampleApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 960, minHeight: 540)
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = RendererViewModel()

    var body: some View {
        Group {
            if let renderer = viewModel.renderer {
                MetalView(renderer: renderer)
                    .onAppear { viewModel.start() }
                    .onDisappear { viewModel.stop() }
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
            } else if let message = viewModel.statusMessage {
                VStack(spacing: 12) {
                    Text("Metal Ray Tracing Unavailable")
                        .font(.title2)
                    Text(message)
                        .font(.body)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

final class RendererViewModel: NSObject, ObservableObject {
    private var context: MetalContext?
    @Published var renderer: Renderer?
    private var displayLink: CVDisplayLink?
    @Published var fps: Double = 0
    @Published var statusMessage: String?
    private var lastTimestamp: CFTimeInterval = 0

    override init() {
        super.init()
        initialiseRenderer()
    }

    private func initialiseRenderer() {
        guard let context = MetalContext() else {
            statusMessage = "This Mac does not expose the Metal ray tracing feature set. Please ensure the system GPU supports hardware ray tracing and that the app has GPU access permissions."
            return
        }
        guard let renderer = Renderer(context: context) else {
            statusMessage = "The ray tracing pipeline could not be created on this GPU."
            return
        }
        self.context = context
        self.renderer = renderer
        setupScene()
    }

    func setupScene() {
        guard let renderer else { return }
        let scene = SceneFactory.cornellBox()
        do {
            try renderer.upload(scene: scene)
        } catch {
            statusMessage = "Failed to upload scene: \(error.localizedDescription)"
            if #available(macOS 11.0, *) {
                os_log("Failed to upload scene: %{public}@", log: renderer.context.log, type: .error, String(describing: error))
            } else {
                print("Failed to upload scene: \(error)")
            }
        }
    }

    func start() {
        guard renderer != nil, displayLink == nil else { return }
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
        guard renderer != nil else { return }
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
        view.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
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
        private var hasLoggedFrame = false

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
                if !hasLoggedFrame {
                    if #available(macOS 11.0, *) {
                        os_log("Coordinator draw call", log: renderer.context.log, type: .info)
                    }
                    hasLoggedFrame = true
                }
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
