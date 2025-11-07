#import "ViewController.h"

#import <MetalKit/MetalKit.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/Renderer.hpp"
#include "RTRMetalEngine/Scene/CornellBox.hpp"

using RTRRendererPtr = std::unique_ptr<rtr::rendering::Renderer>;

namespace {

struct ModeOption {
    const char* title;
    const char* mode;
};

struct ResolutionOption {
    const char* title;
    std::uint32_t width;
    std::uint32_t height;
};

constexpr ModeOption kModeOptions[] = {
    {"Auto", "auto"},
    {"Hardware", "hardware"},
    {"Gradient", "gradient"},
};

constexpr ResolutionOption kResolutionOptions[] = {
    {"512 x 512", 512u, 512u},
    {"1024 x 768", 1024u, 768u},
    {"1280 x 720", 1280u, 720u},
    {"1920 x 1080", 1920u, 1080u},
};

std::string resolveShaderLibraryPath() {
    if (NSBundle* bundle = [NSBundle mainBundle]) {
        NSString* resourcePath = [bundle pathForResource:@"RTRShaders" ofType:@"metallib"];
        if (resourcePath.length > 0) {
            return std::string(resourcePath.UTF8String);
        }
    }
    return "shaders/RTRShaders.metallib";
}

std::string resolveAssetRootPath() {
    if (NSBundle* bundle = [NSBundle mainBundle]) {
        NSString* assetsPath = [[bundle resourcePath] stringByAppendingPathComponent:@"assets"];
        BOOL isDirectory = NO;
        if ([[NSFileManager defaultManager] fileExistsAtPath:assetsPath isDirectory:&isDirectory] && isDirectory) {
            return std::string(assetsPath.UTF8String);
        }
    }
    return "assets";
}

rtr::core::EngineConfig buildEngineConfig() {
    rtr::core::EngineConfig config{};
    config.applicationName = "RTR Metal On-Screen";
    config.shaderLibraryPath = resolveShaderLibraryPath();
    config.shadingMode = "hardware";
    config.accumulationEnabled = true;
    config.accumulationFrames = 0;
    config.samplesPerPixel = 1;
    config.sampleSeed = 0;
    return config;
}

NSDictionary* makeResolutionInfo(std::uint32_t width, std::uint32_t height) {
    return @{ @"width": @(width), @"height": @(height) };
}

bool resolutionMatches(NSDictionary* info, std::uint32_t width, std::uint32_t height) {
    if (!info) {
        return false;
    }
    const auto storedWidth = static_cast<std::uint32_t>([info[@"width"] unsignedIntValue]);
    const auto storedHeight = static_cast<std::uint32_t>([info[@"height"] unsignedIntValue]);
    return storedWidth == width && storedHeight == height;
}

}  // namespace

@interface RTRViewController () <MTKViewDelegate>
- (void)setupOverlayUI;
- (void)modeChanged:(id)sender;
- (void)resolutionChanged:(id)sender;
- (void)captureScreenshot:(id)sender;
- (void)selectCurrentMode;
- (void)selectResolutionForWidth:(std::uint32_t)width height:(std::uint32_t)height;
- (void)performPendingScreenshot;
@end

@implementation RTRViewController {
    MTKView* _mtkView;
    RTRRendererPtr _renderer;
    rtr::scene::Scene _scene;
    id<MTLRenderPipelineState> _displayPipeline;
    bool _sceneLoaded;
    NSPopUpButton* _modePopup;
    NSPopUpButton* _resolutionPopup;
    NSButton* _screenshotButton;
    std::uint32_t _currentWidth;
    std::uint32_t _currentHeight;
    bool _pendingScreenshot;
    std::string _pendingScreenshotPath;
}

- (void)loadView {
    MTKView* view = [[MTKView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 1024.0, 768.0) device:nil];
    view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.view = view;
}

- (void)viewDidLoad {
    [super viewDidLoad];

    _mtkView = static_cast<MTKView*>(self.view);
    _mtkView.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    _mtkView.depthStencilPixelFormat = MTLPixelFormatInvalid;
    _mtkView.sampleCount = 1;
    _mtkView.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    _mtkView.enableSetNeedsDisplay = NO;
    _mtkView.paused = NO;
    _mtkView.preferredFramesPerSecond = 60;
    _mtkView.framebufferOnly = NO;
    _mtkView.delegate = self;

    [self initializeRenderer];
    [self setupOverlayUI];
}

- (void)initializeRenderer {
    rtr::core::Logger::setMinimumLevel(rtr::core::LogLevel::Warning);
    rtr::core::EngineConfig config = buildEngineConfig();
    _renderer = std::make_unique<rtr::rendering::Renderer>(config);

    id<MTLDevice> engineDevice = (__bridge id<MTLDevice>)_renderer->deviceHandle();
    if (!engineDevice) {
        engineDevice = MTLCreateSystemDefaultDevice();
    }
    _mtkView.device = engineDevice;

    if (![self buildDisplayPipelineWithDevice:engineDevice shaderLibrary:config.shaderLibraryPath]) {
        rtr::core::Logger::error("OnScreenSample", "Failed to create display pipeline");
    }

    const CGSize drawableSize = _mtkView.drawableSize;
    _currentWidth = static_cast<std::uint32_t>(drawableSize.width);
    _currentHeight = static_cast<std::uint32_t>(drawableSize.height);
    _pendingScreenshot = false;
    _renderer->setRenderSize(_currentWidth, _currentHeight);

    const std::string assetRoot = resolveAssetRootPath();
    _scene = rtr::scene::createCornellBoxScene(assetRoot);
    _sceneLoaded = _renderer->loadScene(_scene);
    if (!_sceneLoaded) {
        rtr::core::Logger::warn("OnScreenSample", "Renderer failed to load Cornell scene");
    }
}

- (void)setupOverlayUI {
    NSView* container = [[NSView alloc] initWithFrame:NSZeroRect];
    container.translatesAutoresizingMaskIntoConstraints = NO;
    container.wantsLayer = YES;
    container.layer.backgroundColor = [[NSColor colorWithWhite:0.08 alpha:0.6] CGColor];
    container.layer.cornerRadius = 8.0;

    _modePopup = [[NSPopUpButton alloc] init];
    _modePopup.translatesAutoresizingMaskIntoConstraints = NO;
    for (const ModeOption& option : kModeOptions) {
        NSString* title = [NSString stringWithUTF8String:option.title];
        [_modePopup addItemWithTitle:title];
        NSMenuItem* item = [_modePopup lastItem];
        item.representedObject = [NSString stringWithUTF8String:option.mode];
    }
    _modePopup.target = self;
    _modePopup.action = @selector(modeChanged:);

    _resolutionPopup = [[NSPopUpButton alloc] init];
    _resolutionPopup.translatesAutoresizingMaskIntoConstraints = NO;
    for (const ResolutionOption& option : kResolutionOptions) {
        NSString* title = [NSString stringWithUTF8String:option.title];
        [_resolutionPopup addItemWithTitle:title];
        NSMenuItem* item = [_resolutionPopup lastItem];
        item.representedObject = makeResolutionInfo(option.width, option.height);
    }
    _resolutionPopup.target = self;
    _resolutionPopup.action = @selector(resolutionChanged:);

    _screenshotButton = [NSButton buttonWithTitle:@"Screenshot"
                                           target:self
                                           action:@selector(captureScreenshot:)];
    _screenshotButton.translatesAutoresizingMaskIntoConstraints = NO;

    NSStackView* stack = [NSStackView stackViewWithViews:@[_modePopup, _resolutionPopup, _screenshotButton]];
    stack.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    stack.spacing = 8.0;
    stack.edgeInsets = NSEdgeInsetsMake(6.0, 8.0, 6.0, 8.0);
    stack.translatesAutoresizingMaskIntoConstraints = NO;

    [container addSubview:stack];
    [_mtkView addSubview:container];

    [NSLayoutConstraint activateConstraints:@[
        [stack.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
        [stack.trailingAnchor constraintEqualToAnchor:container.trailingAnchor],
        [stack.topAnchor constraintEqualToAnchor:container.topAnchor],
        [stack.bottomAnchor constraintEqualToAnchor:container.bottomAnchor],
        [container.leadingAnchor constraintEqualToAnchor:_mtkView.leadingAnchor constant:16.0],
        [container.topAnchor constraintEqualToAnchor:_mtkView.topAnchor constant:16.0]
    ]];

    [self selectCurrentMode];
    [self selectResolutionForWidth:_currentWidth height:_currentHeight];
}

- (BOOL)buildDisplayPipelineWithDevice:(id<MTLDevice>)device shaderLibrary:(const std::string&)path {
    if (!device) {
        return NO;
    }

    NSError* error = nil;
    NSURL* libraryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
    if (!library || error) {
        rtr::core::Logger::error("OnScreenSample",
                                 "Failed to load shader library for display pipeline (%s)",
                                 error ? error.localizedDescription.UTF8String : "unknown error");
        return NO;
    }

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"RTRDisplayVertex"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"RTRDisplayFragment"];
    if (!vertexFunction || !fragmentFunction) {
        rtr::core::Logger::error("OnScreenSample", "Missing display shaders in metallib");
        return NO;
    }

    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.label = @"RTR Display Pipeline";
    descriptor.vertexFunction = vertexFunction;
    descriptor.fragmentFunction = fragmentFunction;
    descriptor.colorAttachments[0].pixelFormat = _mtkView.colorPixelFormat;

    _displayPipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!_displayPipeline || error) {
        rtr::core::Logger::error("OnScreenSample",
                                 "Failed to create display pipeline state (%s)",
                                 error ? error.localizedDescription.UTF8String : "unknown error");
        return NO;
    }
    return YES;
}

#pragma mark - MTKViewDelegate

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    if (_renderer) {
        _currentWidth = static_cast<std::uint32_t>(size.width);
        _currentHeight = static_cast<std::uint32_t>(size.height);
        _renderer->setRenderSize(_currentWidth, _currentHeight);
        [self selectResolutionForWidth:_currentWidth height:_currentHeight];
    }
}

- (void)drawInMTKView:(MTKView *)view {
    if (!_renderer || !_sceneLoaded) {
        return;
    }

    id<CAMetalDrawable> drawable = view.currentDrawable;
    if (!drawable) {
        return;
    }

    if (!_renderer->renderFrameInteractive()) {
        return;
    }

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)_renderer->commandQueueHandle();
    if (!commandQueue) {
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (!commandBuffer) {
        return;
    }

    id<MTLTexture> sourceTexture = (__bridge id<MTLTexture>)_renderer->currentColorTexture();
    if (!sourceTexture) {
        [commandBuffer commit];
        return;
    }

    MTLRenderPassDescriptor* passDescriptor = view.currentRenderPassDescriptor;
    if (!passDescriptor) {
        [commandBuffer commit];
        return;
    }

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
    [encoder setRenderPipelineState:_displayPipeline];
    [encoder setFragmentTexture:sourceTexture atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];

    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];

    if (_pendingScreenshot) {
        [self performPendingScreenshot];
    }
}

- (void)modeChanged:(id)sender {
    NSMenuItem* item = [_modePopup selectedItem];
    NSString* modeString = (NSString*)item.representedObject;
    if (!modeString) {
        return;
    }
    const std::string mode(modeString.UTF8String);
    _renderer->setShadingMode(mode);
}

- (void)resolutionChanged:(id)sender {
    NSMenuItem* item = [_resolutionPopup selectedItem];
    NSDictionary* info = (NSDictionary*)item.representedObject;
    if (!info) {
        return;
    }
    const std::uint32_t width = static_cast<std::uint32_t>([info[@"width"] unsignedIntValue]);
    const std::uint32_t height = static_cast<std::uint32_t>([info[@"height"] unsignedIntValue]);
    _currentWidth = width;
    _currentHeight = height;
    _mtkView.drawableSize = CGSizeMake(width, height);
    _renderer->setRenderSize(width, height);
}

- (void)captureScreenshot:(id)sender {
    NSDateFormatter* formatter = [[NSDateFormatter alloc] init];
    formatter.dateFormat = @"yyyyMMdd_HHmmss";
    NSString* timestamp = [formatter stringFromDate:[NSDate date]];
    NSArray<NSString*>* pictures = NSSearchPathForDirectoriesInDomains(NSPicturesDirectory, NSUserDomainMask, YES);
    NSString* directory = pictures.firstObject ?: NSHomeDirectory();
    NSString* filename = [NSString stringWithFormat:@"RTR_%@.ppm", timestamp];
    NSString* fullPath = [directory stringByAppendingPathComponent:filename];
    _pendingScreenshotPath = fullPath.UTF8String;
    _pendingScreenshot = true;
    rtr::core::Logger::info("OnScreenSample", "Screenshot requested -> %s", _pendingScreenshotPath.c_str());
}

- (void)selectCurrentMode {
    const std::string mode = _renderer ? _renderer->config().shadingMode : std::string("auto");
    NSString* current = [NSString stringWithUTF8String:mode.c_str()];
    for (NSMenuItem* item in _modePopup.itemArray) {
        if (![item.representedObject isKindOfClass:[NSString class]]) {
            continue;
        }
        if ([(NSString*)item.representedObject isEqualToString:current]) {
            [_modePopup selectItem:item];
            return;
        }
    }
    [_modePopup selectItemAtIndex:0];
}

- (void)selectResolutionForWidth:(std::uint32_t)width height:(std::uint32_t)height {
    for (NSMenuItem* item in _resolutionPopup.itemArray) {
        if (resolutionMatches((NSDictionary*)item.representedObject, width, height)) {
            [_resolutionPopup selectItem:item];
            return;
        }
    }

    NSString* customTitle = [NSString stringWithFormat:@"%u x %u", width, height];
    NSMenuItem* existing = [_resolutionPopup itemWithTitle:customTitle];
    if (!existing) {
        [_resolutionPopup addItemWithTitle:customTitle];
        existing = [_resolutionPopup lastItem];
    }
    existing.representedObject = makeResolutionInfo(width, height);
    [_resolutionPopup selectItem:existing];
}

- (void)performPendingScreenshot {
    if (!_renderer || _pendingScreenshotPath.empty()) {
        _pendingScreenshot = false;
        _pendingScreenshotPath.clear();
        return;
    }

    _renderer->setOutputPath(_pendingScreenshotPath);
    _renderer->renderFrame();
    rtr::core::Logger::info("OnScreenSample", "Wrote screenshot to %s", _pendingScreenshotPath.c_str());
    _pendingScreenshot = false;
    _pendingScreenshotPath.clear();
}

@end
