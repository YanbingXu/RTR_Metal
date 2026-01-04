#import "ViewController.h"

#import <MetalKit/MetalKit.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "RTRMetalEngine/Core/EngineConfig.hpp"
#include "RTRMetalEngine/Core/Logger.hpp"
#include "RTRMetalEngine/Rendering/Renderer.hpp"
#include "SampleAppUtils.hpp"

#ifndef RTR_SOURCE_DIR
#define RTR_SOURCE_DIR ""
#endif
#ifndef RTR_BINARY_DIR
#define RTR_BINARY_DIR ""
#endif

using namespace std::string_literals;
namespace fs = std::filesystem;

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
};

constexpr ResolutionOption kResolutionOptions[] = {
    {"512 x 512", 512u, 512u},
    {"1024 x 768", 1024u, 768u},
    {"1280 x 720", 1280u, 720u},
    {"1920 x 1080", 1920u, 1080u},
};

struct RuntimeResources {
    rtr::core::EngineConfig config;
    fs::path assetRoot;
};

std::vector<fs::path> makeSearchBases(const fs::path& bundleResources) {
    std::vector<fs::path> bases;
    if (!bundleResources.empty()) {
        bases.push_back(bundleResources);
    }
    bases.push_back(fs::current_path());
    bases.push_back(fs::path(RTR_BINARY_DIR));
    bases.push_back(fs::path(RTR_SOURCE_DIR));
    return bases;
}

std::optional<fs::path> resolveBundleResourcePath(NSString* name, NSString* ext) {
    if (NSBundle* bundle = [NSBundle mainBundle]) {
        NSString* resourcePath = [bundle pathForResource:name ofType:ext];
        if (resourcePath.length > 0) {
            return fs::path(resourcePath.UTF8String);
        }
    }
    return std::nullopt;
}

RuntimeResources makeRuntimeResources() {
    RuntimeResources resources{};
    fs::path bundleResources;
    NSBundle* bundle = [NSBundle mainBundle];
    if (bundle) {
        NSString* bundlePath = [bundle resourcePath];
        if (bundlePath.length > 0) {
            bundleResources = fs::path(bundlePath.UTF8String);
        }
    }

    const std::vector<fs::path> searchBases = makeSearchBases(bundleResources);

    auto configPath = rtr::sample::resolvePath("config/engine.ini", false, searchBases);
    resources.config = rtr::sample::loadEngineConfig(configPath.value_or(fs::path{}));
    resources.config.applicationName = "RTR Metal On-Screen";

    const fs::path configDirectory = configPath ? configPath->parent_path() : fs::path{};

    if (auto bundleShader = resolveBundleResourcePath(@"RTRShaders", @"metallib")) {
        resources.config.shaderLibraryPath = bundleShader->string();
    } else {
        std::vector<fs::path> shaderBases = searchBases;
        if (!configDirectory.empty()) {
            shaderBases.insert(shaderBases.begin(), configDirectory);
        }
        if (auto resolvedShader = rtr::sample::resolvePath(resources.config.shaderLibraryPath,
                                                           false,
                                                           shaderBases)) {
            resources.config.shaderLibraryPath = resolvedShader->string();
        } else {
            rtr::core::Logger::warn("OnScreenSample",
                                    "Shader library '%s' not found via search paths",
                                    resources.config.shaderLibraryPath.c_str());
        }
    }

    std::vector<fs::path> assetBases = searchBases;
    if (bundle) {
        NSString* assetsPath = [[bundle resourcePath] stringByAppendingPathComponent:@"assets"];
        BOOL isDirectory = NO;
        if ([[NSFileManager defaultManager] fileExistsAtPath:assetsPath isDirectory:&isDirectory] && isDirectory) {
            assetBases.insert(assetBases.begin(), fs::path(assetsPath.UTF8String));
        }
    }
    resources.assetRoot =
        rtr::sample::resolvePath("assets", true, assetBases).value_or(fs::path("assets"));
    if (resources.assetRoot.empty()) {
        rtr::core::Logger::warn("OnScreenSample", "Asset root not found; demo scenes may degrade");
    } else {
        rtr::core::Logger::info("OnScreenSample", "Using assets from %s", resources.assetRoot.string().c_str());
    }

    resources.config.shadingMode = "hardware";
    return resources;
}

NSDictionary* makeResolutionInfo(std::uint32_t width, std::uint32_t height, bool dynamic) {
    if (dynamic) {
        return @{ @"width": @(width), @"height": @(height), @"dynamic": @YES };
    }
    return @{ @"width": @(width), @"height": @(height) };
}

NSDictionary* makeResolutionInfo(std::uint32_t width, std::uint32_t height) {
    return makeResolutionInfo(width, height, false);
}

bool resolutionMatches(NSDictionary* info, std::uint32_t width, std::uint32_t height) {
    if (!info) {
        return false;
    }
    const auto storedWidth = static_cast<std::uint32_t>([info[@"width"] unsignedIntValue]);
    const auto storedHeight = static_cast<std::uint32_t>([info[@"height"] unsignedIntValue]);
    return storedWidth == width && storedHeight == height;
}

bool isDynamicResolution(NSDictionary* info) {
    if (!info) {
        return false;
    }
    NSNumber* marker = info[@"dynamic"];
    return marker ? [marker boolValue] : false;
}

}  // namespace

@interface RTRViewController ()
- (void)setupOverlayUI;
- (void)modeChanged:(id)sender;
- (void)resolutionChanged:(id)sender;
- (void)captureScreenshot:(id)sender;
- (void)selectCurrentMode;
- (void)selectResolutionForWidth:(std::uint32_t)width height:(std::uint32_t)height;
- (void)performPendingScreenshot;
- (void)updateRenderSizeWithWidth:(std::uint32_t)width height:(std::uint32_t)height;
- (void)updateDynamicResolutionMenuItem;
- (void)handleDrawableSizeChange:(CGSize)size updateRenderer:(BOOL)updateRenderer;
@end

@implementation RTRViewController {
    MTKView* _mtkView;
    RTRRendererPtr _renderer;
    rtr::scene::Scene _scene;
    id<MTLRenderPipelineState> _displayPipeline;
    bool _sceneLoaded;
    NSPopUpButton* _modePopup;
    NSPopUpButton* _resolutionPopup;
    NSMenuItem* _dynamicResolutionItem;
    NSButton* _screenshotButton;
    NSButton* _debugToggle;
    std::uint32_t _currentWidth;
    std::uint32_t _currentHeight;
    std::uint32_t _renderWidth;
    std::uint32_t _renderHeight;
    bool _resolutionOverride;
    bool _pendingScreenshot;
    std::string _pendingScreenshotPath;
    fs::path _assetRootPath;
}

- (void)loadView {
    MTKView* view = [[MTKView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 1024.0, 768.0) device:nil];
    view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.view = view;
}

- (void)viewDidLoad {
    [super viewDidLoad];

    _mtkView = static_cast<MTKView*>(self.view);
    _mtkView.autoResizeDrawable = YES;
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
    rtr::core::Logger::setMinimumLevel(rtr::core::LogLevel::Info);
    RuntimeResources runtime = makeRuntimeResources();

    const fs::path forcedAssetRoot = fs::path(RTR_SOURCE_DIR) / "assets";
    if (!forcedAssetRoot.empty() && fs::exists(forcedAssetRoot)) {
        runtime.assetRoot = forcedAssetRoot;
        rtr::core::Logger::info("OnScreenSample",
                                "Forcing asset root to %s for feature verification",
                                forcedAssetRoot.string().c_str());
    } else {
        rtr::core::Logger::warn("OnScreenSample",
                                "Forced asset root %s missing; continuing with resolved path",
                                forcedAssetRoot.string().c_str());
    }

    setenv("RTR_ENABLE_MARIO", "1", 1);

    _assetRootPath = runtime.assetRoot;
    rtr::core::Logger::info("OnScreenSample",
                            "Using shader library: %s",
                            runtime.config.shaderLibraryPath.c_str());
    const std::string assetPathLog = _assetRootPath.empty() ? std::string("<unset>") : _assetRootPath.string();
    rtr::core::Logger::info("OnScreenSample", "Using asset root: %s", assetPathLog.c_str());

    _renderer = std::make_unique<rtr::rendering::Renderer>(runtime.config);

    id<MTLDevice> engineDevice = (__bridge id<MTLDevice>)_renderer->deviceHandle();
    if (!engineDevice) {
        engineDevice = MTLCreateSystemDefaultDevice();
    }
    _mtkView.device = engineDevice;

    if (![self buildDisplayPipelineWithDevice:engineDevice shaderLibrary:runtime.config.shaderLibraryPath]) {
        rtr::core::Logger::error("OnScreenSample", "Failed to create display pipeline");
    }

    const CGSize drawableSize = _mtkView.drawableSize;
    _currentWidth = static_cast<std::uint32_t>(drawableSize.width);
    _currentHeight = static_cast<std::uint32_t>(drawableSize.height);
    _resolutionOverride = false;
    _pendingScreenshot = false;
    [self updateRenderSizeWithWidth:_currentWidth height:_currentHeight];

    const std::string sceneName = "cornell";
    _scene = rtr::sample::buildScene(sceneName, _assetRootPath);
    _sceneLoaded = _renderer->loadScene(_scene);
    if (!_sceneLoaded) {
        rtr::core::Logger::warn("OnScreenSample", "Renderer failed to load Cornell scene");
    } else {
        const auto bounds = _scene.computeSceneBounds();
        rtr::core::Logger::info("OnScreenSample",
                                "Scene bounds min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f)",
                                bounds.min.x,
                                bounds.min.y,
                                bounds.min.z,
                                bounds.max.x,
                                bounds.max.y,
                                bounds.max.z);
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
    [_resolutionPopup addItemWithTitle:@"Window"];
    _dynamicResolutionItem = [_resolutionPopup lastItem];
    _dynamicResolutionItem.representedObject = makeResolutionInfo(_currentWidth, _currentHeight, true);
    _resolutionPopup.target = self;
    _resolutionPopup.action = @selector(resolutionChanged:);

    _screenshotButton = [NSButton buttonWithTitle:@"Screenshot"
                                           target:self
                                           action:@selector(captureScreenshot:)];
    _screenshotButton.translatesAutoresizingMaskIntoConstraints = NO;

    _debugToggle = [NSButton checkboxWithTitle:@"Debug Albedo"
                                        target:self
                                        action:@selector(debugModeChanged:)];
    _debugToggle.translatesAutoresizingMaskIntoConstraints = NO;
    _debugToggle.state = NSControlStateValueOff;

    NSStackView* stack = [NSStackView stackViewWithViews:@[
        _modePopup,
        _resolutionPopup,
        _screenshotButton,
        _debugToggle,
    ]];
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

    [self updateDynamicResolutionMenuItem];
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
    [self handleDrawableSizeChange:size updateRenderer:(!_resolutionOverride)];
}

- (void)drawInMTKView:(MTKView *)view {
    if (!_renderer || !_sceneLoaded) {
        return;
    }

    id<CAMetalDrawable> drawable = view.currentDrawable;
    MTLRenderPassDescriptor* passDescriptor = view.currentRenderPassDescriptor;
    if (!drawable || !passDescriptor) {
        rtr::core::Logger::warn("OnScreenSample",
                                "Drawable unavailable (descriptor=%s size=%.0fx%.0f)",
                                passDescriptor ? "ok" : "nil",
                                view.drawableSize.width,
                                view.drawableSize.height);
        return;
    }

    if (!_renderer->renderFrameInteractive()) {
        rtr::core::Logger::warn("OnScreenSample",
                                "Renderer skipped interactive frame (size=%ux%u)",
                                _renderWidth,
                                _renderHeight);
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
        rtr::core::Logger::warn("OnScreenSample",
                                "Renderer returned nil color texture (renderSize=%ux%u)",
                                _renderWidth,
                                _renderHeight);
        [commandBuffer commit];
        return;
    }

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
    id<MTLTexture> drawableTexture = passDescriptor.colorAttachments[0].texture;
    const double viewportWidth = drawableTexture ? drawableTexture.width : view.drawableSize.width;
    const double viewportHeight = drawableTexture ? drawableTexture.height : view.drawableSize.height;
    rtr::core::Logger::info("OnScreenSample",
                            "Display pass viewport %.0fx%.0f drawableSize=%.0fx%.0f renderSize=%ux%u",
                            viewportWidth,
                            viewportHeight,
                            view.drawableSize.width,
                            view.drawableSize.height,
                            _renderWidth,
                            _renderHeight);
    MTLViewport viewport = {0.0, 0.0, viewportWidth, viewportHeight, 0.0, 1.0};
    [encoder setViewport:viewport];
    MTLScissorRect scissor = {0, 0, static_cast<NSUInteger>(viewportWidth), static_cast<NSUInteger>(viewportHeight)};
    [encoder setScissorRect:scissor];
    [encoder setRenderPipelineState:_displayPipeline];
    [encoder setFragmentTexture:sourceTexture atIndex:0];
    const auto renderWidth = sourceTexture.width;
    const auto renderHeight = sourceTexture.height;
    const simd_float2 invRenderSize = simd_make_float2(renderWidth > 0 ? 1.0f / static_cast<float>(renderWidth)
                                                                      : 0.0f,
                                                       renderHeight > 0 ? 1.0f / static_cast<float>(renderHeight)
                                                                       : 0.0f);
    [encoder setFragmentBytes:&invRenderSize length:sizeof(invRenderSize) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
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
    _renderer->resetAccumulation();
}

- (void)resolutionChanged:(id)sender {
    NSMenuItem* item = [_resolutionPopup selectedItem];
    NSDictionary* info = (NSDictionary*)item.representedObject;
    if (!info) {
        return;
    }
    const std::uint32_t width = static_cast<std::uint32_t>([info[@"width"] unsignedIntValue]);
    const std::uint32_t height = static_cast<std::uint32_t>([info[@"height"] unsignedIntValue]);
    const bool dynamicSelection = isDynamicResolution(info);
    _resolutionOverride = !dynamicSelection;
    rtr::core::Logger::info("OnScreenSample",
                            "Resolution menu -> %ux%u (%s)",
                            width,
                            height,
                            dynamicSelection ? "dynamic" : "preset");
    if (dynamicSelection) {
        _mtkView.autoResizeDrawable = YES;
        [self handleDrawableSizeChange:_mtkView.drawableSize updateRenderer:YES];
        [self selectResolutionForWidth:_currentWidth height:_currentHeight];
    } else {
        _mtkView.autoResizeDrawable = NO;
        CGSize drawableSize = CGSizeMake(width, height);
        if (!CGSizeEqualToSize(_mtkView.drawableSize, drawableSize)) {
            _mtkView.drawableSize = drawableSize;
        }
        rtr::core::Logger::info("OnScreenSample",
                                "Drawable resized to %.0fx%.0f",
                                _mtkView.drawableSize.width,
                                _mtkView.drawableSize.height);
        [self handleDrawableSizeChange:drawableSize updateRenderer:NO];
        [self updateRenderSizeWithWidth:width height:height];
        [self selectResolutionForWidth:width height:height];
    }
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

- (void)debugModeChanged:(id)sender {
    if (!_renderer) {
        return;
    }
    const bool enabled = (_debugToggle.state == NSControlStateValueOn);
    _renderer->setDebugMode(enabled);
    _renderer->resetAccumulation();
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
    const bool matchesWindow = (width == _currentWidth && height == _currentHeight);
    if (_dynamicResolutionItem && !_resolutionOverride && matchesWindow) {
        [_resolutionPopup selectItem:_dynamicResolutionItem];
        return;
    }

    NSString* customTitle = [NSString stringWithFormat:@"%u x %u", width, height];
    NSMenuItem* existing = [_resolutionPopup itemWithTitle:customTitle];
    if (!existing) {
        [_resolutionPopup addItemWithTitle:customTitle];
        existing = [_resolutionPopup lastItem];
    }
    existing.representedObject = makeResolutionInfo(width, height, false);

    for (NSMenuItem* item in _resolutionPopup.itemArray) {
        if (item == _dynamicResolutionItem) {
            continue;
        }
        if (resolutionMatches((NSDictionary*)item.representedObject, width, height)) {
            [_resolutionPopup selectItem:item];
            return;
        }
    }

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

- (void)updateRenderSizeWithWidth:(std::uint32_t)width height:(std::uint32_t)height {
    const bool sizeChanged = (width != _renderWidth || height != _renderHeight);
    _renderWidth = width;
    _renderHeight = height;
    if (_renderer) {
        _renderer->setRenderSize(width, height);
        if (sizeChanged) {
            _renderer->resetAccumulation();
        }
    }
}

- (void)updateDynamicResolutionMenuItem {
    if (!_dynamicResolutionItem) {
        return;
    }
    NSString* title = [NSString stringWithFormat:@"Window (%u x %u)", _currentWidth, _currentHeight];
    _dynamicResolutionItem.title = title;
    _dynamicResolutionItem.representedObject = makeResolutionInfo(_currentWidth, _currentHeight, true);
    if (!_resolutionOverride) {
        [_resolutionPopup selectItem:_dynamicResolutionItem];
    }
}

- (void)handleDrawableSizeChange:(CGSize)size updateRenderer:(BOOL)updateRenderer {
    _currentWidth = static_cast<std::uint32_t>(std::max(1.0, std::round(size.width)));
    _currentHeight = static_cast<std::uint32_t>(std::max(1.0, std::round(size.height)));
    [self updateDynamicResolutionMenuItem];
    if (updateRenderer) {
        [self updateRenderSizeWithWidth:_currentWidth height:_currentHeight];
        if (!_resolutionOverride) {
            [self selectResolutionForWidth:_currentWidth height:_currentHeight];
        }
    }
}

@end
