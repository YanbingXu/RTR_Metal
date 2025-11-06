#import "AppDelegate.h"
#import "ViewController.h"

@implementation RTRAppDelegate {
    NSWindow *_window;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    const NSRect frame = NSMakeRect(0.0, 0.0, 1024.0, 768.0);
    const NSWindowStyleMask style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                    NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

    _window = [[NSWindow alloc] initWithContentRect:frame
                                           styleMask:style
                                             backing:NSBackingStoreBuffered
                                               defer:NO];
    _window.title = @"RTR Metal On-Screen";

    RTRViewController *controller = [[RTRViewController alloc] init];
    _window.contentViewController = controller;
    [_window center];
    [_window makeKeyAndOrderFront:nil];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end
