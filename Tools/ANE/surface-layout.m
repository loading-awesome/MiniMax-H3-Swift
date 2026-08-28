// surface-layout.m — Recover the private ANE IOSurface allocation contract.
//
// This is allocation-only: it does not compile a model or submit ANE work.
// It asks the runtime to allocate representative NCHW tensor surfaces and
// reports the IOSurface geometry it chose. That geometry is the authoritative
// answer for row pitch and minimum spatial extent; multiplying a flat byte
// allocation until evaluation succeeds cannot reveal either one.
//
// Build and run from the repository root:
//
//   xcrun clang -fobjc-arc -framework Foundation -framework IOSurface \
//     Tools/ANE/surface-layout.m -o /tmp/h3-ane-surface-layout
//   /tmp/h3-ane-surface-layout

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>

static void Probe(Class surfaceClass, uint32_t width, uint32_t channels,
                  uint32_t height, uint32_t bytesPerElement) {
    SEL selector = @selector(createIOSurfaceWithWidth:pixel_size:height:bytesPerElement:);
    IOSurfaceRef surface = ((IOSurfaceRef (*)(Class, SEL, uint32_t, uint32_t,
                                               uint32_t, uint32_t))objc_msgSend)(
        surfaceClass, selector, width, channels, height, bytesPerElement);
    if (surface == NULL) {
        printf("W=%-4u C=%-4u H=%u B=%u  REFUSED\n",
               width, channels, height, bytesPerElement);
        return;
    }

    printf("W=%-4u C=%-4u H=%u B=%u  surface=%zux%zu  bpe=%zu bpr=%-8zu alloc=%zu\n",
           width, channels, height, bytesPerElement,
           IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface),
           IOSurfaceGetBytesPerElement(surface), IOSurfaceGetBytesPerRow(surface),
           IOSurfaceGetAllocSize(surface));
    CFRelease(surface);
}

int main(void) {
    @autoreleasepool {
        void *handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
            "AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL);
        if (handle == NULL) {
            fprintf(stderr, "Unable to load AppleNeuralEngine.framework: %s\n", dlerror());
            return 2;
        }
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        SEL selector =
            @selector(createIOSurfaceWithWidth:pixel_size:height:bytesPerElement:);
        if (surfaceClass == Nil || ![surfaceClass respondsToSelector:selector]) {
            fprintf(stderr, "ANE IOSurface factory is unavailable\n");
            return 3;
        }

        printf("width boundary, C=128 H=1 fp16\n");
        const uint32_t widths[] = {
            1, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512
        };
        for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i)
            Probe(surfaceClass, widths[i], 128, 1, 2);

        printf("\nchannel boundary, W=64 H=1 fp16\n");
        const uint32_t channels[] = {1, 2, 4, 32, 64, 128, 256, 512};
        for (size_t i = 0; i < sizeof(channels) / sizeof(channels[0]); ++i)
            Probe(surfaceClass, 64, channels[i], 1, 2);

        printf("\nMLP tensors\n");
        Probe(surfaceClass, 64, 128, 1, 2);   // x/y [1,128,1,64]
        Probe(surfaceClass, 256, 128, 1, 2);  // gate/up [1,128,1,256]
        Probe(surfaceClass, 128, 256, 1, 2);  // down [1,256,1,128]
        Probe(surfaceClass, 64, 128, 2, 2);   // expose the height contribution

        (void)handle; // Objective-C classes remain registered for process life.
    }
    return 0;
}
