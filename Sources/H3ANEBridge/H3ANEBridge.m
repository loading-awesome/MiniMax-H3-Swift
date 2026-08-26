// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

#import "H3ANEBridge.h"
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <IOSurface/IOSurface.h>
#import <stdatomic.h>
#import <sys/sysctl.h>

// The private surface holds only what this bridge actually calls. Everything
// is reached through `objc_msgSend` with an explicit prototype cast, because
// these classes have no headers and the variadic `objc_msgSend` prototype
// passes floats and structs wrong on arm64.
static Class DescriptorClass = nil;
static Class ModelClass = nil;
static Class RequestClass = nil;
static Class IOSurfaceObjectClass = nil;

static NSString *SysctlString(const char *name) {
    size_t length = 0;
    if (sysctlbyname(name, NULL, &length, NULL, 0) != 0 || length < 2) return nil;
    char *buffer = calloc(1, length);
    if (!buffer) return nil;
    NSString *value = nil;
    if (sysctlbyname(name, buffer, &length, NULL, 0) == 0) {
        value = [NSString stringWithUTF8String:buffer];
    }
    free(buffer);
    return value;
}

static bool ClassHasClassSelector(Class cls, SEL selector) {
    return cls && class_respondsToSelector(object_getClass(cls), selector);
}

static bool ClassHasInstanceSelector(Class cls, SEL selector) {
    return cls && class_respondsToSelector(cls, selector);
}

static size_t Align64(size_t v) { return (v + 63) & ~(size_t)63; }

#pragma mark - Tensors

struct H3ANETensor {
    int rows;
    int width;
    size_t rowBytes;
    IOSurfaceRef _Nullable surface;
    id _Nullable object;            // _ANEIOSurfaceObject
};

H3ANETensor* h3_ane_tensor_create(int rows, int width) {
    if (!h3_ane_is_available() || rows <= 0 || width <= 0) return NULL;

    H3ANETensor *t = (H3ANETensor *)calloc(1, sizeof(H3ANETensor));
    if (!t) return NULL;

    @try {

    t->rows = rows;
    t->width = width;
    t->rowBytes = Align64((size_t)width * sizeof(_Float16));

    // The engine wants one flat allocation; the row structure is a convention
    // between the MIL shape and whoever fills the surface, not something
    // IOSurface enforces. Describing it as a 1 x allocSize byte surface keeps
    // IOSurface from imposing its own padding on top of ours.
    size_t allocSize = Align64((size_t)rows * t->rowBytes);
    if (allocSize < 16384) allocSize = 16384;

    NSDictionary *props = @{
        (id)kIOSurfaceWidth:           @(allocSize),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow:     @(allocSize),
        (id)kIOSurfaceAllocSize:       @(allocSize),
        (id)kIOSurfacePixelFormat:     @0
    };
    t->surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (!t->surface) { free(t); return NULL; }

    // Padded columns are read by the engine even though their results are
    // discarded, so they must be defined rather than whatever the page held.
    void *base = IOSurfaceGetBaseAddress(t->surface);
    if (base) memset(base, 0, allocSize);

    t->object = ((id(*)(Class, SEL, IOSurfaceRef, size_t))objc_msgSend)(
        IOSurfaceObjectClass, @selector(objectWithIOSurface:startOffset:), t->surface, 0);
    if (!t->object) {
        CFRelease(t->surface);
        free(t);
        return NULL;
    }
    return t;
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception while creating a tensor: %@", exception);
        t->object = nil;
        if (t->surface) CFRelease(t->surface);
        free(t);
        return NULL;
    }
}

void h3_ane_tensor_free(H3ANETensor *t) {
    if (!t) return;
    t->object = nil;                // ARC releases; `free` alone would leak it
    if (t->surface) CFRelease(t->surface);
    free(t);
}

void* h3_ane_tensor_ptr(H3ANETensor *t) {
    if (!t || !t->surface) return NULL;
    // Establish the device-to-host coherency boundary before MLX wraps the
    // allocation. The pointer remains valid after unlocking; the Session owns
    // the IOSurface for longer than the managed MLXArray that adopts it.
    if (IOSurfaceLock(t->surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return NULL;
    void *base = IOSurfaceGetBaseAddress(t->surface);
    IOSurfaceUnlock(t->surface, kIOSurfaceLockReadOnly, NULL);
    return base;
}

size_t h3_ane_tensor_row_bytes(H3ANETensor *t) { return t ? t->rowBytes : 0; }

bool h3_ane_tensor_is_dense(H3ANETensor *t) {
    return t && t->rowBytes == (size_t)t->width * sizeof(_Float16);
}

bool h3_ane_tensor_write_prefix(H3ANETensor *t, const void *src, int rows, int width) {
    if (!t || !src) return false;
    if (rows != t->rows || width > t->width) return false;

    uint8_t *dst = (uint8_t *)IOSurfaceGetBaseAddress(t->surface);
    if (!dst) return false;

    const size_t srcRow = (size_t)width * sizeof(_Float16);
    const uint8_t *s = (const uint8_t *)src;
    for (int r = 0; r < rows; ++r) {
        memcpy(dst + (size_t)r * t->rowBytes, s + (size_t)r * srcRow, srcRow);
    }
    return true;
}

bool h3_ane_tensor_write(H3ANETensor *t, const void *src, int rows, int width) {
    if (!t || !src) return false;
    // Refuse rather than truncate. A caller whose shape drifted from what was
    // compiled has a bug, and silently writing the first N bytes of it would
    // produce plausible-looking numbers from the wrong memory.
    if (rows != t->rows || width != t->width) return false;

    if (IOSurfaceLock(t->surface, 0, NULL) != kIOReturnSuccess) return false;
    uint8_t *dst = (uint8_t *)IOSurfaceGetBaseAddress(t->surface);
    if (!dst) {
        IOSurfaceUnlock(t->surface, 0, NULL);
        return false;
    }

    const size_t srcRow = (size_t)width * sizeof(_Float16);
    if (srcRow == t->rowBytes) {
        memcpy(dst, src, (size_t)rows * srcRow);
    } else {
        const uint8_t *s = (const uint8_t *)src;
        for (int r = 0; r < rows; ++r) {
            memcpy(dst + (size_t)r * t->rowBytes, s + (size_t)r * srcRow, srcRow);
        }
    }
    return IOSurfaceUnlock(t->surface, 0, NULL) == kIOReturnSuccess;
}

#pragma mark - Availability

bool h3_ane_is_available(void) {
    static dispatch_once_t once;
    static bool available = false;
    dispatch_once(&once, ^{
        // This bridge is a versioned private ABI, not a generalized ANE API.
        // Refuse unvalidated machines/builds unless a researcher deliberately
        // opts into probing them. This turns an OS update into the normal MLX
        // fallback instead of an unrecognized-selector crash during sampling.
        NSDictionary *env = NSProcessInfo.processInfo.environment;
        bool override = [env[@"H3_ANE_ALLOW_UNVALIDATED"] isEqualToString:@"1"];
        if (!override) {
            if (![SysctlString("kern.osversion") isEqualToString:@"25F84"] ||
                ![SysctlString("hw.model") isEqualToString:@"Mac15,14"]) return;
        }

        void *handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                              "AppleNeuralEngine", RTLD_LAZY);
        if (!handle) return;
        DescriptorClass      = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        ModelClass           = NSClassFromString(@"_ANEInMemoryModel");
        RequestClass         = NSClassFromString(@"_ANERequest");
        IOSurfaceObjectClass = NSClassFromString(@"_ANEIOSurfaceObject");
        available =
            ClassHasClassSelector(DescriptorClass, @selector(modelWithMILText:weights:optionsPlist:)) &&
            ClassHasClassSelector(ModelClass, @selector(inMemoryModelWithDescriptor:)) &&
            ClassHasClassSelector(RequestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:)) &&
            ClassHasClassSelector(IOSurfaceObjectClass, @selector(objectWithIOSurface:startOffset:)) &&
            ClassHasInstanceSelector(ModelClass, @selector(hexStringIdentifier)) &&
            ClassHasInstanceSelector(ModelClass, @selector(compileWithQoS:options:error:)) &&
            ClassHasInstanceSelector(ModelClass, @selector(loadWithQoS:options:error:)) &&
            ClassHasInstanceSelector(ModelClass, @selector(unloadWithQoS:error:)) &&
            ClassHasInstanceSelector(ModelClass, @selector(evaluateWithQoS:options:request:error:));
    });
    return available;
}

#pragma mark - Programs

struct H3ANEProgram {
    int s, k, n;
    id _Nullable model;             // _ANEInMemoryModel
    NSString * _Nullable scratchDir;
};

/// `y[s,n] = x[k,s]^T @ w[k,n]` on the engine.
///
/// Both operands arrive with `k` as the leading axis. That is not the
/// orientation the checkpoint holds, and it is not arbitrary: at the
/// production shard, declaring the activation `[1,s,1,k]` and contracting over
/// the last axis of both operands runs at 2.42 TFLOP/s a die and takes 30
/// seconds to compile, while this form runs at 3.79 and compiles in under a
/// second. The engine wants the sequence as the minor axis of the activation.
///
/// So the activation is transposed by the host — cheap, and MLX folds it into
/// the fp16 conversion it has to do anyway — and the weight is transposed once
/// when it is uploaded. The output needs no transpose at all: `mm` is reshaped
/// straight to `[1,s,1,n]`, which is the row-major `[s, n]` the caller wants
/// and is contiguous, so it can be aliased into MLX rather than copied.
static NSString *MatmulMIL(int s, int k, int n) {
    NSMutableString *m = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16,[1,%d,1,%d]> a, tensor<fp16,[1,%d,1,%d]> w) {\n",
                    k, s, k, n];
    [m appendFormat:@"        tensor<int32,[4]> ra=const()[name=string(\"ra\"),"
                     "val=tensor<int32,[4]>([1,1,%d,%d])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> a2=reshape(shape=ra,x=a)[name=string(\"a2\")];\n"
                     "        tensor<int32,[4]> pm=const()[name=string(\"pm\"),"
                     "val=tensor<int32,[4]>([0,1,3,2])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> at=transpose(perm=pm,x=a2)[name=string(\"at\")];\n",
                    k, s, k, s, s, k];
    [m appendFormat:@"        tensor<int32,[4]> rw=const()[name=string(\"rw\"),"
                     "val=tensor<int32,[4]>([1,1,%d,%d])];\n"
                     "        tensor<fp16,[1,1,%d,%d]> w2=reshape(shape=rw,x=w)[name=string(\"w2\")];\n",
                    k, n, k, n];
    [m appendFormat:@"        bool bf=const()[name=string(\"bf\"),val=bool(false)];\n"
                     "        tensor<fp16,[1,1,%d,%d]> mm=matmul(transpose_x=bf,transpose_y=bf,"
                     "x=at,y=w2)[name=string(\"mm\")];\n"
                     "        tensor<int32,[4]> ro=const()[name=string(\"ro\"),"
                     "val=tensor<int32,[4]>([1,%d,1,%d])];\n"
                     "        tensor<fp16,[1,%d,1,%d]> y=reshape(shape=ro,x=mm)[name=string(\"y\")];\n"
                     "    } -> (y);\n}\n",
                    s, n, s, n, s, n];
    return m;
}

H3ANEProgram* h3_ane_program_create(int s, int k, int n) {
    if (!h3_ane_is_available() || s <= 0 || k <= 0 || n <= 0) return NULL;

    H3ANEProgram *p = NULL;
    @try {
    p = (H3ANEProgram *)calloc(1, sizeof(H3ANEProgram));
    if (!p) return NULL;
    p->s = s; p->k = k; p->n = n;

    NSData *milData = [MatmulMIL(s, k, n) dataUsingEncoding:NSUTF8StringEncoding];

    id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        DescriptorClass, @selector(modelWithMILText:weights:optionsPlist:), milData, @{}, nil);
    if (!descriptor) { h3_ane_program_free(p); return NULL; }

    id model = ((id(*)(Class, SEL, id))objc_msgSend)(
        ModelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
    if (!model) { h3_ane_program_free(p); return NULL; }

    // The compiler writes next to the model URL, so it needs somewhere to
    // write. Under NSTemporaryDirectory and removed on free — the previous
    // version left a directory per context under /tmp for the machine to keep.
    static atomic_uint serial = 0;
    unsigned mine = atomic_fetch_add(&serial, 1);
    NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    p->scratchDir = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"h3-ane-%d-%u-%@", getpid(), mine, identifier]];

    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:[p->scratchDir stringByAppendingPathComponent:@"weights"]
  withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[p->scratchDir stringByAppendingPathComponent:@"model.mil"]
                 options:0 error:nil];

    if ([model respondsToSelector:@selector(setModelURL:)]) {
        ((void(*)(id, SEL, id))objc_msgSend)(model, @selector(setModelURL:),
                                             [NSURL fileURLWithPath:p->scratchDir]);
    }

    NSError *error = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
    if (!ok) {
        NSLog(@"[H3ANEBridge] compile failed for s=%d k=%d n=%d: %@", s, k, n, error);
        h3_ane_program_free(p);
        return NULL;
    }

    ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
    if (!ok) {
        NSLog(@"[H3ANEBridge] load failed for s=%d k=%d n=%d: %@", s, k, n, error);
        h3_ane_program_free(p);
        return NULL;
    }

    p->model = model;
    return p;
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception while compiling: %@", exception);
        h3_ane_program_free(p);
        return NULL;
    }
}

void h3_ane_program_free(H3ANEProgram *p) {
    if (!p) return;
    @try {
        if (p->model) {
            NSError *err = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                p->model, @selector(unloadWithQoS:error:), 21, &err);
            p->model = nil;
        }
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception while unloading: %@", exception);
        p->model = nil;
    }
    if (p->scratchDir) {
        [NSFileManager.defaultManager removeItemAtPath:p->scratchDir error:nil];
        p->scratchDir = nil;
    }
    free(p);
}

int h3_ane_program_s(H3ANEProgram *p) { return p ? p->s : 0; }
int h3_ane_program_k(H3ANEProgram *p) { return p ? p->k : 0; }
int h3_ane_program_n(H3ANEProgram *p) { return p ? p->n : 0; }

bool h3_ane_run(H3ANEProgram *p, H3ANETensor *x, H3ANETensor *w, H3ANETensor *y,
                int instance_hint) {
    if (!p || !p->model || !x || !w || !y) return false;
    // The engine reads whatever the surfaces hold; if they were allocated for
    // a different shape than the program was compiled for it produces numbers
    // rather than an error, so check here.
    if (x->rows != p->k || x->width != p->s) return false;
    if (w->rows != p->k || w->width != p->n) return false;
    if (y->rows != p->s || y->width != p->n) return false;

    @try {
    id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        RequestClass,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[x->object, w->object], @[@0, @1], @[y->object], @[@0], nil, nil, @0);
    if (!request) return false;

    NSDictionary *opts = @{
        @"kANEFProcedureVariantHint": @1,
        @"kANEFAneInstanceHint":      @(instance_hint)
    };

    NSError *error = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        p->model, @selector(evaluateWithQoS:options:request:error:), 21, opts, request, &error);
    if (!ok) NSLog(@"[H3ANEBridge] evaluate failed: %@", error);
    return ok;
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception while evaluating: %@", exception);
        return false;
    }
}

bool h3_ane_run_pair(H3ANEProgram *p0, H3ANETensor *x0, H3ANETensor *w0, H3ANETensor *y0,
                     H3ANEProgram *p1, H3ANETensor *x1, H3ANETensor *w1, H3ANETensor *y1) {
    __block bool ok0 = false, ok1 = false;
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_group_t group = dispatch_group_create();

    dispatch_group_async(group, q, ^{ ok0 = h3_ane_run(p0, x0, w0, y0, 1); });
    // The second shard runs here rather than on another queue slot so one of
    // the two always makes progress even under a saturated global queue.
    ok1 = h3_ane_run(p1, x1, w1, y1, 2);
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);

    return ok0 && ok1;
}
