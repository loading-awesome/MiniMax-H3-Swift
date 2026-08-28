// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

#import "H3ANEBridge.h"
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <IOSurface/IOSurface.h>
#import <math.h>
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

    // Kept as a construction-time check that the private class accepts this
    // surface — a clear failure here beats one inside an evaluation. It is
    // deliberately **not** used to build requests: see `h3_ane_run`, where each
    // evaluation wraps the surface itself so no object is ever bound to two
    // concurrent requests on two dies.
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
            // Builds that have passed both gates: the selector audit
            // (`Tools/ANE/abi-check.m`) and a clean `Tools/ANE/pair-stress.m`
            // watch past +900 s.
            //
            // **25F84 and earlier are deliberately absent.** They carry a
            // driver defect that admits a request arriving during a power
            // transition and hard-locks the machine minutes later, with no
            // symptom at submission time. See "Machine safety" in
            // docs/ANE.md. There is no configuration of this bridge that
            // is safe there, so it does not route.
            //
            // The trailing letter on 26A5421a says GM seed; the shipping 27.0
            // build string will differ and has to be re-audited, not assumed.
            NSArray<NSString *> *validated = @[@"26A5421a"];
            if (![validated containsObject:SysctlString("kern.osversion")] ||
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
    H3ANEForm form;
    _Atomic bool poisoned;
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


/// The same linear as a 1x1 convolution, which is what the engine is.
///
/// `a` is `[1,k,1,s]` — k channels, one row, s columns — and the kernel is
/// 1x1, so every output column is `w @ a[:,col]`: exactly the projection,
/// with the sequence riding the spatial axis where the engine expects it.
/// Nothing is transposed. `MatmulMIL` moves the whole activation before it
/// multiplies; this form does not, and that is the difference between 3.87
/// and 5.4 TFLOP/s a die.
///
/// The weight is `[n,k,1,1]`, so the surface is `[n,k]` — the orientation the
/// checkpoint already holds, which also removes the host-side transpose the
/// matmul form needs when a weight is uploaded.
static NSString *ConvMIL(int s, int k, int n) {
    NSMutableString *m = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16,[1,%d,1,%d]> a, tensor<fp16,[1,%d,1,%d]> w) {\n",
                    k, s, n, k];
    [m appendFormat:@"        tensor<int32,[4]> rw=const()[name=string(\"rw\"),"
                     "val=tensor<int32,[4]>([%d,%d,1,1])];\n"
                     "        tensor<fp16,[%d,%d,1,1]> w4=reshape(shape=rw,x=w)[name=string(\"w4\")];\n",
                    n, k, n, k];
    [m appendString:@"        tensor<int32,[2]> st=const()[name=string(\"st\"),"
                     "val=tensor<int32,[2]>([1,1])];\n"
                     "        tensor<int32,[2]> dl=const()[name=string(\"dl\"),"
                     "val=tensor<int32,[2]>([1,1])];\n"
                     "        tensor<int32,[4]> pd=const()[name=string(\"pd\"),"
                     "val=tensor<int32,[4]>([0,0,0,0])];\n"
                     "        string pt=const()[name=string(\"pt\"),val=string(\"valid\")];\n"
                     "        int32 gp=const()[name=string(\"gp\"),val=int32(1)];\n"];
    [m appendFormat:@"        tensor<fp16,[1,%d,1,%d]> y=conv(dilations=dl,groups=gp,pad=pd,"
                     "pad_type=pt,strides=st,weight=w4,x=a)[name=string(\"y\")];\n"
                     "    } -> (y);\n}\n",
                    n, s];
    return m;
}

H3ANEProgram* h3_ane_program_create(int s, int k, int n) {
    return h3_ane_program_create_form(s, k, n, H3ANEFormMatmul);
}

H3ANEForm h3_ane_program_form(H3ANEProgram *p) { return p ? p->form : H3ANEFormMatmul; }

H3ANEProgram* h3_ane_program_create_form(int s, int k, int n, H3ANEForm form) {
    if (!h3_ane_is_available() || s <= 0 || k <= 0 || n <= 0) return NULL;

    H3ANEProgram *p = NULL;
    @try {
    p = (H3ANEProgram *)calloc(1, sizeof(H3ANEProgram));
    if (!p) return NULL;
    p->s = s; p->k = k; p->n = n; p->form = form;

    NSString *mil = (form == H3ANEFormConv) ? ConvMIL(s, k, n) : MatmulMIL(s, k, n);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

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
        NSLog(@"[H3ANEBridge] compile failed for s=%d k=%d n=%d form=%d: %@", s, k, n, (int)form, error);
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

#pragma mark - Power-transition cancellations

/// 27.0 refuses a request that arrives while the driver is mid power
/// transition, rather than admitting it against a DART whose mappings are not
/// installed — the defect that panicked 25F84. The refusal is transient and,
/// crucially, the request never ran: nothing was submitted, no surface was
/// touched. It is the one failure that is safe to simply retry.
///
/// This matters because the callers treat any failure as evidence that the
/// runtime may still own their surfaces, and retire the session permanently.
/// At process start the engine is cold, so the first submissions are exactly
/// the ones that get cancelled — which retired the route before it had run a
/// single evaluation.
static _Atomic uint64_t CancellationRetries = 0;
static _Atomic uint64_t CancellationGiveUps = 0;

static bool ErrorIsCancellation(NSError *error) {
    if (!error || error.code != 34) return false;
    NSString *text = error.localizedDescription ?: @"";
    return [text containsString:@"Request cancelled"];
}

static BOOL EvaluateAbsorbingCancellation(id model, NSDictionary *options,
                                          id request, const char *what) {
    static dispatch_once_t once;
    static int budget = 8;
    dispatch_once(&once, ^{
        NSString *raw = NSProcessInfo.processInfo.environment[@"H3_ANE_CANCEL_RETRIES"];
        if (raw.length > 0) budget = MAX(0, raw.intValue);
    });
    NSError *error = nil;
    for (int attempt = 0; attempt <= budget; ++attempt) {
        error = nil;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, options, request, &error);
        if (ok) {
            if (attempt > 0) atomic_fetch_add(&CancellationRetries, (uint64_t)attempt);
            return YES;
        }
        if (!ErrorIsCancellation(error)) break;
        // The transition it collides with measures 8-15 ms. Sleep inside that
        // window rather than spinning, and rather than giving up on one refusal.
        usleep(3000);
    }
    if (ErrorIsCancellation(error)) atomic_fetch_add(&CancellationGiveUps, 1);
    NSLog(@"[H3ANEBridge] %s evaluate failed: %@", what, error);
    return NO;
}

uint64_t h3_ane_cancellation_retries(void) { return atomic_load(&CancellationRetries); }
uint64_t h3_ane_cancellation_giveups(void) { return atomic_load(&CancellationGiveUps); }

#pragma mark - Fused attention programs

struct H3ANEAttentionProgram {
    int heads, sequence, headDim;
    _Atomic bool poisoned;
    id _Nullable model;
    NSString * _Nullable scratchDir;
    /// Which of q,k,v (0,1,2) belongs at each compiled input index. The
    /// compiler renames function arguments to alphabetically ordered input
    /// symbols, so textual order is not the ABI; this is read from the load
    /// reply rather than assumed.
    int binding[3];
};

/// Stable SDPA with the quadratic score plane kept entirely inside ANE.
///
/// The fused MIL `softmax` lowering normalises individual tiles at long
/// sequence lengths. Expressing max/sub/exp/sum/div separately keeps the
/// reduction global.
///
/// `scores` is the ordinary `q @ k^T`. An earlier version wrote `k @ q^T` and
/// explained it as the compiler physically transposing a square matmul's
/// output against its declared type. That was a misdiagnosis. The load reply
/// shows this compiler renames function arguments to alphabetically ordered
/// input symbols — `(k, q, v)` at indices `(0, 1, 2)` — so binding textual
/// order swapped q and k, and the reversed matmul cancelled the swap. Both
/// halves are now correct rather than compensating: the graph is natural and
/// `h3_ane_attention_run` binds by the symbol order in the load reply.
static NSString *AttentionMIL(int heads, int sequence, int dimension) {
    NSMutableString *m = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:
        @"    func main<ios18>(tensor<fp16,[1,%d,%d,%d]> q, "
         "tensor<fp16,[1,%d,%d,%d]> k, tensor<fp16,[1,%d,%d,%d]> v) {\n",
        heads, sequence, dimension, heads, sequence, dimension,
        heads, sequence, dimension];
    [m appendString:
        @"        tensor<int32,[4]> pk=const()[name=string(\"pk\"),"
         "val=tensor<int32,[4]>([0,1,3,2])];\n"];
    [m appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> kt=transpose(perm=pk,x=k)"
         "[name=string(\"kt\")];\n", heads, dimension, sequence];
    [m appendString:
        @"        bool bf=const()[name=string(\"bf\"),val=bool(false)];\n"
         "        bool kd=const()[name=string(\"kd\"),val=bool(true)];\n"
         "        tensor<int32,[1]> ax=const()[name=string(\"ax\"),"
         "val=tensor<int32,[1]>([3])];\n"];
    [m appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> scores=matmul("
         "transpose_x=bf,transpose_y=bf,x=q,y=kt)[name=string(\"scores\")];\n"
         "        tensor<fp16,[1,%d,%d,1]> mx=reduce_max(axes=ax,keep_dims=kd,"
         "x=scores)[name=string(\"mx\")];\n"
         "        tensor<fp16,[1,%d,%d,%d]> sh=sub(x=scores,y=mx)"
         "[name=string(\"sh\")];\n"
         "        tensor<fp16,[1,%d,%d,%d]> weights=exp(x=sh)"
         "[name=string(\"weights\")];\n"
         "        tensor<fp16,[1,%d,%d,1]> denominator=reduce_sum("
         "axes=ax,keep_dims=kd,x=weights)[name=string(\"denominator\")];\n"
         "        tensor<fp16,[1,%d,%d,%d]> weighted=matmul("
         "transpose_x=bf,transpose_y=bf,x=weights,y=v)"
         "[name=string(\"weighted\")];\n"
         "        tensor<fp16,[1,%d,%d,%d]> y=real_div(x=weighted,y=denominator)"
         "[name=string(\"y\")];\n"
         "    } -> (y);\n}\n",
        heads, sequence, sequence,
        heads, sequence,
        heads, sequence, sequence,
        heads, sequence, sequence,
        heads, sequence,
        heads, sequence, dimension,
        heads, sequence, dimension];
    return m;
}

H3ANEAttentionProgram* h3_ane_attention_program_create(
    int heads, int sequence, int head_dim) {
    if (!h3_ane_is_available() || heads <= 0 || sequence <= 0 || head_dim <= 0) return NULL;
    H3ANEAttentionProgram *p = NULL;
    @try {
        p = (H3ANEAttentionProgram *)calloc(1, sizeof(H3ANEAttentionProgram));
        if (!p) return NULL;
        p->heads = heads; p->sequence = sequence; p->headDim = head_dim;
        NSString *mil = AttentionMIL(heads, sequence, head_dim);
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            DescriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        if (!descriptor) { h3_ane_attention_program_free(p); return NULL; }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            ModelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) { h3_ane_attention_program_free(p); return NULL; }

        static atomic_uint serial = 0;
        unsigned mine = atomic_fetch_add(&serial, 1);
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier));
        p->scratchDir = [NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"h3-ane-attn-%d-%u-%@", getpid(), mine, identifier]];
        NSFileManager *fm = NSFileManager.defaultManager;
        [fm createDirectoryAtPath:[p->scratchDir stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[p->scratchDir stringByAppendingPathComponent:@"model.mil"]
                     options:0 error:nil];
        if ([model respondsToSelector:@selector(setModelURL:)]) {
            ((void(*)(id, SEL, id))objc_msgSend)(
                model, @selector(setModelURL:), [NSURL fileURLWithPath:p->scratchDir]);
        }
        NSError *error = nil;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        if (!ok) {
            NSLog(@"[H3ANEBridge] attention compile failed h=%d s=%d d=%d: %@",
                  heads, sequence, head_dim, error);
            h3_ane_attention_program_free(p); return NULL;
        }
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        if (!ok) {
            NSLog(@"[H3ANEBridge] attention load failed h=%d s=%d d=%d: %@",
                  heads, sequence, head_dim, error);
            h3_ane_attention_program_free(p); return NULL;
        }
        // Read the compiled input order from the load reply. Guessing it
        // cost a day on the MLP island: the compiler emits alphabetically
        // ordered symbols, so a textual q,k,v binding is silently permuted.
        p->binding[0] = 1; p->binding[1] = 0; p->binding[2] = 2;  // (k,q,v)
        if ([model respondsToSelector:@selector(modelAttributes)]) {
            id attributes = ((id(*)(id, SEL))objc_msgSend)(
                model, @selector(modelAttributes));
            NSArray *symbols = [attributes isKindOfClass:NSDictionary.class]
                ? ((NSDictionary *)attributes)[@"kANEFModelInputSymbolsArrayKey"] : nil;
            if ([symbols isKindOfClass:NSArray.class] && symbols.count == 3) {
                int resolved[3] = {-1, -1, -1};
                for (NSUInteger i = 0; i < 3; ++i) {
                    NSString *name = [symbols[i] description];
                    if ([name isEqualToString:@"q"]) resolved[i] = 0;
                    else if ([name isEqualToString:@"k"]) resolved[i] = 1;
                    else if ([name isEqualToString:@"v"]) resolved[i] = 2;
                }
                if (resolved[0] >= 0 && resolved[1] >= 0 && resolved[2] >= 0 &&
                    resolved[0] != resolved[1] && resolved[1] != resolved[2] &&
                    resolved[0] != resolved[2]) {
                    for (int i = 0; i < 3; ++i) p->binding[i] = resolved[i];
                } else {
                    NSLog(@"[H3ANEBridge] attention input symbols unrecognised: %@", symbols);
                    h3_ane_attention_program_free(p); return NULL;
                }
            }
        }
        p->model = model;
        return p;
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception compiling attention: %@", exception);
        h3_ane_attention_program_free(p); return NULL;
    }
}

void h3_ane_attention_program_free(H3ANEAttentionProgram *p) {
    if (!p) return;
    @try {
        if (p->model) {
            NSError *error = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                p->model, @selector(unloadWithQoS:error:), 21, &error);
            p->model = nil;
        }
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception unloading attention: %@", exception);
        p->model = nil;
    }
    if (p->scratchDir) {
        [NSFileManager.defaultManager removeItemAtPath:p->scratchDir error:nil];
        p->scratchDir = nil;
    }
    free(p);
}

bool h3_ane_attention_run(H3ANEAttentionProgram *p,
                          H3ANETensor *q, H3ANETensor *k,
                          H3ANETensor *v, H3ANETensor *y,
                          int instance_hint) {
    if (!p || !p->model || !q || !k || !v || !y || atomic_load(&p->poisoned)) return false;
    int rows = p->heads * p->sequence;
    if (q->rows != rows || k->rows != rows || v->rows != rows || y->rows != rows ||
        q->width != p->headDim || k->width != p->headDim ||
        v->width != p->headDim || y->width != p->headDim) return false;
    @try {
        // Ordered by the compiled input symbols read at load, not by the
        // order the MIL function declares its arguments in.
        id supplied[3] = {q->object, k->object, v->object};
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            RequestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[supplied[p->binding[0]], supplied[p->binding[1]], supplied[p->binding[2]]],
            @[@0, @1, @2], @[y->object], @[@0], nil, nil, @0);
        if (!request) return false;
        NSDictionary *options = @{
            @"kANEFProcedureVariantHint": @1,
            @"kANEFAneInstanceHint": @(instance_hint)
        };
        return EvaluateAbsorbingCancellation(p->model, options, request, "attention");
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception evaluating attention: %@", exception);
        atomic_store(&p->poisoned, true);
        return false;
    }
}

bool h3_ane_attention_run_pair(
    H3ANEAttentionProgram *p0, H3ANETensor *q0, H3ANETensor *k0,
    H3ANETensor *v0, H3ANETensor *y0,
    H3ANEAttentionProgram *p1, H3ANETensor *q1, H3ANETensor *k1,
    H3ANETensor *v1, H3ANETensor *y1) {
    if (!p0 || !p1) return false;
    __block bool ok0 = false, ok1 = false;
    dispatch_group_t group = dispatch_group_create();
    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool { ok0 = h3_ane_attention_run(p0, q0, k0, v0, y0, 1); }
    });
    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool { ok1 = h3_ane_attention_run(p1, q1, k1, v1, y1, 2); }
    });
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    return ok0 && ok1;
}

bool h3_ane_run(H3ANEProgram *p, H3ANETensor *x, H3ANETensor *w, H3ANETensor *y,
                int instance_hint) {
    if (!p || !p->model || !x || !w || !y || atomic_load(&p->poisoned)) return false;
    // The engine reads whatever the surfaces hold; if they were allocated for
    // a different shape than the program was compiled for it produces numbers
    // rather than an error, so check here.
    if (x->rows != p->k || x->width != p->s) return false;
    if (p->form == H3ANEFormConv) {
        if (w->rows != p->n || w->width != p->k) return false;
        if (y->rows != p->n || y->width != p->s) return false;
    } else {
        if (w->rows != p->k || w->width != p->n) return false;
        if (y->rows != p->s || y->width != p->n) return false;
    }

    @try {
    // **Sharing one `_ANEIOSurfaceObject` across both dies is not a fault.**
    // `h3_ane_job_submit_pair` runs two evaluations concurrently against two
    // dies, each with its own DART, and every caller binds the same input
    // tensor to both. The theory was that this races whatever per-evaluation
    // mapping state the private class holds. Measured head to head at
    // production submission rates on 2026-08-27:
    //
    //   shared object      671,438 pairs in 180 s, 0 failures
    //   per-request object 636,432 pairs in 180 s, 0 failures
    //
    // So the shared object stays — simpler code, and the change was not
    // earned. `H3_ANE_PER_REQUEST_SURFACE_OBJECT=1` keeps the alternative for a
    // future run with reason to suspect it.
    static dispatch_once_t shareOnce;
    static bool perRequest = false;
    dispatch_once(&shareOnce, ^{
        perRequest = [NSProcessInfo.processInfo.environment[@"H3_ANE_PER_REQUEST_SURFACE_OBJECT"]
                          isEqualToString:@"1"];
    });

    id xObject, wObject, yObject;
    if (!perRequest) {
        xObject = x->object; wObject = w->object; yObject = y->object;
    } else {
        xObject = ((id(*)(Class, SEL, IOSurfaceRef, size_t))objc_msgSend)(
            IOSurfaceObjectClass, @selector(objectWithIOSurface:startOffset:), x->surface, 0);
        wObject = ((id(*)(Class, SEL, IOSurfaceRef, size_t))objc_msgSend)(
            IOSurfaceObjectClass, @selector(objectWithIOSurface:startOffset:), w->surface, 0);
        yObject = ((id(*)(Class, SEL, IOSurfaceRef, size_t))objc_msgSend)(
            IOSurfaceObjectClass, @selector(objectWithIOSurface:startOffset:), y->surface, 0);
    }
    if (!xObject || !wObject || !yObject) return false;

    id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        RequestClass,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[xObject, wObject], @[@0, @1], @[yObject], @[@0], nil, nil, @0);
    if (!request) return false;

    NSDictionary *opts = @{
        @"kANEFProcedureVariantHint": @1,
        @"kANEFAneInstanceHint":      @(instance_hint)
    };

    return EvaluateAbsorbingCancellation(p->model, opts, request, "linear");
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception while evaluating: %@", exception);
        return false;
    }
}

bool h3_ane_run_pair(H3ANEProgram *p0, H3ANETensor *x0, H3ANETensor *w0, H3ANETensor *y0,
                     H3ANEProgram *p1, H3ANETensor *x1, H3ANETensor *w1, H3ANETensor *y1) {
    H3ANEJob *job = h3_ane_job_submit_pair(p0, x0, w0, y0, p1, x1, w1, y1);
    if (!job) return false;
    // Bound a wedged private driver without mistaking first-use JIT for a
    // failure. The steady-state path is far below this limit.
    bool ok = h3_ane_job_wait(job, 30000000000ULL);
    h3_ane_job_retire(job);
    return ok;
}

#pragma mark - Asynchronous Bounded Job API (Phase 2)

struct H3ANEJob {
    _Atomic H3ANEJobState state;
    _Atomic uint32_t references;
    uint64_t submitTime;
    uint64_t completeTime;
    dispatch_group_t _Nullable group;
    H3ANEProgram *p0;
    H3ANEProgram *p1;
    bool ok0;
    bool ok1;
};

static void H3ANEJobRelease(H3ANEJob *job) {
    if (atomic_fetch_sub(&job->references, 1) == 1) free(job);
}

H3ANEJob* h3_ane_job_submit_pair(H3ANEProgram *p0, H3ANETensor *x0, H3ANETensor *w0, H3ANETensor *y0,
                                 H3ANEProgram *p1, H3ANETensor *x1, H3ANETensor *w1, H3ANETensor *y1) {
    if (!p0 || !p1 || !x0 || !x1 || !w0 || !w1 || !y0 || !y1) return NULL;

    H3ANEJob *job = (H3ANEJob *)calloc(1, sizeof(H3ANEJob));
    if (!job) return NULL;

    job->state = H3ANEJobStateRunning;
    // The caller and both blocks own the job. A timeout may retire the caller
    // while evaluateWithQoS is still blocked; the worker references prevent a
    // use-after-free until the private calls really return.
    job->references = 3;
    job->submitTime = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
    job->group = dispatch_group_create();
    job->p0 = p0;
    job->p1 = p1;

    dispatch_queue_t q0 = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_queue_t q1 = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

    dispatch_group_async(job->group, q0, ^{
        job->ok0 = h3_ane_run(p0, x0, w0, y0, 1);
        H3ANEJobRelease(job);
    });
    dispatch_group_async(job->group, q1, ^{
        job->ok1 = h3_ane_run(p1, x1, w1, y1, 2);
        H3ANEJobRelease(job);
    });

    return job;
}

bool h3_ane_job_wait(H3ANEJob *job, uint64_t timeout_ns) {
    if (!job) return false;
    if (job->state == H3ANEJobStateComplete) return (job->ok0 && job->ok1);
    if (job->state == H3ANEJobStateError) return false;

    dispatch_time_t t = (timeout_ns == 0) ? DISPATCH_TIME_FOREVER : dispatch_time(DISPATCH_TIME_NOW, timeout_ns);
    long res = dispatch_group_wait(job->group, t);

    if (res == 0) {
        job->completeTime = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
        bool ok = (job->ok0 && job->ok1);
        job->state = ok ? H3ANEJobStateComplete : H3ANEJobStateError;
        return ok;
    } else {
        // There is no cancellation selector. Refuse future submissions to the
        // two programs; the caller must quarantine the mutable surfaces that
        // these evaluations may still be reading and writing.
        NSLog(@"[H3ANEBridge] Job evaluation timed out after %llu ns", timeout_ns);
        atomic_store(&job->p0->poisoned, true);
        atomic_store(&job->p1->poisoned, true);
        job->state = H3ANEJobStateError;
        return false;
    }
}

H3ANEJobState h3_ane_job_status(H3ANEJob *job) {
    return job ? job->state : H3ANEJobStateError;
}

void h3_ane_job_retire(H3ANEJob *job) {
    if (!job) return;
    H3ANEJobRelease(job);
}
