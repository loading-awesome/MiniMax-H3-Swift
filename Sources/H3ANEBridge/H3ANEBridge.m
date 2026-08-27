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

#pragma mark - Native Metal Pack & Merge Operations (Phases 3 & 4)

#import <Metal/Metal.h>

static id<MTLDevice> gMetalDevice = nil;
static id<MTLCommandQueue> gCommandQueue = nil;
static id<MTLComputePipelineState> gPackPipelineState = nil;
static id<MTLComputePipelineState> gMergeAttnPipelineState = nil;
static id<MTLComputePipelineState> gMergeFC1PipelineState = nil;
static dispatch_once_t gMetalOnceToken;

static bool InitMetalKernels(void) {
    dispatch_once(&gMetalOnceToken, ^{
        gMetalDevice = MTLCreateSystemDefaultDevice();
        if (!gMetalDevice) return;
        gCommandQueue = [gMetalDevice newCommandQueue];

        NSError *error = nil;
        NSString *metalSource = @""
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "inline ushort h3_bf16_rne(float v) {\n"
            "    uint b = as_type<uint>(v);\n"
            "    return (ushort)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);\n"
            "}\n"
            "inline float h3_bf16_value(float v) {\n"
            "    return as_type<float>(((uint)h3_bf16_rne(v)) << 16);\n"
            "}\n"
            "inline float h3_read_shard(device const ushort* gpu, device const half* a0,\n"
            "    device const half* a1, uint row, uint col, uint ng, uint n0, uint n1) {\n"
            "    if (col < ng) return as_type<float>(((uint)gpu[row * ng + col]) << 16);\n"
            "    if (col < ng + n0) return h3_bf16_value(((float)a0[row * n0 + col - ng]) * 16.0f);\n"
            "    return h3_bf16_value(((float)a1[row * n1 + col - ng - n0]) * 16.0f);\n"
            "}\n"
            "kernel void h3_pack_bf16_to_fp16_transpose(\n"
            "    device const ushort* src_bf16 [[buffer(0)]],\n"
            "    device half* dst_fp16         [[buffer(1)]],\n"
            "    constant uint& s              [[buffer(2)]],\n"
            "    constant uint& k              [[buffer(3)]],\n"
            "    constant uint& padded_s       [[buffer(4)]],\n"
            "    uint2 gid                     [[thread_position_in_grid]]\n"
            ") {\n"
            "    uint row = gid.y; uint col = gid.x;\n"
            "    if (row >= s || col >= k) return;\n"
            "    ushort u = src_bf16[row * k + col];\n"
            "    float val = as_type<float>(((uint)u) << 16);\n"
            "    half scaled = (half)(val * 0.0625f);\n"
            "    dst_fp16[col * padded_s + row] = scaled;\n"
            "}\n"
            "kernel void h3_merge_attn_out(\n"
            "    device const ushort* gpu_suffix_bf16 [[buffer(0)]],\n"
            "    device const half* ane0_fp16         [[buffer(1)]],\n"
            "    device const half* ane1_fp16         [[buffer(2)]],\n"
            "    device ushort* dst_bf16              [[buffer(3)]],\n"
            "    constant uint& s                     [[buffer(4)]],\n"
            "    constant uint& n_gpu                 [[buffer(5)]],\n"
            "    constant uint& n_ane0                [[buffer(6)]],\n"
            "    constant uint& n_ane1                [[buffer(7)]],\n"
            "    uint2 gid                            [[thread_position_in_grid]]\n"
            ") {\n"
            "    uint row = gid.y; uint col = gid.x;\n"
            "    uint n_total = n_gpu + n_ane0 + n_ane1;\n"
            "    if (row >= s || col >= n_total) return;\n"
            "    float val = 0.0f;\n"
            "    if (col < n_gpu) {\n"
            "        ushort u = gpu_suffix_bf16[row * n_gpu + col];\n"
            "        val = as_type<float>(((uint)u) << 16);\n"
            "    } else if (col < n_gpu + n_ane0) {\n"
            "        uint c = col - n_gpu;\n"
            "        val = ((float)ane0_fp16[row * n_ane0 + c]) * 16.0f;\n"
            "    } else {\n"
            "        uint c = col - (n_gpu + n_ane0);\n"
            "        val = ((float)ane1_fp16[row * n_ane1 + c]) * 16.0f;\n"
            "    }\n"
            "    dst_bf16[row * n_total + col] = h3_bf16_rne(val);\n"
            "}\n"
            "kernel void h3_merge_fc1_swiglu(\n"
            "    device const ushort* gpu_suffix_bf16 [[buffer(0)]],\n"
            "    device const half* ane0_fp16         [[buffer(1)]],\n"
            "    device const half* ane1_fp16         [[buffer(2)]],\n"
            "    device ushort* dst_bf16              [[buffer(3)]],\n"
            "    constant uint& s                     [[buffer(4)]],\n"
            "    constant uint& n_gpu                 [[buffer(5)]],\n"
            "    constant uint& n_ane0                [[buffer(6)]],\n"
            "    constant uint& n_ane1                [[buffer(7)]],\n"
            "    uint2 gid                            [[thread_position_in_grid]]\n"
            ") {\n"
            "    uint row = gid.y; uint col = gid.x;\n"
            "    uint ffn_dim = (n_gpu + n_ane0 + n_ane1) / 2;\n"
            "    if (row >= s || col >= ffn_dim) return;\n"
            "    uint gate_col = col; uint up_col = col + ffn_dim;\n"
            "    float g = h3_read_shard(gpu_suffix_bf16, ane0_fp16, ane1_fp16,\n"
            "                            row, gate_col, n_gpu, n_ane0, n_ane1);\n"
            "    float u = h3_read_shard(gpu_suffix_bf16, ane0_fp16, ane1_fp16,\n"
            "                            row, up_col, n_gpu, n_ane0, n_ane1);\n"
            "    float tail = 1.0f / (1.0f + exp(abs(g)));\n"
            "    float sigmoid_g = h3_bf16_value(g < 0.0f ? tail : 1.0f - tail);\n"
            "    float silu_g = h3_bf16_value(g * sigmoid_g);\n"
            "    dst_bf16[row * ffn_dim + col] = h3_bf16_rne(silu_g * u);\n"
            "}\n";

        id<MTLLibrary> lib = [gMetalDevice newLibraryWithSource:metalSource options:nil error:&error];
        if (!lib) { NSLog(@"[H3ANEBridge] Metal lib init failed: %@", error); return; }

        id<MTLFunction> fnPack = [lib newFunctionWithName:@"h3_pack_bf16_to_fp16_transpose"];
        id<MTLFunction> fnAttn = [lib newFunctionWithName:@"h3_merge_attn_out"];
        id<MTLFunction> fnFC1  = [lib newFunctionWithName:@"h3_merge_fc1_swiglu"];

        if (fnPack) gPackPipelineState = [gMetalDevice newComputePipelineStateWithFunction:fnPack error:nil];
        if (fnAttn) gMergeAttnPipelineState = [gMetalDevice newComputePipelineStateWithFunction:fnAttn error:nil];
        if (fnFC1)  gMergeFC1PipelineState  = [gMetalDevice newComputePipelineStateWithFunction:fnFC1 error:nil];
    });

    return (gMetalDevice && gCommandQueue && gPackPipelineState && gMergeAttnPipelineState && gMergeFC1PipelineState);
}

bool h3_ane_pack_bf16_to_fp16_transpose(const void *srcBF16, H3ANETensor *dstTensor, int s, int k, void *commandQueue) {
    if (!srcBF16 || !dstTensor) return false;
    if (!InitMetalKernels()) return false;

    id<MTLCommandQueue> queue = commandQueue ? (__bridge id<MTLCommandQueue>)commandQueue : gCommandQueue;
    void *dstPtr = h3_ane_tensor_ptr(dstTensor);
    if (!dstPtr) return false;

    uint uS = (uint)s;
    uint uK = (uint)k;
    uint uPaddedS = (uint)dstTensor->width;

    size_t srcBytes = s * k * 2;
    size_t dstBytes = dstTensor->rows * dstTensor->rowBytes;

    id<MTLBuffer> srcBuf = [gMetalDevice newBufferWithBytesNoCopy:(void*)srcBF16 length:srcBytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> dstBuf = [gMetalDevice newBufferWithBytesNoCopy:dstPtr length:dstBytes options:MTLResourceStorageModeShared deallocator:nil];

    if (!srcBuf || !dstBuf) return false;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:gPackPipelineState];
    [enc setBuffer:srcBuf offset:0 atIndex:0];
    [enc setBuffer:dstBuf offset:0 atIndex:1];
    [enc setBytes:&uS length:sizeof(uint) atIndex:2];
    [enc setBytes:&uK length:sizeof(uint) atIndex:3];
    [enc setBytes:&uPaddedS length:sizeof(uint) atIndex:4];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake((k + 15) / 16 * 16, (s + 15) / 16 * 16, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return (cmd.status == MTLCommandBufferStatusCompleted);
}

bool h3_ane_merge_attn_out(const void *gpuSuffixBF16, H3ANETensor *ane0Tensor, H3ANETensor *ane1Tensor, void *dstBF16, int s, int nGpu, int nAne0, int nAne1, void *commandQueue) {
    if (!gpuSuffixBF16 || !ane0Tensor || !ane1Tensor || !dstBF16) return false;
    if (!InitMetalKernels()) return false;

    id<MTLCommandQueue> queue = commandQueue ? (__bridge id<MTLCommandQueue>)commandQueue : gCommandQueue;
    void *ane0Ptr = h3_ane_tensor_ptr(ane0Tensor);
    void *ane1Ptr = h3_ane_tensor_ptr(ane1Tensor);
    if (!ane0Ptr || !ane1Ptr) return false;

    uint uS = (uint)s;
    uint uNGpu = (uint)nGpu;
    uint uNAne0 = (uint)nAne0;
    uint uNAne1 = (uint)nAne1;
    uint nTotal = uNGpu + uNAne0 + uNAne1;

    size_t gpuBytes = s * nGpu * 2;
    size_t ane0Bytes = ane0Tensor->rows * ane0Tensor->rowBytes;
    size_t ane1Bytes = ane1Tensor->rows * ane1Tensor->rowBytes;
    size_t dstBytes = s * nTotal * 2;

    id<MTLBuffer> gpuBuf = [gMetalDevice newBufferWithBytesNoCopy:(void*)gpuSuffixBF16 length:gpuBytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ane0Buf = [gMetalDevice newBufferWithBytesNoCopy:ane0Ptr length:ane0Bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ane1Buf = [gMetalDevice newBufferWithBytesNoCopy:ane1Ptr length:ane1Bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> dstBuf = [gMetalDevice newBufferWithBytesNoCopy:dstBF16 length:dstBytes options:MTLResourceStorageModeShared deallocator:nil];

    if (!gpuBuf || !ane0Buf || !ane1Buf || !dstBuf) return false;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:gMergeAttnPipelineState];
    [enc setBuffer:gpuBuf offset:0 atIndex:0];
    [enc setBuffer:ane0Buf offset:0 atIndex:1];
    [enc setBuffer:ane1Buf offset:0 atIndex:2];
    [enc setBuffer:dstBuf offset:0 atIndex:3];
    [enc setBytes:&uS length:sizeof(uint) atIndex:4];
    [enc setBytes:&uNGpu length:sizeof(uint) atIndex:5];
    [enc setBytes:&uNAne0 length:sizeof(uint) atIndex:6];
    [enc setBytes:&uNAne1 length:sizeof(uint) atIndex:7];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake((nTotal + 15) / 16 * 16, (s + 15) / 16 * 16, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return (cmd.status == MTLCommandBufferStatusCompleted);
}

bool h3_ane_merge_fc1_swiglu(const void *gpuSuffixBF16, H3ANETensor *ane0Tensor, H3ANETensor *ane1Tensor, void *dstBF16, int s, int nGpu, int nAne0, int nAne1, void *commandQueue) {
    if (!gpuSuffixBF16 || !ane0Tensor || !ane1Tensor || !dstBF16) return false;
    if (!InitMetalKernels()) return false;

    id<MTLCommandQueue> queue = commandQueue ? (__bridge id<MTLCommandQueue>)commandQueue : gCommandQueue;
    void *ane0Ptr = h3_ane_tensor_ptr(ane0Tensor);
    void *ane1Ptr = h3_ane_tensor_ptr(ane1Tensor);
    if (!ane0Ptr || !ane1Ptr) return false;

    uint uS = (uint)s;
    uint uNGpu = (uint)nGpu;
    uint uNAne0 = (uint)nAne0;
    uint uNAne1 = (uint)nAne1;
    uint nTotal = uNGpu + uNAne0 + uNAne1;
    uint ffnDim = nTotal / 2;

    size_t gpuBytes = s * nGpu * 2;
    size_t ane0Bytes = ane0Tensor->rows * ane0Tensor->rowBytes;
    size_t ane1Bytes = ane1Tensor->rows * ane1Tensor->rowBytes;
    size_t dstBytes = s * ffnDim * 2;

    id<MTLBuffer> gpuBuf = [gMetalDevice newBufferWithBytesNoCopy:(void*)gpuSuffixBF16 length:gpuBytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ane0Buf = [gMetalDevice newBufferWithBytesNoCopy:ane0Ptr length:ane0Bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ane1Buf = [gMetalDevice newBufferWithBytesNoCopy:ane1Ptr length:ane1Bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> dstBuf = [gMetalDevice newBufferWithBytesNoCopy:dstBF16 length:dstBytes options:MTLResourceStorageModeShared deallocator:nil];

    if (!gpuBuf || !ane0Buf || !ane1Buf || !dstBuf) return false;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:gMergeFC1PipelineState];
    [enc setBuffer:gpuBuf offset:0 atIndex:0];
    [enc setBuffer:ane0Buf offset:0 atIndex:1];
    [enc setBuffer:ane1Buf offset:0 atIndex:2];
    [enc setBuffer:dstBuf offset:0 atIndex:3];
    [enc setBytes:&uS length:sizeof(uint) atIndex:4];
    [enc setBytes:&uNGpu length:sizeof(uint) atIndex:5];
    [enc setBytes:&uNAne0 length:sizeof(uint) atIndex:6];
    [enc setBytes:&uNAne1 length:sizeof(uint) atIndex:7];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake((ffnDim + 15) / 16 * 16, (s + 15) / 16 * 16, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return (cmd.status == MTLCommandBufferStatusCompleted);
}
