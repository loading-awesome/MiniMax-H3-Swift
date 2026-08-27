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
#import <os/lock.h>

// The private surface holds only what this bridge actually calls. Everything
// is reached through `objc_msgSend` with an explicit prototype cast, because
// these classes have no headers and the variadic `objc_msgSend` prototype
// passes floats and structs wrong on arm64.
static Class DescriptorClass = nil;
static Class ModelClass = nil;
static Class RequestClass = nil;
static Class IOSurfaceObjectClass = nil;
static Class ClientClass = nil;

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
            // Builds whose private ABI has been audited selector by selector.
            //
            // `26A5421a` is macOS 27.0 — a major kernel bump, xnu-12377 to
            // xnu-13432 — and every selector this bridge calls still resolves
            // there (`Tools/ANE/abi-check.m`, 2026-08-27). That is the only
            // thing being asserted. Numerics, the saturation cliff, whether
            // `kANEFAneInstanceHint` still selects a die, and whether the
            // deferred DART fault that panicked this machine three times on
            // 25F84 still reproduces are all **unproven on this build**.
            //
            // The trailing letter says GM seed. The shipping 27.0 build string
            // will differ and will have to be re-audited rather than assumed.
            NSArray<NSString *> *validated = @[@"25F84", @"26A5421a"];
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
        ClientClass          = NSClassFromString(@"_ANEClient");
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

#pragma mark - Power bracket

/// `+[_ANEClient sharedConnection]`. Not retained: it is the framework's own
/// singleton and outlives this process's interest in it, which matches how the
/// rest of this file holds runtime objects.
static id gPowerClient = nil;
static bool gPowerHeld = false;
static os_unfair_lock gPowerLock = OS_UNFAIR_LOCK_INIT;

bool h3_ane_power_is_held(void) {
    os_unfair_lock_lock(&gPowerLock);
    bool held = gPowerHeld;
    os_unfair_lock_unlock(&gPowerLock);
    return held;
}

static void H3ANEPowerReleaseAtExit(void) { h3_ane_power_release(); }

bool h3_ane_power_acquire(void) {
    if (!h3_ane_is_available()) return false;
    if (!ClassHasClassSelector(ClientClass, @selector(sharedConnection)) ||
        !ClassHasInstanceSelector(ClientClass, @selector(beginRealTimeTask))) return false;

    os_unfair_lock_lock(&gPowerLock);
    if (gPowerHeld) { os_unfair_lock_unlock(&gPowerLock); return true; }

    bool ok = false;
    @try {
        if (!gPowerClient) {
            gPowerClient = ((id(*)(Class, SEL))objc_msgSend)(
                ClientClass, @selector(sharedConnection));
        }
        if (gPowerClient) {
            ok = ((BOOL(*)(id, SEL))objc_msgSend)(
                gPowerClient, @selector(beginRealTimeTask));
        }
    } @catch (NSException *exception) {
        NSLog(@"[H3ANEBridge] private runtime exception while taking the power "
              @"bracket: %@", exception);
        ok = false;
    }

    if (ok) {
        gPowerHeld = true;
        static dispatch_once_t once;
        dispatch_once(&once, ^{ atexit(H3ANEPowerReleaseAtExit); });
    } else {
        // Measured 2026-08-27: refused on both `sharedConnection` and
        // `sharedPrivateConnection`, in under a millisecond, on retries. The
        // bracket is entitlement-gated and this process does not have it, so
        // the keep-alive below is the defence instead.
        static dispatch_once_t once;
        dispatch_once(&once, ^{
            NSLog(@"[H3ANEBridge] ANE power bracket refused (no entitlement); "
                  @"holding the engine awake with the keep-alive instead.");
        });
    }
    os_unfair_lock_unlock(&gPowerLock);
    return ok;
}

void h3_ane_power_release(void) {
    os_unfair_lock_lock(&gPowerLock);
    if (gPowerHeld && gPowerClient) {
        @try {
            ((BOOL(*)(id, SEL))objc_msgSend)(gPowerClient, @selector(endRealTimeTask));
        } @catch (NSException *exception) {
            NSLog(@"[H3ANEBridge] private runtime exception while releasing the "
                  @"power bracket: %@", exception);
        }
    }
    gPowerHeld = false;
    os_unfair_lock_unlock(&gPowerLock);
}

#pragma mark - Keep-alive

/// Why this exists, and why it is not optional.
///
/// The driver sleeps five seconds after its last work, and a program create
/// landing inside that sleep *transition* wedges the machine — the driver skips
/// the action block that completes the transition, and the blocking, gated
/// power-on it then issues waits forever inside the command gate every other
/// ANE client needs. Nothing faults, so there is no panic. See "Machine safety"
/// in `docs/ANE_STATUS.md`.
///
/// The supported way to prevent that is `-[_ANEClient beginRealTimeTask]`,
/// which declares the engine in use. It is entitlement-gated and refuses this
/// process (measured on both connections, sub-millisecond, on retries), so the
/// bracket is unavailable and this is what is left: submit trivial work to both
/// dies more often than the five-second timer, so the timer never fires while
/// this process intends to use the engine, so no transition is ever open for a
/// create to land in.
///
/// **The residual risk is the keep-alive's own first create**, plus any real
/// create that races it from another thread before the timer is running. The
/// transition is 8-15 ms wide and a fully-asleep die powers on correctly, so
/// the exposure is that window against a handful of creates at startup rather
/// than against every create in the session — one per new shape, dozens across
/// a benchmark. That is a large reduction in dice rolls, not a proof, and it is
/// the best available without the entitlement.
static dispatch_source_t gKeepAliveTimer = nil;
static bool gKeepAliveSettingUp = false;
static H3ANEProgram *gKeepAliveP0 = NULL, *gKeepAliveP1 = NULL;
static H3ANETensor *gKeepAliveX = NULL, *gKeepAliveW = NULL;
static H3ANETensor *gKeepAliveY0 = NULL, *gKeepAliveY1 = NULL;
static os_unfair_lock gKeepAliveLock = OS_UNFAIR_LOCK_INIT;

/// Small enough to be free, large enough to be a legal program: 64x64x64 fp16
/// is 8 KB a surface and runs in microseconds.
#define H3_ANE_KEEPALIVE_DIM 64
/// Under the driver's five-second idle timer with room for a late tick.
#define H3_ANE_KEEPALIVE_SECONDS 2.0

bool h3_ane_keepalive_is_running(void) {
    os_unfair_lock_lock(&gKeepAliveLock);
    bool running = gKeepAliveTimer != nil;
    os_unfair_lock_unlock(&gKeepAliveLock);
    return running;
}

static void H3ANEKeepAliveTearDown(void) {
    if (gKeepAliveTimer) { dispatch_source_cancel(gKeepAliveTimer); gKeepAliveTimer = nil; }
    if (gKeepAliveP0) { h3_ane_program_free(gKeepAliveP0); gKeepAliveP0 = NULL; }
    if (gKeepAliveP1) { h3_ane_program_free(gKeepAliveP1); gKeepAliveP1 = NULL; }
    if (gKeepAliveX)  { h3_ane_tensor_free(gKeepAliveX);  gKeepAliveX = NULL; }
    if (gKeepAliveW)  { h3_ane_tensor_free(gKeepAliveW);  gKeepAliveW = NULL; }
    if (gKeepAliveY0) { h3_ane_tensor_free(gKeepAliveY0); gKeepAliveY0 = NULL; }
    if (gKeepAliveY1) { h3_ane_tensor_free(gKeepAliveY1); gKeepAliveY1 = NULL; }
}

/// Starts the keep-alive if it is not already running. Returns false only if
/// the engine could not be set up at all, in which case the caller must not
/// create programs either — an engine that cannot hold itself awake is one this
/// bridge will not keep creating on.
static bool H3ANEKeepAliveEnsure(void) {
    // OFF BY DEFAULT AS OF 2026-08-27, because it is a suspect rather than a
    // fix. Seventy-two seconds after `Tools/ANE/ane-hold.m` started — with
    // nothing else on the machine touching the engine, no benchmark, no render,
    // only this ticking every two seconds — the kernel panicked:
    //
    //   sptm_t8110dart_clear_err: dart (dart-ane0:46): DART instance 1:
    //   Unrecoverable secondary error 0x80080008
    //
    // That is an IOMMU fault on ANE die 0's DART, not the gate wedge this was
    // written for. Submitting work every two seconds and letting the engine
    // power-cycle in between produces a rate of DART teardown and restore that
    // Apple's own stack never generates, and that is the one thing this code
    // does. Until that is understood it does not run unless asked for.
    static dispatch_once_t enabledOnce;
    static bool enabled = false;
    dispatch_once(&enabledOnce, ^{
        enabled = [NSProcessInfo.processInfo.environment[@"H3_ANE_KEEPALIVE"]
                      isEqualToString:@"1"];
    });
    if (!enabled) return true;

    os_unfair_lock_lock(&gKeepAliveLock);
    if (gKeepAliveTimer) { os_unfair_lock_unlock(&gKeepAliveLock); return true; }
    if (gKeepAliveSettingUp) { os_unfair_lock_unlock(&gKeepAliveLock); return true; }
    gKeepAliveSettingUp = true;
    os_unfair_lock_unlock(&gKeepAliveLock);

    // Best effort, and free when it works: if the bracket ever becomes
    // reachable the keep-alive becomes belt and braces rather than the belt.
    h3_ane_power_acquire();

    const int d = H3_ANE_KEEPALIVE_DIM;
    bool ok = false;
    // This is the one exposed create in the process. It goes first, so every
    // real create that follows happens with the idle timer held off.
    gKeepAliveP0 = h3_ane_program_create(d, d, d);
    gKeepAliveP1 = h3_ane_program_create(d, d, d);
    gKeepAliveX  = h3_ane_tensor_create(d, d);
    gKeepAliveW  = h3_ane_tensor_create(d, d);
    gKeepAliveY0 = h3_ane_tensor_create(d, d);
    gKeepAliveY1 = h3_ane_tensor_create(d, d);
    ok = gKeepAliveP0 && gKeepAliveP1 && gKeepAliveX && gKeepAliveW
        && gKeepAliveY0 && gKeepAliveY1;

    os_unfair_lock_lock(&gKeepAliveLock);
    gKeepAliveSettingUp = false;
    if (!ok) {
        NSLog(@"[H3ANEBridge] keep-alive setup failed; refusing to create programs. "
              @"See docs/ANE_STATUS.md, 'Machine safety'.");
        H3ANEKeepAliveTearDown();
        os_unfair_lock_unlock(&gKeepAliveLock);
        return false;
    }

    dispatch_queue_t queue = dispatch_queue_create("h3.ane.keepalive", DISPATCH_QUEUE_SERIAL);
    gKeepAliveTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue);
    dispatch_source_set_timer(gKeepAliveTimer,
                              dispatch_time(DISPATCH_TIME_NOW, 0),
                              (uint64_t)(H3_ANE_KEEPALIVE_SECONDS * NSEC_PER_SEC),
                              NSEC_PER_SEC / 4);
    dispatch_source_set_event_handler(gKeepAliveTimer, ^{
        // Both dies: an idle die sleeps on its own timer even while the other
        // is busy, and the idle second die is the one that got hit.
        if (!h3_ane_run_pair(gKeepAliveP0, gKeepAliveX, gKeepAliveW, gKeepAliveY0,
                             gKeepAliveP1, gKeepAliveX, gKeepAliveW, gKeepAliveY1)) {
            static dispatch_once_t once;
            dispatch_once(&once, ^{
                NSLog(@"[H3ANEBridge] keep-alive tick failed; the engine is no longer "
                      @"being held awake.");
            });
        }
    });
    dispatch_resume(gKeepAliveTimer);
    os_unfair_lock_unlock(&gKeepAliveLock);
    return true;
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
    // Before anything else reaches the driver: a create landing inside the
    // idle sleep transition is what takes the machine down, and the keep-alive
    // is what stops that transition from ever starting while this process is
    // using the engine. Its own setup creates through this function, hence the
    // reentrancy flag rather than a second create path.
    if (!H3ANEKeepAliveEnsure()) return NULL;

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
    // **A fresh `_ANEIOSurfaceObject` per evaluation, not the one cached on the
    // tensor.** `h3_ane_job_submit_pair` runs two evaluations concurrently
    // against two dies, each with its own DART, and every caller passes the
    // same input tensor to both — the island, the shard path, and the
    // keep-alive all do. Sharing one surface object across two in-flight
    // requests shares whatever per-evaluation mapping state the private class
    // keeps on it, and a request programmed with an IOVA that is not valid in
    // its own DART is exactly the fault this machine panicked with on
    // 2026-08-27:
    //
    //   sptm_t8110dart_clear_err: dart-ane0: DART instance 1:
    //   Unrecoverable secondary error 0x80080008
    //
    // The class is private and undocumented, so whether it is safe to share is
    // unknowable by reading; wrapping the same IOSurface again is cheap and
    // removes the question. **This is untested against the hardware.**
    // **Measured, and it is not the fault.** The theory was that sharing one
    // `_ANEIOSurfaceObject` between two concurrent requests on two dies — two
    // DARTs — races whatever per-evaluation mapping state the private class
    // holds, and programs one request with an IOVA invalid in its own DART.
    // Tested head to head at production submission rates on 2026-08-27:
    //
    //   shared object      671,438 pairs in 180 s, 0 failures
    //   per-request object 636,432 pairs in 180 s, 0 failures
    //
    // Both clean, so the shared object stays: it is the simpler code and the
    // change was not earned. `H3_ANE_PER_REQUEST_SURFACE_OBJECT=1` keeps the
    // alternative available for a future run that has reason to suspect it.
    //
    // What those runs did establish is where the fault is not. 1.3 million pair
    // submissions across six minutes did nothing, while 36 submissions spread
    // over 72 seconds panicked the machine. Failures track **power
    // transitions**, not submission volume.
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
static id<MTLComputePipelineState> gSwiGLUTransposePipelineState = nil;
static id<MTLComputePipelineState> gSwiGLUTransposeSplit4PipelineState = nil;
static id<MTLComputePipelineState> gMergeMLPIslandPipelineState = nil;
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
            "}\n"
            "kernel void h3_swiglu_transpose_fp16(\n"
            "    device const half* gate_fp16 [[buffer(0)]],\n"
            "    device const half* up_fp16   [[buffer(1)]],\n"
            "    device half* dst_fp16        [[buffer(2)]],\n"
            "    constant uint& s             [[buffer(3)]],\n"
            "    constant uint& ffn           [[buffer(4)]],\n"
            "    constant float& input_unscale [[buffer(5)]],\n"
            "    constant float& output_scale [[buffer(6)]],\n"
            "    uint2 gid                    [[thread_position_in_grid]]\n"
            ") {\n"
            "    uint row = gid.y; uint col = gid.x;\n"
            "    if (row >= s || col >= ffn) return;\n"
            "    float gate = ((float)gate_fp16[row * ffn + col]) * input_unscale;\n"
            "    float up = ((float)up_fp16[row * ffn + col]) * input_unscale;\n"
            "    float tail = 1.0f / (1.0f + exp(abs(gate)));\n"
            "    float sigmoid_gate = gate < 0.0f ? tail : 1.0f - tail;\n"
            "    dst_fp16[col * s + row] = (half)((gate * sigmoid_gate * up) * output_scale);\n"
            "}\n"
            "kernel void h3_swiglu_transpose_split4_fp16(\n"
            "    device const half* g0 [[buffer(0)]], device const half* g1 [[buffer(1)]],\n"
            "    device const half* g2 [[buffer(2)]], device const half* g3 [[buffer(3)]],\n"
            "    device const half* u0 [[buffer(4)]], device const half* u1 [[buffer(5)]],\n"
            "    device const half* u2 [[buffer(6)]], device const half* u3 [[buffer(7)]],\n"
            "    device half* dst [[buffer(8)]], constant uint& s [[buffer(9)]],\n"
            "    constant uint& ffn [[buffer(10)]], constant float& iu [[buffer(11)]],\n"
            "    constant float& os [[buffer(12)]], uint2 gid [[thread_position_in_grid]]) {\n"
            "    uint row=gid.y, col=gid.x; if(row>=s || col>=ffn) return;\n"
            "    uint i=row*ffn+col;\n"
            "    float g=((float)g0[i]+(float)g1[i]+(float)g2[i]+(float)g3[i])*iu;\n"
            "    float u=((float)u0[i]+(float)u1[i]+(float)u2[i]+(float)u3[i])*iu;\n"
            "    float tail=1.0f/(1.0f+exp(abs(g)));\n"
            "    float sigmoid_g=g<0.0f?tail:1.0f-tail;\n"
            "    dst[col*s+row]=(half)((g*sigmoid_g*u)*os);\n"
            "}\n"
            "kernel void h3_merge_mlp_island_partials(\n"
            "    device const ushort* gpu [[buffer(0)]],\n"
            "    device const half* a00 [[buffer(1)]], device const half* a01 [[buffer(2)]],\n"
            "    device const half* a02 [[buffer(3)]], device const half* a03 [[buffer(4)]],\n"
            "    device const half* a10 [[buffer(5)]], device const half* a11 [[buffer(6)]],\n"
            "    device const half* a12 [[buffer(7)]], device const half* a13 [[buffer(8)]],\n"
            "    device ushort* dst [[buffer(9)]], constant uint& s [[buffer(10)]],\n"
            "    constant uint& hidden [[buffer(11)]],\n"
            "    constant float& u0 [[buffer(12)]], constant float& u1 [[buffer(13)]],\n"
            "    uint2 gid [[thread_position_in_grid]]) {\n"
            "    uint row=gid.y, col=gid.x; if(row>=s || col>=hidden) return;\n"
            "    uint i=row*hidden+col;\n"
            "    float v=as_type<float>(((uint)gpu[i])<<16);\n"
            "    v += ((float)a00[i]+(float)a01[i]+(float)a02[i]+(float)a03[i])*u0;\n"
            "    v += ((float)a10[i]+(float)a11[i]+(float)a12[i]+(float)a13[i])*u1;\n"
            "    dst[i]=h3_bf16_rne(v);\n"
            "}\n";

        id<MTLLibrary> lib = [gMetalDevice newLibraryWithSource:metalSource options:nil error:&error];
        if (!lib) { NSLog(@"[H3ANEBridge] Metal lib init failed: %@", error); return; }

        id<MTLFunction> fnPack = [lib newFunctionWithName:@"h3_pack_bf16_to_fp16_transpose"];
        id<MTLFunction> fnAttn = [lib newFunctionWithName:@"h3_merge_attn_out"];
        id<MTLFunction> fnFC1  = [lib newFunctionWithName:@"h3_merge_fc1_swiglu"];
        id<MTLFunction> fnSwiGLUTranspose =
            [lib newFunctionWithName:@"h3_swiglu_transpose_fp16"];
        id<MTLFunction> fnSwiGLUTransposeSplit4 =
            [lib newFunctionWithName:@"h3_swiglu_transpose_split4_fp16"];
        id<MTLFunction> fnMergeMLPIsland =
            [lib newFunctionWithName:@"h3_merge_mlp_island_partials"];

        if (fnPack) gPackPipelineState = [gMetalDevice newComputePipelineStateWithFunction:fnPack error:nil];
        if (fnAttn) gMergeAttnPipelineState = [gMetalDevice newComputePipelineStateWithFunction:fnAttn error:nil];
        if (fnFC1)  gMergeFC1PipelineState  = [gMetalDevice newComputePipelineStateWithFunction:fnFC1 error:nil];
        if (fnSwiGLUTranspose) gSwiGLUTransposePipelineState =
            [gMetalDevice newComputePipelineStateWithFunction:fnSwiGLUTranspose error:nil];
        if (fnSwiGLUTransposeSplit4) gSwiGLUTransposeSplit4PipelineState =
            [gMetalDevice newComputePipelineStateWithFunction:fnSwiGLUTransposeSplit4 error:nil];
        if (fnMergeMLPIsland) gMergeMLPIslandPipelineState =
            [gMetalDevice newComputePipelineStateWithFunction:fnMergeMLPIsland error:nil];
    });

    return (gMetalDevice && gCommandQueue && gPackPipelineState && gMergeAttnPipelineState &&
            gMergeFC1PipelineState && gSwiGLUTransposePipelineState &&
            gSwiGLUTransposeSplit4PipelineState && gMergeMLPIslandPipelineState);
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

static bool ValidateSwiGLUTensors(H3ANETensor *gateTensor, H3ANETensor *upTensor,
                                  H3ANETensor *dstTensor, int s, int ffn) {
    if (!gateTensor || !upTensor || !dstTensor || s <= 0 || ffn <= 0 ||
        gateTensor->rows != s || gateTensor->width != ffn ||
        upTensor->rows != s || upTensor->width != ffn ||
        dstTensor->rows != ffn || dstTensor->width != s ||
        !h3_ane_tensor_is_dense(gateTensor) || !h3_ane_tensor_is_dense(upTensor) ||
        !h3_ane_tensor_is_dense(dstTensor)) return false;
    return true;
}

static bool EncodeSwiGLUTranspose(id<MTLCommandBuffer> cmd,
                                  H3ANETensor *gateTensor, H3ANETensor *upTensor,
                                  H3ANETensor *dstTensor, int s, int ffn,
                                  float inputUnscale, float outputScale) {
    if (gateTensor->rows != s || gateTensor->width != ffn ||
        upTensor->rows != s || upTensor->width != ffn ||
        dstTensor->rows != ffn || dstTensor->width != s ||
        !h3_ane_tensor_is_dense(gateTensor) || !h3_ane_tensor_is_dense(upTensor) ||
        !h3_ane_tensor_is_dense(dstTensor)) return false;
    void *gatePtr = h3_ane_tensor_ptr(gateTensor);
    void *upPtr = h3_ane_tensor_ptr(upTensor);
    void *dstPtr = h3_ane_tensor_ptr(dstTensor);
    if (!gatePtr || !upPtr || !dstPtr) return false;

    size_t sourceBytes = gateTensor->rows * gateTensor->rowBytes;
    size_t dstBytes = dstTensor->rows * dstTensor->rowBytes;
    id<MTLBuffer> gateBuf = [gMetalDevice newBufferWithBytesNoCopy:gatePtr
        length:sourceBytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> upBuf = [gMetalDevice newBufferWithBytesNoCopy:upPtr
        length:sourceBytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> dstBuf = [gMetalDevice newBufferWithBytesNoCopy:dstPtr
        length:dstBytes options:MTLResourceStorageModeShared deallocator:nil];
    if (!gateBuf || !upBuf || !dstBuf) return false;

    uint uS = (uint)s, uFFN = (uint)ffn;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gSwiGLUTransposePipelineState];
    [enc setBuffer:gateBuf offset:0 atIndex:0];
    [enc setBuffer:upBuf offset:0 atIndex:1];
    [enc setBuffer:dstBuf offset:0 atIndex:2];
    [enc setBytes:&uS length:sizeof(uint) atIndex:3];
    [enc setBytes:&uFFN length:sizeof(uint) atIndex:4];
    [enc setBytes:&inputUnscale length:sizeof(float) atIndex:5];
    [enc setBytes:&outputScale length:sizeof(float) atIndex:6];
    MTLSize group = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((ffn + 15) / 16 * 16, (s + 15) / 16 * 16, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    return true;
}

bool h3_ane_swiglu_transpose_fp16(H3ANETensor *gateTensor, H3ANETensor *upTensor,
                                  H3ANETensor *dstTensor, int s, int ffn,
                                  float inputUnscale, float outputScale,
                                  void *commandQueue) {
    if (!ValidateSwiGLUTensors(gateTensor, upTensor, dstTensor, s, ffn) ||
        !isfinite(inputUnscale) || !isfinite(outputScale) || !InitMetalKernels()) return false;
    id<MTLCommandQueue> queue = commandQueue ?
        (__bridge id<MTLCommandQueue>)commandQueue : gCommandQueue;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!EncodeSwiGLUTranspose(cmd, gateTensor, upTensor, dstTensor, s, ffn,
                              inputUnscale, outputScale)) return false;
    [cmd commit]; [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}

bool h3_ane_swiglu_transpose_pair_fp16(
    H3ANETensor *gate0, H3ANETensor *up0, H3ANETensor *dst0, float outputScale0,
    H3ANETensor *gate1, H3ANETensor *up1, H3ANETensor *dst1, float outputScale1,
    int s, int ffn, float inputUnscale, void *commandQueue) {
    if (!ValidateSwiGLUTensors(gate0, up0, dst0, s, ffn) ||
        !ValidateSwiGLUTensors(gate1, up1, dst1, s, ffn) ||
        !isfinite(inputUnscale) || !isfinite(outputScale0) ||
        !isfinite(outputScale1) || !InitMetalKernels()) return false;
    id<MTLCommandQueue> queue = commandQueue ?
        (__bridge id<MTLCommandQueue>)commandQueue : gCommandQueue;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!EncodeSwiGLUTranspose(cmd, gate0, up0, dst0, s, ffn,
                              inputUnscale, outputScale0) ||
        !EncodeSwiGLUTranspose(cmd, gate1, up1, dst1, s, ffn,
                              inputUnscale, outputScale1)) return false;
    [cmd commit]; [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}

static bool EncodeSwiGLUTransposeSplit4(id<MTLCommandBuffer> cmd,
                                        H3ANETensor *const *gate,
                                        H3ANETensor *const *up,
                                        H3ANETensor *dst, int s, int ffn,
                                        float inputUnscale, float outputScale) {
    if (!gate || !up || !dst || dst->rows != ffn || dst->width != s ||
        !h3_ane_tensor_is_dense(dst)) return false;
    id<MTLBuffer> buffers[9];
    for (int i = 0; i < 4; ++i) {
        if (!gate[i] || !up[i] || gate[i]->rows != s || gate[i]->width != ffn ||
            up[i]->rows != s || up[i]->width != ffn ||
            !h3_ane_tensor_is_dense(gate[i]) || !h3_ane_tensor_is_dense(up[i])) return false;
        size_t gateBytes = gate[i]->rows * gate[i]->rowBytes;
        size_t upBytes = up[i]->rows * up[i]->rowBytes;
        buffers[i] = [gMetalDevice newBufferWithBytesNoCopy:h3_ane_tensor_ptr(gate[i])
            length:gateBytes options:MTLResourceStorageModeShared deallocator:nil];
        buffers[4 + i] = [gMetalDevice newBufferWithBytesNoCopy:h3_ane_tensor_ptr(up[i])
            length:upBytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!buffers[i] || !buffers[4 + i]) return false;
    }
    buffers[8] = [gMetalDevice newBufferWithBytesNoCopy:h3_ane_tensor_ptr(dst)
        length:dst->rows * dst->rowBytes options:MTLResourceStorageModeShared deallocator:nil];
    if (!buffers[8]) return false;
    uint uS = (uint)s, uFFN = (uint)ffn;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gSwiGLUTransposeSplit4PipelineState];
    for (int i = 0; i < 9; ++i) [enc setBuffer:buffers[i] offset:0 atIndex:i];
    [enc setBytes:&uS length:sizeof(uint) atIndex:9];
    [enc setBytes:&uFFN length:sizeof(uint) atIndex:10];
    [enc setBytes:&inputUnscale length:sizeof(float) atIndex:11];
    [enc setBytes:&outputScale length:sizeof(float) atIndex:12];
    MTLSize group = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((ffn + 15) / 16 * 16, (s + 15) / 16 * 16, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    return true;
}

bool h3_ane_swiglu_transpose_pair_split4_fp16(
    H3ANETensor *const *gate0, H3ANETensor *const *up0,
    H3ANETensor *dst0, float outputScale0,
    H3ANETensor *const *gate1, H3ANETensor *const *up1,
    H3ANETensor *dst1, float outputScale1,
    int s, int ffn, float inputUnscale, void *commandQueue) {
    if (s <= 0 || ffn <= 0 || !isfinite(inputUnscale) ||
        !isfinite(outputScale0) || !isfinite(outputScale1) || !InitMetalKernels()) return false;
    id<MTLCommandQueue> queue = commandQueue ?
        (__bridge id<MTLCommandQueue>)commandQueue : gCommandQueue;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!EncodeSwiGLUTransposeSplit4(cmd, gate0, up0, dst0, s, ffn,
                                    inputUnscale, outputScale0) ||
        !EncodeSwiGLUTransposeSplit4(cmd, gate1, up1, dst1, s, ffn,
                                    inputUnscale, outputScale1)) return false;
    [cmd commit]; [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}

bool h3_ane_merge_mlp_island_partials(
    const void *gpuPartialBF16, H3ANETensor *const *ane0Partials, float ane0Unscale,
    H3ANETensor *const *ane1Partials, float ane1Unscale,
    void *dstBF16, int s, int hidden, void *commandQueue) {
    if (!gpuPartialBF16 || !ane0Partials || !ane1Partials || !dstBF16 ||
        s <= 0 || hidden <= 0 || !isfinite(ane0Unscale) || !isfinite(ane1Unscale) ||
        !InitMetalKernels()) return false;
    for (int i = 0; i < 4; ++i) {
        if (!ane0Partials[i] || !ane1Partials[i] ||
            ane0Partials[i]->rows < s || ane0Partials[i]->width != hidden ||
            ane1Partials[i]->rows < s || ane1Partials[i]->width != hidden ||
            !h3_ane_tensor_is_dense(ane0Partials[i]) ||
            !h3_ane_tensor_is_dense(ane1Partials[i])) return false;
    }
    size_t bytes = (size_t)s * hidden * 2;
    id<MTLBuffer> gpu = [gMetalDevice newBufferWithBytesNoCopy:(void *)gpuPartialBF16
        length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> dst = [gMetalDevice newBufferWithBytesNoCopy:dstBF16
        length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    if (!gpu || !dst) return false;
    id<MTLCommandQueue> queue = commandQueue ?
        (__bridge id<MTLCommandQueue>)commandQueue : gCommandQueue;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gMergeMLPIslandPipelineState];
    [enc setBuffer:gpu offset:0 atIndex:0];
    for (int i = 0; i < 4; ++i) {
        id<MTLBuffer> b0 = [gMetalDevice newBufferWithBytesNoCopy:h3_ane_tensor_ptr(ane0Partials[i])
            length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> b1 = [gMetalDevice newBufferWithBytesNoCopy:h3_ane_tensor_ptr(ane1Partials[i])
            length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!b0 || !b1) return false;
        [enc setBuffer:b0 offset:0 atIndex:(NSUInteger)(1 + i)];
        [enc setBuffer:b1 offset:0 atIndex:(NSUInteger)(5 + i)];
    }
    [enc setBuffer:dst offset:0 atIndex:9];
    uint uS=(uint)s, uHidden=(uint)hidden;
    [enc setBytes:&uS length:sizeof(uint) atIndex:10];
    [enc setBytes:&uHidden length:sizeof(uint) atIndex:11];
    [enc setBytes:&ane0Unscale length:sizeof(float) atIndex:12];
    [enc setBytes:&ane1Unscale length:sizeof(float) atIndex:13];
    MTLSize group=MTLSizeMake(16,16,1);
    MTLSize grid=MTLSizeMake((hidden+15)/16*16,(s+15)/16*16,1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group]; [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}
