// attention-spike.mm — Can one fixed-shape SDPA head group stay on ANE?
//
// This is deliberately a standalone research probe. It uses the same private
// in-memory MIL compiler and IOSurface request path as the shipping linear
// bridge, but it does not add an attention API to that bridge or touch the
// render path. The first question is whether the compiler accepts the complete
// QK^T -> softmax -> AV graph. Only an accepted, fused graph earns integration.
//
// Build from the repository root:
//
//   xcrun clang++ -std=c++17 -fobjc-arc -fblocks \
//     -framework Foundation -framework IOSurface \
//     Tools/ANE/attention-spike.mm -o /tmp/h3-ane-attention-spike
//
// Runnable on audited builds only — macOS 27.0 and later. On 26.5.2 (25F84)
// and earlier this spike hard-locked the machine: a program create landing in
// the driver's idle power transition was admitted rather than refused, and the
// fault surfaced minutes later. 27.0 refuses such a request instead. See
// "Machine safety" in docs/ANE_STATUS.md, and confirm the build with
// Tools/ANE/pair-stress.m before running anything here.
//
// The first question is still whether the compiler accepts the complete
// QK^T -> softmax -> AV graph. Only an accepted, fused graph earns integration:
// a three-stage island with Metal seams between the pieces has already been
// measured by proxy and lost.
//
//   H3_ATTN_S=15744 H3_ATTN_H=1 /tmp/h3-ane-attention-spike
//   H3_ATTN_MODE=dual-isolated /tmp/h3-ane-attention-spike

#define DIFF_K 64
#define DIFF_N 64
#define DIFF_S 64
#define DIFF_ITERATIONS 1
#define DIFF_DYNAMIC_ONLY 1
#define main h3_embedded_differential_main
#include "differential.m"
#undef main

#import <dispatch/dispatch.h>
#include <algorithm>
#include <vector>

static int EnvInt(const char *name, int fallback, int minimum, int maximum) {
    const char *raw = getenv(name);
    if (!raw || !*raw) return fallback;
    long value = strtol(raw, NULL, 10);
    return value >= minimum && value <= maximum ? (int)value : fallback;
}

static NSString *EnvString(const char *name, NSString *fallback) {
    const char *raw = getenv(name);
    return raw && *raw ? [NSString stringWithUTF8String:raw] : fallback;
}

static NSString *AttentionMIL(int heads, int sequence, int dimension) {
    NSMutableString *m = [NSMutableString stringWithString:MILHeader()];
    [m appendFormat:
        @"    func main<ios18>(tensor<fp16,[1,%d,%d,%d]> q, "
         "tensor<fp16,[1,%d,%d,%d]> k, tensor<fp16,[1,%d,%d,%d]> v) {\n",
        heads, sequence, dimension, heads, sequence, dimension,
        heads, sequence, dimension];
    [m appendString:
        @"        tensor<int32,[4]> pk=const()[name=string(\"pk\"),"
         "val=tensor<int32,[4]>([0,1,3,2])];\n"];
    // On this private lowering, q @ k^T arrives as softmax(k @ q^T): the
    // square score plane is transposed even though the MIL types say it is not.
    // Feed k @ q^T so that physical transposition produces the SDPA score plane
    // QK^T. The scalar oracle below checks both orientations and keeps this
    // workaround tied to evidence rather than to the compiler's declared type.
    [m appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> qt=transpose(perm=pk,x=q)"
         "[name=string(\"qt\")];\n",
        heads, dimension, sequence];
    [m appendString:
        @"        bool bf=const()[name=string(\"bf\"),val=bool(false)];\n"];
    [m appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> scores=matmul("
         "transpose_x=bf,transpose_y=bf,x=k,y=qt)[name=string(\"scores\")];\n",
        heads, sequence, sequence];
    [m appendString:
        @"        int32 axis=const()[name=string(\"axis\"),val=int32(3)];\n"];
    [m appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> probabilities=softmax("
         "axis=axis,x=scores)[name=string(\"probabilities\")];\n",
        heads, sequence, sequence];
    [m appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> y=matmul("
         "transpose_x=bf,transpose_y=bf,x=probabilities,y=v)"
         "[name=string(\"y\")];\n"
         "    } -> (y);\n}\n",
        heads, sequence, dimension];
    return m;
}

static uint32_t Mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; x ^= x >> 16; return x;
}

static void Fill(_Float16 *p, size_t count, uint32_t seed, float scale) {
    for (size_t i = 0; i < count; ++i) {
        float unit = ((float)(Mix((uint32_t)i + seed) & 0xffffU) - 32768.0f) / 32768.0f;
        p[i] = (_Float16)(unit * scale);
    }
}

static double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

static double Run(id model, NSArray *inputs, IOSurfaceRef output,
                  NSDictionary *options, NSString **errorText) {
    uint64_t begin = mach_absolute_time();
    BOOL ok = EvaluateWithOptions(model, inputs, output, NULL, @0, options, errorText);
    return ok ? Milliseconds(mach_absolute_time() - begin) : NAN;
}

static BOOL FiniteAndNonconstant(IOSurfaceRef output, size_t count,
                                 float *minimum, float *maximum) {
    IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *p = (const _Float16 *)IOSurfaceGetBaseAddress(output);
    float lo = INFINITY, hi = -INFINITY;
    BOOL finite = YES;
    for (size_t i = 0; i < count; ++i) {
        float value = (float)p[i];
        finite &= isfinite(value); lo = fminf(lo, value); hi = fmaxf(hi, value);
    }
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    *minimum = lo; *maximum = hi;
    return finite && hi > lo;
}

struct ReferenceErrors { double keyAxis, queryAxis, swappedQK; };

/// Writes a surface's fp16 contents so the reference can be computed outside.
///
/// `ReferenceError` above is a scalar O(H*S^2*D) loop and is capped at S=512 for
/// that reason, which left precision at the production sequence unmeasured —
/// the one number that decides whether the fused graph is usable. A chunked
/// fp32 reference over the same tensors costs about a second in numpy, so the
/// cap is a tooling limit rather than a real one.
static bool DumpSurface(IOSurfaceRef surface, size_t bytes, NSString *path) {
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    NSData *data = [NSData dataWithBytes:IOSurfaceGetBaseAddress(surface) length:bytes];
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    NSError *error = nil;
    bool ok = [data writeToFile:path options:NSDataWritingAtomic error:&error];
    if (!ok) fprintf(stderr, "dump failed for %s: %s\n",
                     path.UTF8String, error.description.UTF8String);
    return ok;
}

static ReferenceErrors ReferenceError(IOSurfaceRef qSurface, IOSurfaceRef kSurface,
                                      IOSurfaceRef vSurface, IOSurfaceRef output,
                                      int heads, int sequence, int dimension) {
    // Deliberately bounded. This is O(H*S^2*D), and the production probe is a
    // placement/rate experiment rather than an invitation to spend hours in a
    // scalar reference implementation.
    if (heads > 2 || sequence > 512) return {NAN, NAN, NAN};
    IOSurfaceLock(qSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(kSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(vSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *q = (const _Float16 *)IOSurfaceGetBaseAddress(qSurface);
    const _Float16 *k = (const _Float16 *)IOSurfaceGetBaseAddress(kSurface);
    const _Float16 *v = (const _Float16 *)IOSurfaceGetBaseAddress(vSurface);
    const _Float16 *got = (const _Float16 *)IOSurfaceGetBaseAddress(output);
    std::vector<float> scores((size_t)sequence * sequence);
    std::vector<float> rowDenominator(sequence), columnDenominator(sequence);
    std::vector<float> rowMaximum(sequence, -INFINITY), columnMaximum(sequence, -INFINITY);
    std::vector<float> expectedKey(dimension), expectedQuery(dimension), expectedSwapped(dimension);
    double keyError2 = 0, keyReference2 = 0, queryError2 = 0, queryReference2 = 0;
    double swappedError2 = 0, swappedReference2 = 0;
    for (int h = 0; h < heads; ++h) {
        std::fill(rowMaximum.begin(), rowMaximum.end(), -INFINITY);
        std::fill(columnMaximum.begin(), columnMaximum.end(), -INFINITY);
        for (int row = 0; row < sequence; ++row) for (int key = 0; key < sequence; ++key) {
            float sum = 0;
            size_t qb = ((size_t)h * sequence + row) * dimension;
            size_t kb = ((size_t)h * sequence + key) * dimension;
            for (int d = 0; d < dimension; ++d) sum += (float)q[qb+d] * (float)k[kb+d];
            scores[(size_t)row * sequence + key] = sum;
            rowMaximum[row] = fmaxf(rowMaximum[row], sum);
            columnMaximum[key] = fmaxf(columnMaximum[key], sum);
        }
        std::fill(rowDenominator.begin(), rowDenominator.end(), 0.0f);
        std::fill(columnDenominator.begin(), columnDenominator.end(), 0.0f);
        for (int row = 0; row < sequence; ++row) for (int key = 0; key < sequence; ++key) {
            float score = scores[(size_t)row * sequence + key];
            rowDenominator[row] += expf(score - rowMaximum[row]);
            columnDenominator[key] += expf(score - columnMaximum[key]);
        }
        for (int row = 0; row < sequence; ++row) {
            std::fill(expectedKey.begin(), expectedKey.end(), 0.0f);
            std::fill(expectedQuery.begin(), expectedQuery.end(), 0.0f);
            std::fill(expectedSwapped.begin(), expectedSwapped.end(), 0.0f);
            for (int key = 0; key < sequence; ++key) {
                float score = scores[(size_t)row * sequence + key];
                float pk = expf(score - rowMaximum[row]) / rowDenominator[row];
                float pq = expf(score - columnMaximum[key]) / columnDenominator[key];
                float swappedScore = scores[(size_t)key * sequence + row];
                float ps = expf(swappedScore - columnMaximum[row]) / columnDenominator[row];
                size_t vb = ((size_t)h * sequence + key) * dimension;
                for (int d = 0; d < dimension; ++d) {
                    expectedKey[d] += pk * (float)v[vb+d];
                    expectedQuery[d] += pq * (float)v[vb+d];
                    expectedSwapped[d] += ps * (float)v[vb+d];
                }
            }
            size_t ob = ((size_t)h * sequence + row) * dimension;
            for (int d = 0; d < dimension; ++d) {
                double ek = expectedKey[d], eq = expectedQuery[d], es = expectedSwapped[d];
                double dk = (double)got[ob+d] - ek, dq = (double)got[ob+d] - eq;
                double ds = (double)got[ob+d] - es;
                keyError2 += dk * dk; keyReference2 += ek * ek;
                queryError2 += dq * dq; queryReference2 += eq * eq;
                swappedError2 += ds * ds; swappedReference2 += es * es;
            }
        }
    }
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(vSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(kSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(qSurface, kIOSurfaceLockReadOnly, NULL);
    return {sqrt(keyError2 / fmax(keyReference2, 1e-300)),
            sqrt(queryError2 / fmax(queryReference2, 1e-300)),
            sqrt(swappedError2 / fmax(swappedReference2, 1e-300))};
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL); mach_timebase_info(&Timebase);
        int heads = EnvInt("H3_ATTN_H", 1, 1, 56);
        int sequence = EnvInt("H3_ATTN_S", 512, 16, 15744);
        int dimension = EnvInt("H3_ATTN_D", 128, 16, 256);
        int samples = EnvInt("H3_ATTN_SAMPLES", 7, 1, 31);
        NSString *mode = EnvString("H3_ATTN_MODE", @"serial");
        NSString *fixture = EnvString("H3_ATTN_FIXTURE", @"random");
        BOOL dual = [mode isEqualToString:@"dual-shared"] ||
                    [mode isEqualToString:@"dual-isolated"];
        BOOL isolatedInputs = [mode isEqualToString:@"dual-isolated"];
        if (![mode isEqualToString:@"serial"] && !dual) {
            fprintf(stderr, "H3_ATTN_MODE must be serial, dual-shared, or dual-isolated\n");
            return 2;
        }
        if (![fixture isEqualToString:@"random"] &&
            ![fixture isEqualToString:@"uniform"] &&
            ![fixture isEqualToString:@"constant-v"]) {
            fprintf(stderr, "H3_ATTN_FIXTURE must be random, uniform, or constant-v\n");
            return 2;
        }
        size_t elements = (size_t)heads * sequence * dimension;
        double exposedScoreGiB = (double)heads * sequence * sequence * 2.0 /
                                 (1024.0 * 1024.0 * 1024.0);

        if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                    "AppleNeuralEngine", RTLD_NOW)) {
            fprintf(stderr, "Unable to load AppleNeuralEngine.framework\n"); return 2;
        }
        DescriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        ModelClass = NSClassFromString(@"_ANEInMemoryModel");
        RequestClass = NSClassFromString(@"_ANERequest");
        SurfaceObjectClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!DescriptorClass || !ModelClass || !RequestClass || !SurfaceObjectClass) {
            fprintf(stderr, "Required private runtime classes are unavailable\n"); return 2;
        }

        NSDictionary *options0 = @{@"kANEFProcedureVariantHint": @1,
                                    @"kANEFAneInstanceHint": @1};
        NSDictionary *options1 = @{@"kANEFProcedureVariantHint": @1,
                                    @"kANEFAneInstanceHint": @2};
        NSString *mil = AttentionMIL(heads, sequence, dimension);
        ExecutionOptions = options0;

        printf("ANE fused-attention spike H=%d S=%d D=%d mode=%s fixture=%s\n",
               heads, sequence, dimension, mode.UTF8String, fixture.UTF8String);
        printf("explicit_io=%.3f MiB hypothetical_exposed_scores=%.3f GiB\n",
               4.0 * elements * 2.0 / 1048576.0, exposedScoreGiB);

        uint64_t compileStart = mach_absolute_time();
        BuiltModel model0 = Build(mil, @{});
        double compile0 = Milliseconds(mach_absolute_time() - compileStart);
        if (!model0.loaded) {
            printf("compile_load=REFUSED ms=%.1f error=%s\n",
                   compile0, (model0.error ?: @"unknown").UTF8String);
            Destroy(model0); return 4;
        }
        BuiltModel model1 = {0};
        double compile1 = NAN;
        if (dual) {
            ExecutionOptions = options1;
            compileStart = mach_absolute_time();
            model1 = Build(mil, @{});
            compile1 = Milliseconds(mach_absolute_time() - compileStart);
            if (!model1.loaded) {
                printf("second_compile_load=REFUSED ms=%.1f error=%s\n",
                       compile1, (model1.error ?: @"unknown").UTF8String);
                Destroy(model0); Destroy(model1); return 4;
            }
        }
        printf("compile_load=ACCEPTED first_ms=%.1f", compile0);
        if (dual) printf(" second_ms=%.1f", compile1);
        printf("\n");

        IOSurfaceRef q = NewSurface(elements * 2), k = NewSurface(elements * 2);
        IOSurfaceRef v = NewSurface(elements * 2), y0 = NewSurface(elements * 2);
        IOSurfaceRef y1 = dual ? NewSurface(elements * 2) : NULL;
        IOSurfaceRef q1 = isolatedInputs ? NewSurface(elements * 2) : q;
        IOSurfaceRef k1 = isolatedInputs ? NewSurface(elements * 2) : k;
        IOSurfaceRef v1 = isolatedInputs ? NewSurface(elements * 2) : v;
        if (!q || !k || !v || !y0 || (dual && !y1) || !q1 || !k1 || !v1) {
            fprintf(stderr, "IOSurface allocation failed\n"); return 2;
        }
        _Float16 *host = (_Float16 *)malloc(elements * 2);
        // Pre-scale Q by 1/sqrt(D), which is algebraically the production SDPA
        // scale and keeps the MIL graph to the three operations under test.
        // Unit-scale K makes the score distribution representative (std ~1)
        // instead of the nearly-uniform softmax produced by two 0.03 operands.
        if ([fixture isEqualToString:@"uniform"]) {
            memset(host, 0, elements * 2); WriteSurface(q, host, elements * 2);
            WriteSurface(k, host, elements * 2);
            Fill(host, elements, 0x6c078965U, 0.20f); WriteSurface(v, host, elements * 2);
        } else {
            Fill(host, elements, 0x2545f491U, 1.0f / sqrtf((float)dimension));
            WriteSurface(q, host, elements * 2);
            Fill(host, elements, 0x14402a29U, 1.0f); WriteSurface(k, host, elements * 2);
            if ([fixture isEqualToString:@"constant-v"]) {
                for (size_t i = 0; i < elements; ++i) host[i] = (_Float16)0.125f;
            } else {
                Fill(host, elements, 0x6c078965U, 0.20f);
            }
            WriteSurface(v, host, elements * 2);
        }
        if (isolatedInputs) {
            IOSurfaceLock(q, kIOSurfaceLockReadOnly, NULL);
            WriteSurface(q1, IOSurfaceGetBaseAddress(q), elements * 2);
            IOSurfaceUnlock(q, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceLock(k, kIOSurfaceLockReadOnly, NULL);
            WriteSurface(k1, IOSurfaceGetBaseAddress(k), elements * 2);
            IOSurfaceUnlock(k, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceLock(v, kIOSurfaceLockReadOnly, NULL);
            WriteSurface(v1, IOSurfaceGetBaseAddress(v), elements * 2);
            IOSurfaceUnlock(v, kIOSurfaceLockReadOnly, NULL);
        }
        free(host);
        NSArray *inputs = @[(__bridge id)q, (__bridge id)k, (__bridge id)v];
        NSArray *inputs1 = @[(__bridge id)q1, (__bridge id)k1, (__bridge id)v1];

        NSString *error = nil;
        double warm = Run(model0.model, inputs, y0, options0, &error);
        if (!isfinite(warm)) {
            printf("evaluate=REFUSED error=%s\n", (error ?: @"unknown").UTF8String);
            return 5;
        }
        float lo = 0, hi = 0;
        BOOL sane = FiniteAndNonconstant(y0, elements, &lo, &hi);
        printf("evaluate=%s warm_ms=%.3f output_range=[%.6g,%.6g]\n",
               sane ? "SANE" : "INVALID", warm, lo, hi);
        if (!sane) return 6;
        // `H3_ATTN_DUMP=/path/prefix` writes q/k/v/y for an external reference.
        if (const char *prefix = getenv("H3_ATTN_DUMP")) {
            size_t bytes = (size_t)heads * sequence * dimension * sizeof(_Float16);
            NSString *base = [NSString stringWithUTF8String:prefix];
            bool ok = DumpSurface(q,  bytes, [base stringByAppendingString:@"q.bin"])
                   && DumpSurface(k,  bytes, [base stringByAppendingString:@"k.bin"])
                   && DumpSurface(v,  bytes, [base stringByAppendingString:@"v.bin"])
                   && DumpSurface(y0, bytes, [base stringByAppendingString:@"y.bin"]);
            printf("dump=%s bytes_each=%zu heads=%d sequence=%d dimension=%d\n",
                   ok ? "ok" : "FAILED", bytes, heads, sequence, dimension);
        }
        ReferenceErrors reference = ReferenceError(q, k, v, y0, heads, sequence, dimension);
        if (isfinite(reference.keyAxis)) {
            printf("cpu_reference_rel_rms key_axis=%.6g query_axis=%.6g swapped_qk=%.6g\n",
                   reference.keyAxis, reference.queryAxis, reference.swappedQK);
        }

        std::vector<double> one, pairTimes;
        for (int i = 0; i < samples; ++i) one.push_back(Run(model0.model, inputs, y0, options0, NULL));
        if (dual) for (int i = 0; i < samples; ++i) {
            __block double a = NAN, b = NAN;
            dispatch_group_t group = dispatch_group_create();
            dispatch_semaphore_t start = dispatch_semaphore_create(0);
            dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                @autoreleasepool { dispatch_semaphore_wait(start, DISPATCH_TIME_FOREVER);
                    a = Run(model0.model, inputs, y0, options0, NULL); }
            });
            dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                @autoreleasepool { dispatch_semaphore_wait(start, DISPATCH_TIME_FOREVER);
                    b = Run(model1.model, inputs1, y1, options1, NULL); }
            });
            uint64_t begin = mach_absolute_time();
            dispatch_semaphore_signal(start); dispatch_semaphore_signal(start);
            dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
            pairTimes.push_back(Milliseconds(mach_absolute_time() - begin));
            if (!isfinite(a) || !isfinite(b)) { fprintf(stderr, "dual evaluation failed\n"); return 5; }
        }
        double isolated = Median(one);
        double flop = 4.0 * heads * sequence * sequence * dimension;
        printf("single_ms=%.3f effective_tflops=%.3f\n", isolated, flop / isolated / 1e9);
        if (dual) {
            double pair = Median(pairTimes);
            printf("dual_wall_ms=%.3f dual_speedup_vs_serial=%.3fx inputs=%s\n",
                   pair, (2.0 * isolated) / pair, isolatedInputs ? "isolated" : "shared");
            printf("verdict=%s\n", pair <= isolated * 1.15 ?
                   "two fused evaluations can occupy the two dies" :
                   "fused evaluations do not scale across the two dies");
        } else {
            printf("verdict=serial graph accepted; concurrency not exercised\n");
        }

        // evaluateWithQoS returns synchronously, but the first dual-attention
        // probe was followed by a hard lock after it immediately unloaded both
        // models. Keep teardown out of the completion edge while that boundary
        // is being isolated.
        [NSThread sleepForTimeInterval:2.0];
        CFRelease(q); CFRelease(k); CFRelease(v); CFRelease(y0); if (y1) CFRelease(y1);
        if (isolatedInputs) { CFRelease(q1); CFRelease(k1); CFRelease(v1); }
        Destroy(model0); if (dual) Destroy(model1);
    }
    return 0;
}
