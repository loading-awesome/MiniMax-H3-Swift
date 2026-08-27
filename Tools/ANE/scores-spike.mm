// scores-spike.mm — is the ANE's *matmul* correct at attention's N, or is the
// softmax to blame?
//
// The fused attention graph is 97% wrong at S=15,744 and 2.2% wrong at S=512.
// The diagnosis — a softmax normalised per tile over a score plane too large to
// hold — is inference. This discriminates it, and it has to run before any
// effort goes into rebuilding softmax from explicit reductions: if the matmul
// is already wrong at this shape, an explicit softmax is wasted work.
//
// It emits only `matmul(k, transpose(q))` — no softmax, no AV — over a *slice*
// of queries, so the output is [S, T] rather than [S, S]:
//
//   q [1,H,T,D]  ->  qt [1,H,D,T]  ->  scores = k @ qt  ->  [1,H,S,T]
//
// T != S is deliberate. The square case leaves a transposition ambiguity the
// attention spike had to work around; a non-square output cannot be silently
// transposed, so what comes back is unambiguous.
//
// N = 15,744 is far outside anything the bridge routes — the largest production
// N is 10,240 whole and 2,048 a die — so this shape is genuinely untested.
//
//   xcrun clang++ -std=c++17 -fobjc-arc -fblocks -framework Foundation \
//     -framework IOSurface Tools/ANE/scores-spike.mm -o /tmp/h3-scores-spike
//   H3_SCORES_S=15744 H3_SCORES_T=512 H3_SCORES_DUMP=/tmp/sc- /tmp/h3-scores-spike
//
// Then: python3 Tools/ANE/scores_reference.py /tmp/sc- --keys 15744 --queries 512

#define DIFF_K 64
#define DIFF_N 64
#define DIFF_S 64
#define DIFF_ITERATIONS 1
#define DIFF_DYNAMIC_ONLY 1
#define main h3_embedded_differential_main
#include "differential.m"
#undef main

#include <vector>

static int EnvInt(const char *name, int fallback) {
    const char *raw = getenv(name);
    if (!raw || !*raw) return fallback;
    long v = strtol(raw, NULL, 10);
    return v > 0 ? (int)v : fallback;
}

static void FillHalf(_Float16 *p, size_t count, uint32_t seed, float scale) {
    uint32_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        float u = (float)((state >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
        p[i] = (_Float16)((u * 2.0f - 1.0f) * scale);
    }
}

static NSString *ScoresMIL(int heads, int keys, int queries, int dimension) {
    NSMutableString *m = [NSMutableString stringWithString:MILHeader()];
    [m appendFormat:
        @"    func main<ios18>(tensor<fp16,[1,%d,%d,%d]> q, "
         "tensor<fp16,[1,%d,%d,%d]> k) {\n",
        heads, queries, dimension, heads, keys, dimension];
    [m appendString:
        @"        tensor<int32,[4]> pk=const()[name=string(\"pk\"),"
         "val=tensor<int32,[4]>([0,1,3,2])];\n"];
    [m appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> qt=transpose(perm=pk,x=q)"
         "[name=string(\"qt\")];\n", heads, dimension, queries];
    [m appendString:
        @"        bool bf=const()[name=string(\"bf\"),val=bool(false)];\n"];
    [m appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> scores=matmul("
         "transpose_x=bf,transpose_y=bf,x=k,y=qt)[name=string(\"scores\")];\n",
        heads, keys, queries];
    [m appendFormat:@"    } -> (scores);\n}\n"];
    return m;
}

static bool Dump(IOSurfaceRef s, size_t bytes, NSString *path) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    NSData *d = [NSData dataWithBytes:IOSurfaceGetBaseAddress(s) length:bytes];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return [d writeToFile:path atomically:YES];
}

int main(void) {
    @autoreleasepool {
        int heads = EnvInt("H3_SCORES_H", 1);
        int keys = EnvInt("H3_SCORES_S", 15744);
        int queries = EnvInt("H3_SCORES_T", 512);
        int dimension = EnvInt("H3_SCORES_D", 128);

        if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                    "AppleNeuralEngine", RTLD_NOW)) {
            fprintf(stderr, "Unable to load AppleNeuralEngine.framework\n"); return 2;
        }
        DescriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        ModelClass = NSClassFromString(@"_ANEInMemoryModel");
        RequestClass = NSClassFromString(@"_ANERequest");
        SurfaceObjectClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!DescriptorClass || !ModelClass || !RequestClass || !SurfaceObjectClass) {
            fprintf(stderr, "Required private runtime classes unavailable\n"); return 2;
        }
        mach_timebase_info(&Timebase);
        ExecutionOptions = @{};

        size_t qElements = (size_t)heads * queries * dimension;
        size_t kElements = (size_t)heads * keys * dimension;
        size_t oElements = (size_t)heads * keys * queries;
        printf("ANE scores-only spike H=%d keys=%d queries=%d D=%d\n",
               heads, keys, queries, dimension);
        printf("output=%.3f MiB (the score plane, exposed on purpose) pad=%dx\n",
               oElements * 2 / 1048576.0, EnvInt("H3_SCORES_PAD", 1));

        uint64_t t0 = mach_absolute_time();
        BuiltModel model = Build(ScoresMIL(heads, keys, queries, dimension), @{});
        double compile = Milliseconds(mach_absolute_time() - t0);
        if (!model.loaded) {
            printf("compile_load=REFUSED ms=%.1f error=%s\n",
                   compile, (model.error ?: @"unknown").UTF8String);
            Destroy(model); return 4;
        }
        printf("compile_load=ACCEPTED ms=%.1f\n", compile);

        IOSurfaceRef q = NewSurface(qElements * 2);
        IOSurfaceRef k = NewSurface(kElements * 2);
        // `H3_SCORES_PAD=n` oversizes the output. 0x1D decodes on macOS 27 as
        // "IOSurface smaller than the model expects", so if that is what is
        // happening, more room is all this needs — and the multiplier that
        // first succeeds bounds what the model actually wants.
        int pad = EnvInt("H3_SCORES_PAD", 1);
        IOSurfaceRef out = NewSurface(oElements * 2 * (size_t)pad);
        if (!q || !k || !out) { fprintf(stderr, "IOSurface allocation failed\n"); return 2; }

        std::vector<_Float16> host(kElements);
        FillHalf(host.data(), qElements, 0x2545f491U, 1.0f / sqrtf((float)dimension));
        WriteSurface(q, host.data(), qElements * 2);
        FillHalf(host.data(), kElements, 0x14402a29U, 1.0f);
        WriteSurface(k, host.data(), kElements * 2);

        NSString *error = nil;
        if (!Evaluate(model.model, @[(__bridge id)q, (__bridge id)k], out, NULL, @0, &error)) {
            printf("evaluate=REFUSED error=%s\n", error.UTF8String);
            Destroy(model); return 5;
        }
        IOSurfaceLock(out, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *o = (const _Float16 *)IOSurfaceGetBaseAddress(out);
        float lo = INFINITY, hi = -INFINITY; bool finite = true;
        for (size_t i = 0; i < oElements; ++i) {
            float v = (float)o[i];
            if (!isfinite(v)) { finite = false; break; }
            lo = fminf(lo, v); hi = fmaxf(hi, v);
        }
        IOSurfaceUnlock(out, kIOSurfaceLockReadOnly, NULL);
        printf("evaluate=%s output_range=[%.6g,%.6g]\n", finite ? "FINITE" : "NONFINITE", lo, hi);

        if (const char *prefix = getenv("H3_SCORES_DUMP")) {
            NSString *b = [NSString stringWithUTF8String:prefix];
            bool ok = Dump(q, qElements * 2, [b stringByAppendingString:@"q.bin"])
                   && Dump(k, kElements * 2, [b stringByAppendingString:@"k.bin"])
                   && Dump(out, oElements * 2, [b stringByAppendingString:@"scores.bin"]);
            printf("dump=%s\n", ok ? "ok" : "FAILED");
        }
        Destroy(model);
        return 0;
    }
}
