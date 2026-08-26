// numerics.m — What arithmetic does this engine actually perform?
//
// Every ANE number in this tree so far is a statistic: a relative RMS against
// some fixture, which mixes a hardware property with that fixture's
// conditioning and cannot separate them. `differential.m` reported 1.35% at
// K=5376 on a fixture that cancels 3,900:1, and the percentage is mostly the
// fixture.
//
// These probes are not statistics. Each one has an exact integer-valued answer
// that differs between the candidate hardware models, so a single evaluation
// discriminates rather than estimates. They are the tests Bryngelson 2026
// (arXiv:2606.22283) chapter 3 uses to fix the datapath, reproduced here
// because that work measured M1/H13 and M5/H17 and names A15/M3 — this machine
// — "the one rail that remains unmeasured".
//
// Measured on this machine, 2026-08-26, M3 Ultra / h15g / macOS 25F84 — the
// generation Bryngelson leaves unmeasured. The pipeline these probes establish:
//
//   fp16 in -> fp16 multiply, PRODUCT ROUNDED TO FP16
//           -> wide (fp32-class) accumulator, exact
//           -> fp16 out, ties round up
//
//   | property                  | M1/H13 (paper) | M3/H15 (here)     |
//   | wide accumulator          | yes            | yes               |
//   | first-stage tile of four  | yes (5116)     | NO (5120 exact)   |
//   | per-product fp16 rounding | yes            | yes               |
//   | partial saturates at 2^15 | yes            | yes, RETURNS ZERO |
//   | denormals in the MAC      | flushed        | flushed           |
//   | tie rounding              | to even        | up (like M5)      |
//
// Three things follow for H3. The accumulator is exact, so a long contraction
// is not itself a hazard and fc2's K=14336 is not the worst case anyone feared.
// The drift is entirely per-product rounding, which makes relative error
// 2^-12 * cancellation / sqrt(K) — a property of the data's conditioning, not
// of K. And "the accumulator width is a fixed hardware property, not a
// per-program or per-chip setting" predicts that no lowering can move any of
// this, which is why the static and dynamic paths measure bit identical.
//
// The saturation row is the one that can lose a render. A dot product whose
// running partial reaches 2^15 comes back ZERO, not inf and not NaN, so it
// carries no signal that anything went wrong. Cancellation-heavy projections
// reach partials far above their own result, so the threshold that matters is
// max|partial|, not max|output|.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     Tools/ANE/numerics.m -o /tmp/h3-ane-numerics
//   /tmp/h3-ane-numerics

#define DIFF_K 4096
#define DIFF_N 256
#define DIFF_S 64
#define DIFF_ITERATIONS 1
#define main h3_embedded_differential_main
#include "differential.m"
#undef main

// One probe writes a column of the activation; the weight is all ones, so the
// engine reduces that column and the output is the sum. Every column is an
// independent experiment and they all run in one dispatch.
typedef struct {
    const char *name;
    double exact;         // the mathematically correct sum
    double naiveFP16;     // what an fp16 running sum returns
    double tiledWide;     // what tiles-of-four into a wide accumulator returns
    const char *reads;    // what the discrimination means
} Probe;

enum { MaxProbes = DIFF_S };
static Probe Probes[MaxProbes];
static int ProbeCount = 0;

static _Float16 *Column = NULL;   // [K] scratch for the probe being built
static _Float16 *Activation = NULL;

static void BeginProbe(void) { for (int k = 0; k < K; ++k) Column[k] = (_Float16)0.0f; }

static void EmitProbe(const char *name, double exact, double naive, double tiled,
                      const char *reads) {
    if (ProbeCount >= MaxProbes) return;
    for (int k = 0; k < K; ++k) Activation[k * S + ProbeCount] = Column[k];
    Probes[ProbeCount] = (Probe){ name, exact, naive, tiled, reads };
    ++ProbeCount;
}

// A naive fp16 running sum, for the "narrow accumulator" hypothesis.
static double NaiveFP16Sum(const _Float16 *v, int n) {
    _Float16 acc = (_Float16)0.0f;
    for (int i = 0; i < n; ++i) acc = (_Float16)((float)acc + (float)v[i]);
    return (double)(float)acc;
}

// Tiles of four summed in fp16, then accumulated exactly: the specified model.
static double TiledWideSum(const _Float16 *v, int n) {
    double wide = 0;
    for (int i = 0; i < n; i += 4) {
        _Float16 tile = (_Float16)0.0f;
        for (int j = i; j < i + 4 && j < n; ++j) tile = (_Float16)((float)tile + (float)v[j]);
        wide += (double)(float)tile;
    }
    return wide;
}

static double ExactSum(const _Float16 *v, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)(float)v[i];
    return s;
}

static void EmitFromColumn(const char *name, int used, const char *reads) {
    EmitProbe(name, ExactSum(Column, used), NaiveFP16Sum(Column, used),
              TiledWideSum(Column, used), reads);
}

static void BuildProbes(void) {
    // 1. Reduction of ones. A wide accumulator returns the count exactly; an
    //    fp16 running sum stalls once the partial passes 2048, because ulp
    //    there exceeds 1 and each further increment is swallowed.
    for (int count = 1024; count <= 4096; count *= 2) {
        BeginProbe();
        for (int k = 0; k < count; ++k) Column[k] = (_Float16)1.0f;
        char *name = strdup([NSString stringWithFormat:@"ones x%d", count].UTF8String);
        EmitFromColumn(name, count, "exact => wide; stalls near 2048 => narrow");
    }

    // 2. One large value followed by ones. The paper's worked case: exact 5120,
    //    naive fp16 4096, and the engine between the two. The deficit is the
    //    fp16 rounding of the tile holding the large value, so its size is a
    //    direct read on the first-stage tile.
    BeginProbe();
    Column[0] = (_Float16)4096.0f;
    for (int k = 1; k <= 1024; ++k) Column[k] = (_Float16)1.0f;
    EmitFromColumn("4096 + 1024 ones", 1025, "deficit from exact = first-stage tile rounding");

    // 3. The discriminating cancellation triple. [M, -M, 1] repeated sixteen
    //    times: a running sum that ever holds M has spacing ulp(M) there, so
    //    every 1 rounds away once ulp(M) > 2. If all sixteen survive, the
    //    reduction is not being held at fp16.
    static const float magnitudes[] = { 512, 2048, 4096, 16384, 32768, 40000, 60000 };
    for (size_t i = 0; i < sizeof magnitudes / sizeof *magnitudes; ++i) {
        BeginProbe();
        int k = 0;
        for (int repeat = 0; repeat < 16; ++repeat) {
            Column[k++] = (_Float16)magnitudes[i];
            Column[k++] = (_Float16)(-magnitudes[i]);
            Column[k++] = (_Float16)1.0f;
            Column[k++] = (_Float16)0.0f;      // pad the triple onto the lane lattice
        }
        char *name = strdup([NSString stringWithFormat:@"triple@%g x16", magnitudes[i]].UTF8String);
        EmitFromColumn(name, k, "survivors of 16 ones fix the accumulator width");
    }

    // 4. Output-port saturation. The multiply-accumulate output stage is
    //    reported to saturate at 2^15 = 32768, half the fp16 ceiling of 65504,
    //    and to be a property of the port rather than of the term count.
    static const float totals[] = { 16384, 32764, 32768, 40000, 65504 };
    for (size_t i = 0; i < sizeof totals / sizeof *totals; ++i) {
        BeginProbe();
        int lanes = 64;
        for (int k = 0; k < lanes; ++k) Column[k] = (_Float16)(totals[i] / lanes);
        char *name = strdup([NSString stringWithFormat:@"sum to %g", totals[i]].UTF8String);
        EmitFromColumn(name, lanes, "finite past 32768 => no port saturation");
    }

    // 5. Denormal handling inside the multiply-accumulate. M1 flushes, M5
    //    preserves; this generation is unrecorded either way.
    BeginProbe();
    for (int k = 0; k < 16; ++k) Column[k] = (_Float16)5.96e-8f;   // fp16 denormal
    EmitFromColumn("denormals x16", 16, "nonzero => preserved, zero => flushed in the MAC");

    // 6. Tie rounding. 2048 + 1 sits exactly halfway between representable
    //    fp16 neighbours at that magnitude. M1 rounds to even (2048), M5
    //    rounds up (2050) — a documented cross-generation divergence.
    BeginProbe();
    Column[0] = (_Float16)2048.0f;
    Column[1] = (_Float16)1.0f;
    EmitFromColumn("2048 + 1 tie", 2, "2048 => ties-to-even, 2050 => ties-up");
}

int main(void) { @autoreleasepool {
    setbuf(stdout, NULL);
    mach_timebase_info(&Timebase);
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                RTLD_NOW)) { fprintf(stderr, "Unable to load AppleNeuralEngine.framework\n"); return 2; }
    DescriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    ModelClass      = NSClassFromString(@"_ANEInMemoryModel");
    RequestClass    = NSClassFromString(@"_ANERequest");
    SurfaceObjectClass = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!DescriptorClass || !ModelClass || !RequestClass || !SurfaceObjectClass) {
        fprintf(stderr, "Required private classes unavailable\n"); return 2;
    }

    // Build() hands this straight to compileWithQoS:options: and
    // loadWithQoS:options:. Leaving it nil fails the compile with a bare
    // "Cannot load network" that names neither the option nor the shape.
    ExecutionOptions = @{};

    Column     = (_Float16 *)calloc(K, 2);
    Activation = (_Float16 *)calloc((size_t)K * S, 2);
    _Float16 *ones = (_Float16 *)calloc((size_t)K * N, 2);
    for (int i = 0; i < K * N; ++i) ones[i] = (_Float16)1.0f;
    BuildProbes();

    BuiltModel model = Build(SeparateMIL(), @{});
    if (!model.loaded) { fprintf(stderr, "program build failed: %s\n", model.error.UTF8String); return 2; }
    IOSurfaceRef xs = NewSurface(TensorBytes(K, S)), ws = NewSurface(TensorBytes(K, N)),
                 out = NewSurface(TensorBytes(N, S));
    WriteTensor(xs, Activation, K, S);
    WriteTensor(ws, ones, K, N);
    NSString *error = nil;
    if (!Evaluate(model.model, @[(__bridge id)xs, (__bridge id)ws], out, NULL, @0, &error)) {
        fprintf(stderr, "evaluation failed: %s\n", (error ?: @"unknown").UTF8String); return 2;
    }
    _Float16 *got = (_Float16 *)calloc((size_t)N * S, 2);
    ReadTensor(out, got, N, S);

    printf("ANE numerics probe: reduction against a ones vector, K=%d lanes, one dispatch\n", K);
    printf("hypotheses: naive = fp16 running sum, tiled = fp16 tiles of four into a wide accumulator\n\n");
    printf("%-20s %12s %12s %12s %12s   %s\n",
           "probe", "exact", "naive fp16", "tiled+wide", "MEASURED", "verdict");

    int agreeTiled = 0, agreeNaive = 0, agreeExact = 0, ambiguous = 0;
    for (int p = 0; p < ProbeCount; ++p) {
        double measured = (double)(float)got[0 * S + p];
        // Every output channel reduces the same column; disagreement between
        // them would mean the probe is not measuring what it claims to.
        int consistent = 1;
        for (int n = 1; n < N; ++n)
            if ((double)(float)got[n * S + p] != measured) consistent = 0;

        const Probe *probe = &Probes[p];
        int mTiled = measured == probe->tiledWide, mNaive = measured == probe->naiveFP16,
            mExact = measured == probe->exact;
        const char *verdict;
        if (mTiled && !mNaive) { verdict = "TILED+WIDE"; ++agreeTiled; }
        else if (mNaive && !mTiled) { verdict = "NAIVE FP16"; ++agreeNaive; }
        else if (mExact && mTiled && mNaive) { verdict = "all agree"; ++ambiguous; }
        else if (mExact) { verdict = "exact, wider than tiled"; ++agreeExact; }
        else { verdict = "NONE — unmodelled"; }
        printf("%-20s %12g %12g %12g %12g   %s%s\n",
               probe->name, probe->exact, probe->naiveFP16, probe->tiledWide, measured,
               verdict, consistent ? "" : "  [channels disagree!]");
    }
    printf("\nagreement: tiled+wide=%d  naive fp16=%d  exact-beyond-tiled=%d  undiscriminating=%d\n",
           agreeTiled, agreeNaive, agreeExact, ambiguous);
    printf("\nnotes\n");
    for (int p = 0; p < ProbeCount; ++p) printf("  %-20s %s\n", Probes[p].name, Probes[p].reads);

    // Phase 2. The ones weight above cannot see per-product rounding, because
    // x*1 is exact. Feed both sides a value whose square is not representable:
    // a = 1 + 2^-10 is exact in fp16, a*a = 1 + 2^-9 + 2^-20 is not. If the
    // product rounds to fp16 before the accumulator, every term loses the
    // 2^-20 and the sum falls short of exact by K * 2^-20. If the multiplier
    // hands a full-width product to the wide accumulator, the sum is exact.
    //
    // This is the mechanism the paper names for transformer loss — "the
    // per-product fp16 rounding of the inputs and weights before they enter
    // the accumulator" — and it is the one the ones-vector probes leave open.
    {
        const float a = 1.0f + 0x1p-10f;
        const double product = (double)a * (double)a;
        const double rounded = (double)(float)(_Float16)product;
        for (int i = 0; i < K * N; ++i) ones[i] = (_Float16)a;
        memset(Activation, 0, (size_t)K * S * 2);
        // Summing a*a directly cannot see the difference: the shortfall is
        // K*2^-20 against a total near K, which is ~2^9 below the fp16 output
        // resolution and rounds away in every case. The result has to land
        // near zero, where fp16's spacing is fine enough to hold it.
        //
        // So subtract the rounded product back off inside the accumulator:
        //   lanes [0,m)     x = a,           w = a   -> product a*a
        //   lanes [m,2m)    x = -fp16(a*a),  w = 1   -> product exact
        // A full-width multiply leaves m*(a*a - fp16(a*a)) = m*2^-20 standing.
        // A product rounded to fp16 leaves exactly zero.
        for (int i = 0; i < K * N; ++i) ones[i] = (_Float16)0.0f;
        int counts[] = { 256, 1024, 2048 };
        int used = (int)(sizeof counts / sizeof *counts);
        int widest = counts[used - 1];
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < widest; ++k)             ones[k * N + n] = (_Float16)a;
            for (int k = widest; k < 2 * widest; ++k)    ones[k * N + n] = (_Float16)1.0f;
        }
        for (int c = 0; c < used; ++c)
            for (int k = 0; k < counts[c]; ++k) {
                Activation[k * S + c]              = (_Float16)a;
                Activation[(widest + k) * S + c]   = (_Float16)(-(float)(_Float16)product);
            }
        WriteTensor(xs, Activation, K, S);
        WriteTensor(ws, ones, K, N);
        if (Evaluate(model.model, @[(__bridge id)xs, (__bridge id)ws], out, NULL, @0, &error)) {
            ReadTensor(out, got, N, S);
            printf("\nper-product rounding: a=1+2^-10, a*a=%.17g, fp16(a*a)=%.17g\n",
                   product, rounded);
            printf("%-14s %18s %18s %16s   %s\n",
                   "lanes m", "full-width m*2^-20", "rounded product", "MEASURED", "verdict");
            for (int c = 0; c < used; ++c) {
                double full = counts[c] * (product - rounded);
                double measured = (double)(float)got[0 * S + c];
                const char *verdict = measured == 0.0 ? "PRODUCT ROUNDED TO FP16"
                                    : fabs(measured - full) <= full * 0.25 ? "FULL-WIDTH PRODUCT"
                                    : "neither — unmodelled";
                printf("%-14d %18.3e %18.1f %16.3e   %s\n",
                       counts[c], full, 0.0, measured, verdict);
            }
            printf("\n  The residual is now near zero, where fp16 spacing is ~2^-19 and can\n"
                   "  hold m*2^-20. Zero means the multiplier rounds each product to fp16\n"
                   "  before the accumulator sees it — the per-term error that makes a dot\n"
                   "  product drift as sqrt(K) even though the accumulator itself is exact.\n");
        }
    }

    free(got); free(ones); free(Column); free(Activation);
    CFRelease(xs); CFRelease(ws); CFRelease(out); Destroy(model);
} return 0; }
