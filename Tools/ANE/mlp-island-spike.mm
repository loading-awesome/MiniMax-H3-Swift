// mlp-island-spike.mm — Can one tensor-parallel FFN shard remain on ANE?
//
// The NVIDIA-style decomposition does not return fc1 to the host. Each device
// owns matching gate/up neurons, applies SwiGLU locally, consumes the matching
// rows of fc2, and returns one hidden-width partial for the final reduction.
// This standalone probe asks only whether the installed compiler accepts that
// complete island with dynamic weights and whether its arithmetic is sane.
// It does not touch the render path, exercise two dies, or use production
// dimensions. Those are separate gates because a previous dual-attention probe
// was followed by a machine hard lock.
//
// Build and run from the repository root:
//
//   xcrun clang++ -std=c++17 -fobjc-arc -fblocks \
//     -framework Foundation -framework IOSurface \
//     Tools/ANE/mlp-island-spike.mm -o /tmp/h3-ane-mlp-island-spike
//   /tmp/h3-ane-mlp-island-spike

#define DIFF_K 128
#define DIFF_N 128
#define DIFF_S 64
#define DIFF_ITERATIONS 1
#define DIFF_DYNAMIC_ONLY 1
#define main h3_embedded_differential_main
#include "differential.m"
#undef main

#include <algorithm>
#include <vector>

enum { Hidden = DIFF_K, Intermediate = 256, Sequence = DIFF_S, Samples = 7 };

static NSString *MLPIslandMIL(void) {
    NSMutableString *m = [NSMutableString stringWithString:MILHeader()];
    [m appendFormat:
        @"    func main<ios18>(tensor<fp16,[1,%d,1,%d]> x, "
         "tensor<fp16,[1,%d,1,%d]> gate_weight, "
         "tensor<fp16,[1,%d,1,%d]> up_weight, "
         "tensor<fp16,[1,%d,1,%d]> down_weight) {\n",
        Hidden, Sequence, Hidden, Intermediate, Hidden, Intermediate,
        Intermediate, Hidden];

    [m appendFormat:
        @"        tensor<int32,[4]> rx=const()[name=string(\"rx\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "        tensor<fp16,[1,1,%d,%d]> x2=reshape(shape=rx,x=x)"
         "[name=string(\"x2\")];\n"
         "        tensor<int32,[4]> perm=const()[name=string(\"perm\"),"
         "val=tensor<int32,[4]>([0,1,3,2])];\n"
         "        tensor<fp16,[1,1,%d,%d]> xt=transpose(perm=perm,x=x2)"
         "[name=string(\"xt\")];\n",
        Hidden, Sequence, Hidden, Sequence, Sequence, Hidden];

    [m appendFormat:
        @"        tensor<int32,[4]> rw1=const()[name=string(\"rw1\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "        tensor<fp16,[1,1,%d,%d]> wg=reshape(shape=rw1,x=gate_weight)"
         "[name=string(\"wg\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> wu=reshape(shape=rw1,x=up_weight)"
         "[name=string(\"wu\")];\n"
         "        bool bf=const()[name=string(\"bf\"),val=bool(false)];\n"
         "        tensor<fp16,[1,1,%d,%d]> gate=matmul(transpose_x=bf,"
         "transpose_y=bf,x=xt,y=wg)[name=string(\"gate\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> up=matmul(transpose_x=bf,"
         "transpose_y=bf,x=xt,y=wu)[name=string(\"up\")];\n",
        Hidden, Intermediate, Hidden, Intermediate, Hidden, Intermediate,
        Sequence, Intermediate, Sequence, Intermediate];

    [m appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> gate_probability=sigmoid(x=gate)"
         "[name=string(\"gate_probability\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> gated=mul(x=gate,y=gate_probability)"
         "[name=string(\"gated\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> activation=mul(x=gated,y=up)"
         "[name=string(\"activation\")];\n"
         "        tensor<int32,[4]> rw2=const()[name=string(\"rw2\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "        tensor<fp16,[1,1,%d,%d]> wd=reshape(shape=rw2,x=down_weight)"
         "[name=string(\"wd\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> partial=matmul(transpose_x=bf,"
         "transpose_y=bf,x=activation,y=wd)[name=string(\"partial\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> partial_t=transpose(perm=perm,"
         "x=partial)[name=string(\"partial_t\")];\n"
         "        tensor<int32,[4]> ro=const()[name=string(\"ro\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n"
         "        tensor<fp16,[1,%d,1,%d]> y=reshape(shape=ro,x=partial_t)"
         "[name=string(\"y\")];\n"
         "    } -> (y);\n}\n",
        Sequence, Intermediate, Sequence, Intermediate, Sequence, Intermediate,
        Intermediate, Hidden, Intermediate, Hidden,
        Sequence, Hidden, Hidden, Sequence,
        Hidden, Sequence, Hidden, Sequence];
    return m;
}

static uint32_t Mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; x ^= x >> 16; return x;
}

static float UnitValue(uint32_t x) {
    return ((float)(Mix32(x) & 0xffffU) - 32768.0f) / 32768.0f;
}

static void FillFixture(_Float16 *x, _Float16 *gate, _Float16 *up, _Float16 *down) {
    for (int k = 0; k < Hidden; ++k) for (int s = 0; s < Sequence; ++s)
        x[(size_t)k * Sequence + s] = (_Float16)(UnitValue(k * 131U + s) * 0.15f);
    for (int k = 0; k < Hidden; ++k) for (int f = 0; f < Intermediate; ++f) {
        gate[(size_t)k * Intermediate + f] =
            (_Float16)(UnitValue(k * 977U + f + 17U) * 0.08f);
        up[(size_t)k * Intermediate + f] =
            (_Float16)(UnitValue(k * 619U + f + 31U) * 0.08f);
    }
    for (int f = 0; f < Intermediate; ++f) for (int h = 0; h < Hidden; ++h)
        down[(size_t)f * Hidden + h] =
            (_Float16)(UnitValue(f * 811U + h + 47U) * 0.06f);
}

static void Reference(const _Float16 *x, const _Float16 *gate, const _Float16 *up,
                      const _Float16 *down, float *output) {
    std::vector<float> activation(Intermediate);
    for (int s = 0; s < Sequence; ++s) {
        for (int f = 0; f < Intermediate; ++f) {
            float g = 0, u = 0;
            for (int k = 0; k < Hidden; ++k) {
                float xv = (float)x[(size_t)k * Sequence + s];
                g += xv * (float)gate[(size_t)k * Intermediate + f];
                u += xv * (float)up[(size_t)k * Intermediate + f];
            }
            activation[f] = (g / (1.0f + expf(-g))) * u;
        }
        for (int h = 0; h < Hidden; ++h) {
            float sum = 0;
            for (int f = 0; f < Intermediate; ++f)
                sum += activation[f] * (float)down[(size_t)f * Hidden + h];
            output[(size_t)h * Sequence + s] = sum;
        }
    }
}

static void Score(const _Float16 *actual, const float *expected) {
    double error2 = 0, reference2 = 0;
    float maxAbs = 0;
    for (int i = 0; i < Hidden * Sequence; ++i) {
        float delta = (float)actual[i] - expected[i];
        error2 += (double)delta * delta;
        reference2 += (double)expected[i] * expected[i];
        maxAbs = fmaxf(maxAbs, fabsf(delta));
    }
    printf("reference max_abs=%.6g rel_rms=%.6g elements=%d\n", maxAbs,
           sqrt(error2 / fmax(reference2, 1e-30)), Hidden * Sequence);
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL); mach_timebase_info(&Timebase);
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

        ExecutionOptions = @{@"kANEFProcedureVariantHint": @1,
                             @"kANEFAneInstanceHint": @1};
        uint64_t compileBegin = mach_absolute_time();
        BuiltModel model = Build(MLPIslandMIL(), @{});
        double compileMS = Milliseconds(mach_absolute_time() - compileBegin);
        if (!model.loaded) {
            printf("compile_load=REFUSED ms=%.1f error=%s\n", compileMS,
                   (model.error ?: @"unknown").UTF8String);
            Destroy(model); return 4;
        }

        _Float16 *x = (_Float16 *)calloc((size_t)Hidden * Sequence, 2);
        _Float16 *gate = (_Float16 *)calloc((size_t)Hidden * Intermediate, 2);
        _Float16 *up = (_Float16 *)calloc((size_t)Hidden * Intermediate, 2);
        _Float16 *down = (_Float16 *)calloc((size_t)Intermediate * Hidden, 2);
        _Float16 *actual = (_Float16 *)calloc((size_t)Hidden * Sequence, 2);
        float *expected = (float *)calloc((size_t)Hidden * Sequence, sizeof(float));
        FillFixture(x, gate, up, down); Reference(x, gate, up, down, expected);

        // `H3_MLP_PAD=n` oversizes every surface. On macOS 27 this graph's
        // 0x1D decodes as "IOSurface smaller than the model expects", so the
        // rejection recorded in docs/ANE_STATUS.md as evidence that the runtime
        // cannot execute a multi-weight island may only ever have been a
        // sizing error. If a multiplier makes it run, that is the answer.
        const char *padRaw = getenv("H3_MLP_PAD");
        size_t pad = padRaw ? (size_t)strtol(padRaw, NULL, 10) : 1;
        if (pad < 1) pad = 1;
        printf("surface_pad=%zux\n", pad);
        IOSurfaceRef xs = NewSurface(TensorBytes(Hidden, Sequence) * pad);
        IOSurfaceRef gs = NewSurface(TensorBytes(Hidden, Intermediate) * pad);
        IOSurfaceRef us = NewSurface(TensorBytes(Hidden, Intermediate) * pad);
        IOSurfaceRef ds = NewSurface(TensorBytes(Intermediate, Hidden) * pad);
        IOSurfaceRef ys = NewSurface(TensorBytes(Hidden, Sequence) * pad);
        WriteTensor(xs, x, Hidden, Sequence);
        WriteTensor(gs, gate, Hidden, Intermediate);
        WriteTensor(us, up, Hidden, Intermediate);
        WriteTensor(ds, down, Intermediate, Hidden);
        NSArray *inputs = @[(__bridge id)xs, (__bridge id)gs, (__bridge id)us,
                            (__bridge id)ds];

        NSString *error = nil;
        BOOL ok = EvaluateWithOptions(model.model, inputs, ys, NULL, @0,
                                      ExecutionOptions, &error);
        printf("ANE MLP island H=%d F=%d S=%d dynamic_weights=3\n",
               Hidden, Intermediate, Sequence);
        printf("compile_load=ACCEPTED ms=%.1f evaluate=%s", compileMS, ok ? "OK" : "FAIL");
        if (!ok) printf(" error=%s", (error ?: @"unknown").UTF8String);
        printf("\n");
        if (!ok) return 5;

        ReadTensor(ys, actual, Hidden, Sequence); Score(actual, expected);
        std::vector<double> times;
        for (int i = 0; i < Samples; ++i) {
            uint64_t begin = mach_absolute_time();
            if (!EvaluateWithOptions(model.model, inputs, ys, NULL, @0,
                                     ExecutionOptions, NULL)) return 5;
            times.push_back(Milliseconds(mach_absolute_time() - begin));
        }
        std::sort(times.begin(), times.end());
        printf("single_serial_median_ms=%.4f verdict=complete MLP island accepted\n",
               times[times.size() / 2]);

        free(x); free(gate); free(up); free(down); free(actual); free(expected);
        CFRelease(xs); CFRelease(gs); CFRelease(us); CFRelease(ds); CFRelease(ys);
        [NSThread sleepForTimeInterval:2.0];
        Destroy(model);
    }
    return 0;
}
