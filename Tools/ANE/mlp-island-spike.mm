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

#ifndef H3_MLP_HIDDEN
#define H3_MLP_HIDDEN DIFF_K
#endif
#ifndef H3_MLP_INTERMEDIATE
#define H3_MLP_INTERMEDIATE 256
#endif
#ifndef H3_MLP_SEQUENCE
#define H3_MLP_SEQUENCE DIFF_S
#endif
enum {
    Hidden = H3_MLP_HIDDEN,
    Intermediate = H3_MLP_INTERMEDIATE,
    Sequence = H3_MLP_SEQUENCE,
    Samples = 7
};

static NSString *MLPIslandMIL(void) {
    const char *scaleRaw = getenv("H3_MLP_INTERNAL_SCALE");
    int internalScale = scaleRaw && *scaleRaw ? (int)strtol(scaleRaw, NULL, 10) : 1;
    if (internalScale < 1) internalScale = 1;
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
         "        bool bf=const()[name=string(\"bf\"),val=bool(false)];\n",
        Hidden, Intermediate, Hidden, Intermediate, Hidden, Intermediate];
    NSString *fc1Input = @"xt";
    NSString *gateInput = @"gate_raw";
    NSString *upInput = @"up_raw";
    if (internalScale > 1) {
        [m appendFormat:
            @"        fp16 scale=const()[name=string(\"scale\"),val=fp16(%d.0)];\n"
             "        fp16 inv_scale=const()[name=string(\"inv_scale\"),val=fp16(%.9g)];\n"
             "        tensor<fp16,[1,1,%d,%d]> xt_scaled=mul(x=xt,y=scale)"
             "[name=string(\"xt_scaled\")];\n",
            internalScale, 1.0 / internalScale, Sequence, Hidden];
        fc1Input = @"xt_scaled";
    }
    [m appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> gate_raw=matmul(transpose_x=bf,"
         "transpose_y=bf,x=%@,y=wg)[name=string(\"gate_raw\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> up_raw=matmul(transpose_x=bf,"
         "transpose_y=bf,x=%@,y=wu)[name=string(\"up_raw\")];\n",
        Sequence, Intermediate, fc1Input, Sequence, Intermediate, fc1Input];
    if (internalScale > 1) {
        [m appendFormat:
            @"        tensor<fp16,[1,1,%d,%d]> gate=mul(x=gate_raw,y=inv_scale)"
             "[name=string(\"gate\")];\n"
             "        tensor<fp16,[1,1,%d,%d]> up=mul(x=up_raw,y=inv_scale)"
             "[name=string(\"up\")];\n",
            Sequence, Intermediate, Sequence, Intermediate];
        gateInput = @"gate";
        upInput = @"up";
    }

    [m appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> gate_probability=sigmoid(x=%@)"
         "[name=string(\"gate_probability\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> gated=mul(x=%@,y=gate_probability)"
         "[name=string(\"gated\")];\n"
         "        tensor<fp16,[1,1,%d,%d]> activation=mul(x=gated,y=%@)"
         "[name=string(\"activation\")];\n"
         "        tensor<int32,[4]> rw2=const()[name=string(\"rw2\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "        tensor<fp16,[1,1,%d,%d]> wd=reshape(shape=rw2,x=down_weight)"
         "[name=string(\"wd\")];\n",
        Sequence, Intermediate, gateInput,
        Sequence, Intermediate, gateInput,
        Sequence, Intermediate, upInput,
        Intermediate, Hidden, Intermediate, Hidden];

    NSString *fc2Input = @"activation";
    if (internalScale > 1) {
        [m appendFormat:
            @"        tensor<fp16,[1,1,%d,%d]> activation_scaled="
             "mul(x=activation,y=scale)[name=string(\"activation_scaled\")];\n",
            Sequence, Intermediate];
        fc2Input = @"activation_scaled";
    }
    [m appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> partial_raw=matmul(transpose_x=bf,"
         "transpose_y=bf,x=%@,y=wd)[name=string(\"partial_raw\")];\n",
        Sequence, Hidden, fc2Input];
    NSString *partialInput = @"partial_raw";
    if (internalScale > 1) {
        [m appendFormat:
            @"        tensor<fp16,[1,1,%d,%d]> partial="
             "mul(x=partial_raw,y=inv_scale)[name=string(\"partial\")];\n",
            Sequence, Hidden];
        partialInput = @"partial";
    }
    [m appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> partial_t=transpose(perm=perm,"
         "x=%@)[name=string(\"partial_t\")];\n"
         "        tensor<int32,[4]> ro=const()[name=string(\"ro\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n"
         "        tensor<fp16,[1,%d,1,%d]> y=reshape(shape=ro,x=partial_t)"
         "[name=string(\"y\")];\n"
         "    } -> (y);\n}\n",
        Sequence, Hidden, partialInput, Hidden, Sequence,
        Hidden, Sequence];
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
    const char *scaleRaw = getenv("H3_MLP_FIXTURE_SCALE");
    float scale = scaleRaw && *scaleRaw ? strtof(scaleRaw, NULL) : 1.0f;
    if (!(scale > 0)) scale = 1.0f;
    for (int k = 0; k < Hidden; ++k) for (int s = 0; s < Sequence; ++s)
        x[(size_t)k * Sequence + s] = (_Float16)(UnitValue(k * 131U + s) * 0.15f * scale);
    for (int k = 0; k < Hidden; ++k) for (int f = 0; f < Intermediate; ++f) {
        gate[(size_t)k * Intermediate + f] =
            (_Float16)(UnitValue(k * 977U + f + 17U) * 0.08f * scale);
        up[(size_t)k * Intermediate + f] =
            (_Float16)(UnitValue(k * 619U + f + 31U) * 0.08f * scale);
    }
    for (int f = 0; f < Intermediate; ++f) for (int h = 0; h < Hidden; ++h)
        down[(size_t)f * Hidden + h] =
            (_Float16)(UnitValue(f * 811U + h + 47U) * 0.06f * scale);
    printf("fixture_scale=%.4g\n", scale);
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
    double error2 = 0, reference2 = 0, actual2 = 0, dot = 0;
    float maxAbs = 0;
    for (int h = 0; h < Hidden; ++h) for (int s = 0; s < Sequence; ++s) {
        int i = h * Sequence + s;
        float value = (float)actual[i];
        float delta = value - expected[i];
        error2 += (double)delta * delta;
        reference2 += (double)expected[i] * expected[i];
        actual2 += (double)value * value;
        dot += (double)value * expected[i];
        maxAbs = fmaxf(maxAbs, fabsf(delta));
    }
    printf("reference max_abs=%.6g rel_rms=%.6g cosine=%.6g norm_ratio=%.6g\n",
           maxAbs, sqrt(error2 / fmax(reference2, 1e-30)),
           dot / sqrt(fmax(actual2 * reference2, 1e-30)),
           sqrt(actual2 / fmax(reference2, 1e-30)));
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
        if (getenv("H3_ANE_DUMP_ATTRIBUTES") &&
            [model.model respondsToSelector:@selector(modelAttributes)]) {
            id attributes = ((id(*)(id,SEL))objc_msgSend)(model.model,
                @selector(modelAttributes));
            printf("model_attributes=%s\n",
                   attributes ? [[attributes description] UTF8String] : "(nil)");
        }

        _Float16 *x = (_Float16 *)calloc((size_t)Hidden * Sequence, 2);
        _Float16 *gate = (_Float16 *)calloc((size_t)Hidden * Intermediate, 2);
        _Float16 *up = (_Float16 *)calloc((size_t)Hidden * Intermediate, 2);
        _Float16 *down = (_Float16 *)calloc((size_t)Intermediate * Hidden, 2);
        _Float16 *actual = (_Float16 *)calloc((size_t)Hidden * Sequence, 2);
        float *expected = (float *)calloc((size_t)Hidden * Sequence, sizeof(float));
        FillFixture(x, gate, up, down); Reference(x, gate, up, down, expected);

        IOSurfaceRef xs = NewSurface(TensorBytes(Hidden, Sequence));
        IOSurfaceRef gs = NewSurface(TensorBytes(Hidden, Intermediate));
        IOSurfaceRef us = NewSurface(TensorBytes(Hidden, Intermediate));
        IOSurfaceRef ds = NewSurface(TensorBytes(Intermediate, Hidden));
        IOSurfaceRef ys = NewSurface(TensorBytes(Hidden, Sequence));
        WriteTensor(xs, x, Hidden, Sequence);
        WriteTensor(gs, gate, Hidden, Intermediate);
        WriteTensor(us, up, Hidden, Intermediate);
        WriteTensor(ds, down, Intermediate, Hidden);
        // The compiler does not preserve textual function-argument order. Its
        // model attributes assign input symbols alphabetically here:
        // down_weight=0, gate_weight=1, up_weight=2, x=3. Binding the textual
        // order made the 16 KiB x surface occupy the 64 KiB down slot (0x1d);
        // padding x to 64 KiB only hid that error and ran with swapped tensors.
        NSArray *inputs = @[(__bridge id)ds, (__bridge id)gs,
                            (__bridge id)us, (__bridge id)xs];

        NSString *error = nil;
        BOOL ok = EvaluateWithOptions(model.model, inputs, ys, NULL, @0,
                                      ExecutionOptions, &error);
        printf("ANE MLP island H=%d F=%d S=%d dynamic_weights=3\n",
               Hidden, Intermediate, Sequence);
        printf("compile_load=ACCEPTED ms=%.1f evaluate=%s", compileMS, ok ? "OK" : "FAIL");
        if (!ok) printf(" error=%s", (error ?: @"unknown").UTF8String);
        printf("\n");
        if (!ok) return 5;

        ReadTensor(ys, actual, Hidden, Sequence);
        Score(actual, expected);
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
