// mlp-island-chain-spike.mm — staged fallback for the persistent FFN island.
//
// The installed h15g compiler accepts a four-input fused FFN MIL program, but
// its runtime rejects inference with status 0x1d. This probe therefore uses the
// two-input dynamic matmul ABI already proven by H3ANEBridge. Gate and up remain
// in IOSurfaces, one Metal kernel applies SwiGLU while transposing directly into
// the next ANE input surface, and that activation feeds row-sharded fc2. Nothing
// here is integrated into inference and the safe default is one serial die.
//
// Build from the repository root:
//
//   xcrun clang -fobjc-arc -fblocks -I Sources/H3ANEBridge/include \
//     -c Sources/H3ANEBridge/H3ANEBridge.m -o /tmp/H3ANEBridge-spike.o
//   xcrun clang++ -std=c++17 -fobjc-arc -fblocks \
//     -I Sources/H3ANEBridge/include -c Tools/ANE/mlp-island-chain-spike.mm \
//     -o /tmp/mlp-island-chain-spike.o
//   xcrun clang++ /tmp/mlp-island-chain-spike.o /tmp/H3ANEBridge-spike.o \
//     -framework Foundation -framework IOSurface -framework Metal \
//     -o /tmp/h3-ane-mlp-island-chain-spike

#import "H3ANEBridge.h"
#import <Foundation/Foundation.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

enum { S = 64, K = 128, F = 256, H = 128 };

static float EnvScale(const char *name, float fallback) {
    const char *raw = getenv(name);
    if (!raw || !*raw) return fallback;
    float value = strtof(raw, NULL);
    return isfinite(value) && value > 0 ? value : fallback;
}

static uint32_t Mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; x ^= x >> 16; return x;
}

static float Unit(uint32_t x) {
    return ((float)(Mix(x) & 0xffffU) - 32768.0f) / 32768.0f;
}

static void Fill(std::vector<_Float16>& x, std::vector<_Float16>& gate,
                 std::vector<_Float16>& up, std::vector<_Float16>& down) {
    for (int k = 0; k < K; ++k) for (int s = 0; s < S; ++s)
        x[(size_t)k*S+s] = (_Float16)(Unit(k*131U+s) * 1.00f);
    for (int k = 0; k < K; ++k) for (int f = 0; f < F; ++f) {
        gate[(size_t)k*F+f] = (_Float16)(Unit(k*977U+f+17U) * 1.50f);
        up[(size_t)k*F+f] = (_Float16)(Unit(k*619U+f+31U) * 1.50f);
    }
    for (int f = 0; f < F; ++f) for (int h = 0; h < H; ++h)
        down[(size_t)f*H+h] = (_Float16)(Unit(f*811U+h+47U) * 0.03f);
}

static void Reference(const std::vector<_Float16>& x,
                      const std::vector<_Float16>& gate,
                      const std::vector<_Float16>& up,
                      const std::vector<_Float16>& down,
                      std::vector<float>& output) {
    std::vector<float> activation(F);
    for (int s = 0; s < S; ++s) {
        for (int f = 0; f < F; ++f) {
            float g = 0, u = 0;
            for (int k = 0; k < K; ++k) {
                float xv = (float)x[(size_t)k*S+s];
                g += xv * (float)gate[(size_t)k*F+f];
                u += xv * (float)up[(size_t)k*F+f];
            }
            activation[f] = (g / (1.0f + expf(-g))) * u;
        }
        for (int h = 0; h < H; ++h) {
            float sum = 0;
            for (int f = 0; f < F; ++f)
                sum += activation[f] * (float)down[(size_t)f*H+h];
            output[(size_t)s*H+h] = sum;
        }
    }
}

static void Score(const _Float16 *actual, const std::vector<float>& expected,
                  float outputUnscale) {
    double error2 = 0, reference2 = 0;
    float maxAbs = 0;
    for (int i = 0; i < S*H; ++i) {
        float d = (float)actual[i] * outputUnscale - expected[i];
        error2 += (double)d*d; reference2 += (double)expected[i]*expected[i];
        maxAbs = fmaxf(maxAbs, fabsf(d));
    }
    printf("reference max_abs=%.6g rel_rms=%.6g elements=%d\n", maxAbs,
           sqrt(error2 / fmax(reference2, 1e-30)), S*H);
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        if (!h3_ane_is_available()) {
            fprintf(stderr, "ANE bridge unavailable on this machine/OS fingerprint\n"); return 2;
        }
        H3ANEProgram *gateProgram = h3_ane_program_create(S, K, F);
        H3ANEProgram *upProgram = h3_ane_program_create(S, K, F);
        H3ANEProgram *downProgram = h3_ane_program_create(S, F, H);
        if (!gateProgram || !upProgram || !downProgram) {
            fprintf(stderr, "program construction failed\n"); return 3;
        }

        H3ANETensor *xTensor = h3_ane_tensor_create(K, S);
        H3ANETensor *gateWeight = h3_ane_tensor_create(K, F);
        H3ANETensor *upWeight = h3_ane_tensor_create(K, F);
        H3ANETensor *downWeight = h3_ane_tensor_create(F, H);
        H3ANETensor *gateOut = h3_ane_tensor_create(S, F);
        H3ANETensor *upOut = h3_ane_tensor_create(S, F);
        H3ANETensor *activationTensor = h3_ane_tensor_create(F, S);
        H3ANETensor *partialOut = h3_ane_tensor_create(S, H);
        if (!xTensor || !gateWeight || !upWeight || !downWeight || !gateOut ||
            !upOut || !activationTensor || !partialOut) {
            fprintf(stderr, "tensor allocation failed\n"); return 3;
        }

        std::vector<_Float16> x((size_t)K*S), scaledX((size_t)K*S),
                              gate((size_t)K*F), up((size_t)K*F), down((size_t)F*H);
        std::vector<float> expected((size_t)S*H);
        Fill(x, gate, up, down); Reference(x, gate, up, down, expected);
        float fc1Scale = EnvScale("H3_MLP_FC1_SCALE", 1.0f);
        float fc2Scale = EnvScale("H3_MLP_FC2_SCALE", 1.0f);
        for (size_t i = 0; i < x.size(); ++i)
            scaledX[i] = (_Float16)((float)x[i] * fc1Scale);
        if (!h3_ane_tensor_write(xTensor, scaledX.data(), K, S) ||
            !h3_ane_tensor_write(gateWeight, gate.data(), K, F) ||
            !h3_ane_tensor_write(upWeight, up.data(), K, F) ||
            !h3_ane_tensor_write(downWeight, down.data(), F, H)) {
            fprintf(stderr, "tensor upload failed\n"); return 3;
        }

        auto runIsland = [&]() -> bool {
            return h3_ane_run(gateProgram, xTensor, gateWeight, gateOut, 1) &&
                   h3_ane_run(upProgram, xTensor, upWeight, upOut, 1) &&
                   // The Metal seam reads gate/up in place and writes the
                   // transposed fp16 activation directly into fc2's input.
                   h3_ane_swiglu_transpose_fp16(gateOut, upOut, activationTensor,
                                                S, F, 1.0f / fc1Scale, fc2Scale,
                                                NULL) &&
                   h3_ane_run(downProgram, activationTensor, downWeight,
                              partialOut, 1);
        };
        // Runtime Metal source compilation is a process-start cost and is not
        // part of a block. Warm the complete island once before timing it.
        if (!runIsland()) { fprintf(stderr, "MLP island warm-up failed\n"); return 4; }
        uint64_t begin = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
        if (!runIsland()) { fprintf(stderr, "MLP island evaluation failed\n"); return 4; }
        double elapsed = (clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)-begin)/1e6;
        printf("ANE chained MLP island S=%d K=%d F=%d H=%d mode=serial "
               "fc1_scale=%g fc2_scale=%g\n", S,K,F,H,fc1Scale,fc2Scale);
        Score((const _Float16 *)h3_ane_tensor_ptr(partialOut), expected,
              1.0f / fc2Scale);
        printf("wall_ms=%.4f verdict=two-input chained island executes\n", elapsed);

        h3_ane_tensor_free(xTensor); h3_ane_tensor_free(gateWeight);
        h3_ane_tensor_free(upWeight); h3_ane_tensor_free(downWeight);
        h3_ane_tensor_free(gateOut); h3_ane_tensor_free(upOut);
        h3_ane_tensor_free(activationTensor); h3_ane_tensor_free(partialOut);
        [NSThread sleepForTimeInterval:2.0];
        h3_ane_program_free(gateProgram); h3_ane_program_free(upProgram);
        h3_ane_program_free(downProgram);
    }
    return 0;
}
