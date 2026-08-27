// mlp-island-dual-spike.mm — execute the intended two-die persistent schedule.
//
// Each die owns a disjoint SwiGLU neuron range. Gate projections run as a pair,
// up projections run as a pair, both Metal seams share one command buffer, and
// the two row-sharded fc2 projections run as a pair. Only the two hidden-width
// partials leave the islands.
//
// Build from the repository root:
//
//   xcrun clang -fobjc-arc -fblocks -I Sources/H3ANEBridge/include \
//     -c Sources/H3ANEBridge/H3ANEBridge.m -o /tmp/H3ANEBridge-dual-spike.o
//   xcrun clang++ -std=c++17 -fobjc-arc -fblocks \
//     -I Sources/H3ANEBridge/include -c Tools/ANE/mlp-island-dual-spike.mm \
//     -o /tmp/mlp-island-dual-spike.o
//   xcrun clang++ /tmp/mlp-island-dual-spike.o /tmp/H3ANEBridge-dual-spike.o \
//     -framework Foundation -framework IOSurface -framework Metal \
//     -o /tmp/h3-ane-mlp-island-dual-spike

#import "H3ANEBridge.h"
#import <Foundation/Foundation.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>

static int S = 64, K = 128, F = 128, H = 128;
static bool Production = false, Serial = false;

static uint32_t Mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; x ^= x >> 16; return x;
}

static float Unit(uint32_t x) {
    return ((float)(Mix(x) & 0xffffU) - 32768.0f) / 32768.0f;
}

struct Island {
    H3ANEProgram *gateProgram = nullptr, *upProgram = nullptr, *downProgram = nullptr;
    H3ANETensor *x = nullptr, *gateWeight = nullptr, *upWeight = nullptr;
    H3ANETensor *downWeight = nullptr, *gateOut = nullptr, *upOut = nullptr;
    H3ANETensor *activation = nullptr, *partial = nullptr;
    std::vector<_Float16> gate, up, down;
    float fc2Scale;
};

static bool Allocate(Island& a, float fc2Scale) {
    a.fc2Scale = fc2Scale;
    a.gateProgram = h3_ane_program_create(S, K, F);
    a.upProgram = h3_ane_program_create(S, K, F);
    a.downProgram = h3_ane_program_create(S, F, H);
    a.x = h3_ane_tensor_create(K, S);
    a.gateWeight = h3_ane_tensor_create(K, F);
    a.upWeight = h3_ane_tensor_create(K, F);
    a.downWeight = h3_ane_tensor_create(F, H);
    a.gateOut = h3_ane_tensor_create(S, F);
    a.upOut = h3_ane_tensor_create(S, F);
    a.activation = h3_ane_tensor_create(F, S);
    a.partial = h3_ane_tensor_create(S, H);
    return a.gateProgram && a.upProgram && a.downProgram && a.x && a.gateWeight &&
           a.upWeight && a.downWeight && a.gateOut && a.upOut && a.activation && a.partial;
}

static void Release(Island& a) {
    h3_ane_tensor_free(a.x); h3_ane_tensor_free(a.gateWeight);
    h3_ane_tensor_free(a.upWeight); h3_ane_tensor_free(a.downWeight);
    h3_ane_tensor_free(a.gateOut); h3_ane_tensor_free(a.upOut);
    h3_ane_tensor_free(a.activation); h3_ane_tensor_free(a.partial);
    [NSThread sleepForTimeInterval:1.0];
    h3_ane_program_free(a.gateProgram); h3_ane_program_free(a.upProgram);
    h3_ane_program_free(a.downProgram);
}

static void Fill(Island& a, int die) {
    a.gate.resize((size_t)K*F); a.up.resize((size_t)K*F);
    a.down.resize((size_t)F*H);
    float fc1Amplitude = Production ? 0.02f : 1.5f;
    float fc2Amplitude = Production ? 0.02f : 0.03f;
    for (int k = 0; k < K; ++k) for (int f = 0; f < F; ++f) {
        a.gate[(size_t)k*F+f] = (_Float16)(Unit(k*977U+f+17U+die*101U) * fc1Amplitude);
        a.up[(size_t)k*F+f] = (_Float16)(Unit(k*619U+f+31U+die*211U) * fc1Amplitude);
    }
    for (int f = 0; f < F; ++f) for (int h = 0; h < H; ++h)
        a.down[(size_t)f*H+h] = (_Float16)(Unit(f*811U+h+47U+die*307U) * fc2Amplitude);
}

static void Reference(const std::vector<_Float16>& x, const Island& a,
                      std::vector<float>& output) {
    std::vector<float> activation(F);
    for (int s = 0; s < S; ++s) {
        for (int f = 0; f < F; ++f) {
            float g = 0, u = 0;
            for (int k = 0; k < K; ++k) {
                float xv = (float)x[(size_t)k*S+s];
                g += xv * (float)a.gate[(size_t)k*F+f];
                u += xv * (float)a.up[(size_t)k*F+f];
            }
            activation[f] = (g / (1.0f + expf(-g))) * u;
        }
        for (int h = 0; h < H; ++h) for (int f = 0; f < F; ++f)
            output[(size_t)s*H+h] += activation[f] * (float)a.down[(size_t)f*H+h];
    }
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        if (!h3_ane_is_available()) return 2;
        Production = getenv("H3_MLP_PRODUCTION") != NULL;
        Serial = getenv("H3_MLP_SERIAL") != NULL;
        if (Production) {
            // Pad S to the IOSurface's 64-byte fp16 row alignment. The real
            // 15,461 tokens occupy the prefix; zero padding does not change a
            // row-wise MLP and is excluded from render integration.
            S = 15488; K = 7168; F = 3584; H = 7168;
            if (const char *raw = getenv("H3_MLP_F_PER_DIE")) {
                int requested = atoi(raw);
                if (requested > 0 && requested % 32 == 0 && requested * 2 < 14336)
                    F = requested;
            }
        }
        printf("config S=%d K=%d F_per_die=%d H=%d mode=%s\n", S,K,F,H,
               Serial ? "sequential" : "paired");
        Island a0, a1;
        printf("phase=compile begin\n");
        if (!Allocate(a0, Production ? 1.0f/2.0f : 1.0f/16.0f) ||
            !Allocate(a1, 1.0f/256.0f)) {
            fprintf(stderr, "dual-island allocation failed\n"); return 3;
        }
        printf("phase=compile complete\nphase=fill begin\n");
        Fill(a0, 0); Fill(a1, 1);
        std::vector<_Float16> x((size_t)K*S), scaledX((size_t)K*S);
        for (int k = 0; k < K; ++k) for (int s = 0; s < S; ++s) {
            x[(size_t)k*S+s] = (_Float16)Unit(k*131U+s);
            scaledX[(size_t)k*S+s] = (_Float16)((float)x[(size_t)k*S+s] / 16.0f);
        }
        for (Island *a : {&a0, &a1}) {
            if (!h3_ane_tensor_write(a->x, scaledX.data(), K, S) ||
                !h3_ane_tensor_write(a->gateWeight, a->gate.data(), K, F) ||
                !h3_ane_tensor_write(a->upWeight, a->up.data(), K, F) ||
                !h3_ane_tensor_write(a->downWeight, a->down.data(), F, H)) {
                fprintf(stderr, "dual-island upload failed\n"); return 3;
            }
        }
        printf("phase=fill complete\n");
        std::vector<float> expected;
        if (!Production) {
            expected.assign((size_t)S*H, 0.0f);
            Reference(x, a0, expected); Reference(x, a1, expected);
        }

        auto pairOrSerial = [&](H3ANEProgram *p0, H3ANETensor *x0,
                                H3ANETensor *w0, H3ANETensor *y0,
                                H3ANEProgram *p1, H3ANETensor *x1,
                                H3ANETensor *w1, H3ANETensor *y1) {
            return Serial ? (h3_ane_run(p0, x0, w0, y0, 1) &&
                             h3_ane_run(p1, x1, w1, y1, 2))
                          : h3_ane_run_pair(p0, x0, w0, y0, p1, x1, w1, y1);
        };

        auto run = [&]() {
            printf("phase=gate begin\n");
            if (!pairOrSerial(a0.gateProgram, a0.x, a0.gateWeight, a0.gateOut,
                              a1.gateProgram, a1.x, a1.gateWeight, a1.gateOut)) return false;
            printf("phase=gate complete\nphase=up begin\n");
            if (!pairOrSerial(a0.upProgram, a0.x, a0.upWeight, a0.upOut,
                              a1.upProgram, a1.x, a1.upWeight, a1.upOut)) return false;
            printf("phase=up complete\nphase=seam begin\n");
            if (!h3_ane_swiglu_transpose_pair_fp16(
                        a0.gateOut, a0.upOut, a0.activation, a0.fc2Scale,
                        a1.gateOut, a1.upOut, a1.activation, a1.fc2Scale,
                        S, F, 16.0f, NULL)) return false;
            printf("phase=seam complete\nphase=fc2 begin\n");
            bool ok = pairOrSerial(
                        a0.downProgram, a0.activation, a0.downWeight, a0.partial,
                        a1.downProgram, a1.activation, a1.downWeight, a1.partial);
            printf("phase=fc2 %s\n", ok ? "complete" : "failed");
            return ok;
        };
        if (!run()) { fprintf(stderr, "dual-island warm-up failed\n"); return 4; }
        int repeats = 1;
        if (const char *raw = getenv("H3_MLP_REPEATS")) repeats = std::max(1, atoi(raw));
        std::vector<double> times;
        for (int iteration = 0; iteration < repeats; ++iteration) {
            uint64_t begin = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
            if (!run()) { fprintf(stderr, "dual-island evaluation failed\n"); return 4; }
            times.push_back((clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)-begin)/1e6);
        }
        std::sort(times.begin(), times.end());
        double elapsed = times[times.size()/2];

        if (!Production) {
            const _Float16 *p0 = (const _Float16 *)h3_ane_tensor_ptr(a0.partial);
            const _Float16 *p1 = (const _Float16 *)h3_ane_tensor_ptr(a1.partial);
            double error2 = 0, reference2 = 0; float maxAbs = 0;
            for (int i = 0; i < S*H; ++i) {
                float actual = (float)p0[i] / a0.fc2Scale + (float)p1[i] / a1.fc2Scale;
                float d = actual - expected[i];
                error2 += (double)d*d; reference2 += (double)expected[i]*expected[i];
                maxAbs = fmaxf(maxAbs, fabsf(d));
            }
            printf("scales=(1/16,1/256) max_abs=%.6g rel_rms=%.6g wall_ms=%.4f\n",
                   maxAbs, sqrt(error2/fmax(reference2,1e-30)), elapsed);
        } else {
            printf("scales=(1/2,1/256) median_ms=%.4f repeats=%d "
                   "accuracy=covered-by-production-capture\n", elapsed, repeats);
        }
        if (Production && getenv("H3_MLP_MERGE_BENCH")) {
            H3ANETensor *p0[4] = {a0.partial, nullptr, nullptr, nullptr};
            H3ANETensor *p1[4] = {a1.partial, nullptr, nullptr, nullptr};
            bool allocated = true;
            for (int i = 1; i < 4; ++i) {
                p0[i] = h3_ane_tensor_create(S, H);
                p1[i] = h3_ane_tensor_create(S, H);
                allocated = allocated && p0[i] && p1[i];
            }
            std::vector<uint16_t> gpu((size_t)S*H), joined((size_t)S*H);
            auto merge = [&]() {
                return allocated && h3_ane_merge_mlp_island_partials(
                    gpu.data(), p0, 1.0f/a0.fc2Scale, p1, 1.0f/a1.fc2Scale,
                    joined.data(), S, H, NULL);
            };
            if (!merge()) { fprintf(stderr, "production merge warm-up failed\n"); return 5; }
            std::vector<double> mergeTimes;
            for (int i = 0; i < repeats; ++i) {
                uint64_t b = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
                if (!merge()) { fprintf(stderr, "production merge failed\n"); return 5; }
                mergeTimes.push_back((clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)-b)/1e6);
            }
            std::sort(mergeTimes.begin(), mergeTimes.end());
            printf("final_join_median_ms=%.4f reads=GPU+8_ANE writes=bf16\n",
                   mergeTimes[mergeTimes.size()/2]);
            for (int i = 1; i < 4; ++i) {
                h3_ane_tensor_free(p0[i]); h3_ane_tensor_free(p1[i]);
            }
        }
        printf("ANE dual MLP island S=%d K=%d F_per_die=%d H=%d\n", S,K,F,H);
        printf("verdict=gate/up/fc2 pairs plus paired Metal seam execute\n");
        Release(a0); Release(a1);
    }
    return 0;
}
