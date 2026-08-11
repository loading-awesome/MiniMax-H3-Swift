// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import H3Foundation
@testable import H3Modules

/// Can a hand-written Metal GEMM reach MPS's rate from inside MLX's stream?
///
/// This is the prototype section 9 asks for, and it exists because of two
/// measurements rather than a hunch:
///
///  * §8 — MPSGraph reaches **19.2 TFLOP/s** at this exact shape where MLX
///    reaches 17.0, bit-identically. The hardware has headroom MLX is not
///    taking. That is a demonstrated number, not a hoped-for one.
///  * §9 — routing out to MPSGraph costs **~28 ms per crossing**, which is why
///    §8's win did not survive a block. `MLXFast.metalKernel` has no crossing at
///    all: MLX JIT-compiles the source and dispatches it in its own command
///    stream, so a kernel that is merely *as fast* as MLX's is free, and one
///    that is faster is a straight win.
///
/// **fc1 is the right shape to prototype.** It is the largest single GEMM in a
/// block — 292.7 ms of 1224 ms, 25.2% of the forward's arithmetic — so it
/// carries the most signal, and `[S,H]x[H,2F]` divides cleanly: K = 5376 = 168
/// steps of 32, N = 28672 = 448 tiles of 64. Only M = 15,731 is ragged, and it
/// costs one predicated tile row.
///
///     H3_BIG=1 swift test --filter customGEMM
///
/// **What would make this fail is not subtle.** MLX's steel kernels are tuned;
/// a first attempt landing near 10 TFLOP/s means the tiling is wrong, and one
/// landing near 17 means MLX's schedule is close to what this design can do and
/// the remaining 2 TFLOP/s lives somewhere this prototype cannot see.
@Suite("custom GEMM", .serialized)
struct CustomGEMMTests {

    // The fc1 shape, from `H3Config`.
    static let m = 15_731
    static let k = 5_376                                 // hiddenSize
    static let n = 28_672                                // 2 * ffnHidden

    // Tile geometry, and it is the whole result.
    //
    // A 64x64 tile was the first attempt and it reached 4.1 TFLOP/s — correct,
    // and memory-bound before it started. The ratio that decides this is
    // arithmetic intensity per tile: a BMxBN tile does `2*BM*BN*BK` FLOP per
    // `2*(BM+BN)*BK` bytes of global traffic, which is `BM*BN/(BM+BN)` FLOP per
    // byte and depends on the tile, not on BK.
    //
    //     64x64    32 FLOP/byte  ->  594 GB/s needed at 19 TFLOP/s
    //     128x128  64 FLOP/byte  ->  297 GB/s needed at 19 TFLOP/s
    //
    // The first number is most of this machine's bandwidth, so the kernel could
    // not have reached MLX's rate at any occupancy. 256 threads = 8 simdgroups
    // in a 4x2 arrangement, each owning 32x64 = a 4x8 grid of 8x8 accumulators.
    /// One tiling to measure.
    ///
    /// `sgM x sgN` simdgroups tile the block, so each owns
    /// `(bm/sgM) x (bn/sgN)` outputs — `(bm/sgM/8) * (bn/sgN/8)` accumulators of
    /// `simdgroup_matrix<float,8,8>`, at 2 floats per lane each. That product is
    /// the register budget, and it is the axis the first two attempts moved
    /// without meaning to.
    struct Tiling {
        let bm: Int, bn: Int, bk: Int, sgM: Int, sgN: Int
        var threads: Int { sgM * sgN * 32 }
        var accumulators: Int { (bm / sgM / 8) * (bn / sgN / 8) }
        var intensity: Double { Double(bm * bn) / Double(bm + bn) }
        var label: String { "\(bm)x\(bn)x\(bk) \(sgM)x\(sgN)sg" }
    }

    /// Builds the kernel source for one tiling.
    static func source(_ t: Tiling) -> String {
        let fm = t.bm / t.sgM / 8, fn = t.bn / t.sgN / 8
        return """
        constexpr int BM = \(t.bm);
        constexpr int BN = \(t.bn);
        constexpr int BK = \(t.bk);
        constexpr int M = \(m);
        constexpr int K = \(k);
        constexpr int N = \(n);

        threadgroup bfloat As[BM * BK];
        threadgroup bfloat Bs[BK * BN];
        threadgroup float  edge[\(t.sgM * t.sgN) * 64];

        device const bfloat* A = (device const bfloat*)a;
        device const bfloat* B = (device const bfloat*)b;
        device bfloat* C = (device bfloat*)out;

        uint tid   = thread_position_in_threadgroup.x;
        uint sg    = tid / 32;
        uint lane  = tid % 32;
        int row0 = int(threadgroup_position_in_grid.y) * BM;
        int col0 = int(threadgroup_position_in_grid.x) * BN;
        int sgM  = int(sg / \(t.sgN)) * \(t.bm / t.sgM);
        int sgN  = int(sg % \(t.sgN)) * \(t.bn / t.sgN);

        simdgroup_matrix<float, 8, 8> acc[\(fm)][\(fn)];
        for (int i = 0; i < \(fm); i++) {
            for (int j = 0; j < \(fn); j++) {
                acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }

        for (int k0 = 0; k0 < K; k0 += BK) {
            for (int idx = int(tid); idx < BM * BK; idx += \(t.threads)) {
                int r = idx / BK, c = idx % BK;
                int gr = row0 + r;
                As[r * BK + c] = (gr < M) ? A[gr * K + k0 + c] : bfloat(0.0f);
            }
            for (int idx = int(tid); idx < BK * BN; idx += \(t.threads)) {
                int r = idx / BN, c = idx % BN;
                Bs[r * BN + c] = B[(k0 + r) * N + col0 + c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int kk = 0; kk < BK; kk += 8) {
                simdgroup_matrix<bfloat, 8, 8> af[\(fm)], bf[\(fn)];
                for (int i = 0; i < \(fm); i++) {
                    simdgroup_load(af[i], As + (sgM + i * 8) * BK + kk, BK);
                }
                for (int j = 0; j < \(fn); j++) {
                    simdgroup_load(bf[j], Bs + kk * BN + sgN + j * 8, BN);
                }
                for (int i = 0; i < \(fm); i++) {
                    for (int j = 0; j < \(fn); j++) {
                        simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // fp32 accumulators into a bf16 output: `simdgroup_store` needs the
        // matrix and the destination to agree on type, so each 8x8 tile lands in
        // a per-simdgroup float scratch and converts on the way out. That scratch
        // doubles as the bounds check the ragged M edge needs. Epilogue only —
        // it runs once per output element, not once per K step.
        threadgroup float* mine = edge + sg * 64;
        for (int i = 0; i < \(fm); i++) {
            int gr = row0 + sgM + i * 8;
            for (int j = 0; j < \(fn); j++) {
                int gc = col0 + sgN + j * 8;
                simdgroup_store(acc[i][j], mine, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int e = int(lane); e < 64; e += 32) {
                    int r = e / 8, c = e % 8;
                    if (gr + r < M) {
                        C[(gr + r) * N + gc + c] = bfloat(mine[e]);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        """
    }

    static func kernel(_ t: Tiling) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "h3_gemm_bf16_\(t.bm)_\(t.bn)_\(t.bk)_\(t.sgM)_\(t.sgN)",
            inputNames: ["a", "b"], outputNames: ["out"],
            source: source(t),
            header: """
                #include <metal_stdlib>
                #include <metal_simdgroup_matrix>
                using namespace metal;
                """)
    }

    /// A shape-specialised wrapper around the same Steel building blocks MLX's
    /// regular matmul uses on an Ultra.  Keeping this beside the from-scratch
    /// kernel makes the comparison useful: it separates integration overhead
    /// from the loader/MMA schedule that the prototype is trying to reproduce.
    ///
    /// The important differences from `source(_:)` are Steel's aligned vector
    /// loader, 16-byte threadgroup-row padding, its register-fragment layout,
    /// and direct fp32-accumulator to bf16 device stores.  No split-K and no
    /// second dispatch are involved.
    static func steelKernel() -> MLXFast.MLXFastKernel {
        let source = """
        constexpr short BM = 64;
        constexpr short BN = 64;
        constexpr short BK = 16;
        constexpr short WM = 1;
        constexpr short WN = 2;
        constexpr short PAD = 16 / sizeof(bfloat);
        constexpr short LDA_TGP = BK + PAD;
        constexpr short LDB_TGP = BN + PAD;
        constexpr short TGP_SIZE = WM * WN * 32;

        threadgroup bfloat As[BM * LDA_TGP];
        threadgroup bfloat Bs[BK * LDB_TGP];

        uint sgid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tid = thread_position_in_threadgroup.x;
        constexpr int SWIZZLE_LOG = 2;
        int tile_m = (int(threadgroup_position_in_grid.y) << SWIZZLE_LOG)
            + (int(threadgroup_position_in_grid.x) & 3);
        int tile_n = int(threadgroup_position_in_grid.x) >> SWIZZLE_LOG;
        if (tile_m >= \((m + 63) / 64)) {
            return;
        }
        int row0 = tile_m * BM;
        int col0 = tile_n * BN;

        const device bfloat* A = (const device bfloat*)a + ulong(row0) * \(k);
        const device bfloat* B = (const device bfloat*)b + col0;
        device bfloat* C = (device bfloat*)out + ulong(row0) * \(n) + col0;

        // The lane-to-fragment mapping used by MLX Steel's BaseMMAFrag.
        short qid = short(lane / 4);
        short frag_row = short((qid & 4) + ((lane / 2) % 4));
        short frag_col = short((qid & 2) * 2 + (lane % 2) * 2);

        // Each SIMDgroup owns 64x32 output values: an 8x4 grid of 8x8
        // fragments.  Keeping the two per-lane values as float2 rather than an
        // array of opaque simdgroup_matrix objects is the register layout Steel
        // uses and permits a direct bf16 epilogue.
        float2 acc[8][4];
        H3_UNROLL
        for (short i = 0; i < 8; ++i) {
            H3_UNROLL
            for (short j = 0; j < 4; ++j) {
                acc[i][j] = float2(0.0f);
            }
        }

        struct alignas(2) Read16 { uchar bytes[32]; };

        constexpr int K_ITERS = \(k / 16);
        short valid_m = short(min(int(BM), \(m) - row0));
        for (int kit = 0; kit < K_ITERS; ++kit) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // One 32-byte read per thread, matching Steel's loader geometry.
            // A assigns one row to each thread; B assigns four threads to each
            // row. Padding keeps threadgroup rows off conflicting bank strides.
            {
                int r = int(tid);
                threadgroup Read16* dst =
                    (threadgroup Read16*)(As + r * LDA_TGP);
                if (r < valid_m) {
                    const device Read16* src = (const device Read16*)(
                        A + r * \(k) + kit * BK);
                    *dst = *src;
                } else {
                    for (short z = 0; z < BK; ++z) {
                        As[r * LDA_TGP + z] = bfloat(0.0f);
                    }
                }
            }
            {
                int r = int(tid) / 4, c = (int(tid) % 4) * 16;
                threadgroup Read16* dst =
                    (threadgroup Read16*)(Bs + r * LDB_TGP + c);
                const device Read16* src = (const device Read16*)(
                    B + (kit * BK + r) * \(n) + c);
                *dst = *src;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            H3_UNROLL
            for (short kk = 0; kk < BK; kk += 8) {
                bfloat2 afrag[8];
                bfloat2 bfrag[4];
                simdgroup_barrier(mem_flags::mem_none);
                H3_UNROLL
                for (short i = 0; i < 8; ++i) {
                    afrag[i] = *((threadgroup bfloat2*)(
                        As + (i * 8 + frag_row) * LDA_TGP + kk + frag_col));
                }
                simdgroup_barrier(mem_flags::mem_none);
                H3_UNROLL
                for (short j = 0; j < 4; ++j) {
                    short nbase = short(sgid * 8 + j * 16);
                    bfrag[j] = *((threadgroup bfloat2*)(
                        Bs + (kk + frag_row) * LDB_TGP + nbase + frag_col));
                }
                simdgroup_barrier(mem_flags::mem_none);
                H3_UNROLL
                for (short i = 0; i < 8; ++i) {
                    simdgroup_matrix<bfloat, 8, 8> am;
                    reinterpret_cast<thread bfloat2&>(am.thread_elements()) = afrag[i];
                    H3_UNROLL
                    for (short jj = 0; jj < 4; ++jj) {
                        short j = (i & 1) ? short(3 - jj) : jj;
                        simdgroup_matrix<bfloat, 8, 8> bm;
                        reinterpret_cast<thread bfloat2&>(bm.thread_elements()) =
                            bfrag[j];
                        simdgroup_matrix<float, 8, 8> cm;
                        simdgroup_matrix<float, 8, 8> dm;
                        reinterpret_cast<thread float2&>(cm.thread_elements()) = acc[i][j];
                        simdgroup_multiply_accumulate(dm, am, bm, cm);
                        acc[i][j] =
                            reinterpret_cast<thread float2&>(dm.thread_elements());
                    }
                }
            }
        }

        H3_UNROLL
        for (short i = 0; i < 8; ++i) {
            int r = i * 8 + frag_row;
            if (r < valid_m) {
                H3_UNROLL
                for (short j = 0; j < 4; ++j) {
                    int c = int(sgid) * 8 + j * 16 + frag_col;
                    C[ulong(r) * \(n) + c] = bfloat(acc[i][j][0]);
                    C[ulong(r) * \(n) + c + 1] = bfloat(acc[i][j][1]);
                }
            }
        }
        """

        return MLXFast.metalKernel(
            name: "h3_gemm_bf16_steel_fc1",
            inputNames: ["a", "b"], outputNames: ["out"],
            source: source,
            header: """
                #include <metal_stdlib>
                #include <metal_simdgroup_matrix>
                #define H3_UNROLL _Pragma("clang loop unroll(full)")
                using namespace metal;
                """)
    }

    /// `a` is `[M, K]`, `b` is `[K, N]` — B **already transposed**, as a
    /// production weight would be once at load. Comparing against
    /// `matmul(x, w.T)` on a strided view would price MLX's transpose, not its
    /// GEMM.
    static func custom(_ kernel: MLXFast.MLXFastKernel, _ t: Tiling,
                       _ a: MLXArray, _ b: MLXArray) -> MLXArray {
        kernel([a, b],
               grid: (t.threads * ((n + t.bn - 1) / t.bn), (m + t.bm - 1) / t.bm, 1),
               threadGroup: (t.threads, 1, 1),
               outputShapes: [[m, n]],
               outputDTypes: [.bfloat16])[0]
    }

    static func steelCustom(_ kernel: MLXFast.MLXFastKernel,
                            _ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let threads = 64
        let tilesN = n / 64
        let tilesM = (m + 63) / 64
        let swizzle = 4
        return kernel(
            [a, b],
            grid: (threads * tilesN * swizzle, (tilesM + swizzle - 1) / swizzle, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [.bfloat16])[0]
    }

    static func rate(seconds: Double) -> Double {
        2.0 * Double(m) * Double(k) * Double(n) / seconds / 1e12
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func customGEMM() {
        MLXRandom.seed(7)
        let a = (MLXRandom.normal([Self.m, Self.k]) * 0.02).asType(.bfloat16)
        let b = (MLXRandom.normal([Self.k, Self.n]) * 0.02).asType(.bfloat16)
        let reference = matmul(a, b)
        MLX.eval(reference)
        let mlxSeconds = BenchmarkSupport.medianArrays(rounds: 5) { [matmul(a, b)] }

        // The two axes that matter, crossed: arithmetic intensity (the tile) and
        // register pressure (accumulators per simdgroup). The first two attempts
        // moved both at once, which is why neither told us anything.
        let tilings = [
            Tiling(bm: 64, bn: 64, bk: 32, sgM: 2, sgN: 2),      // attempt 1
            Tiling(bm: 128, bn: 128, bk: 32, sgM: 4, sgN: 2),    // attempt 2
            Tiling(bm: 128, bn: 128, bk: 32, sgM: 4, sgN: 4),    // same tile, half the registers
            Tiling(bm: 128, bn: 64, bk: 32, sgM: 4, sgN: 2),
            Tiling(bm: 128, bn: 128, bk: 16, sgM: 4, sgN: 4),
            Tiling(bm: 256, bn: 128, bk: 16, sgM: 8, sgN: 4),
        ]

        print(String(format: """

            fc1  [%d, %d] x [%d, %d]  bf16
              MLX        %6.1f ms  %5.1f TFLOP/s
              MPSGraph                19.3 TFLOP/s   (§8, the target)

              tiling                acc/sg  FLOP/byte     ms   TFLOP/s   vs MLX  exact
            """, Self.m, Self.k, Self.k, Self.n,
             mlxSeconds * 1000, Self.rate(seconds: mlxSeconds)))

        let steelKernel = Self.steelKernel()
        let steel = Self.steelCustom(steelKernel, a, b)
        MLX.eval(steel)
        let steelExact = MLX.all(steel .== reference).item(Bool.self)
        let steelSeconds = BenchmarkSupport.medianArrays(rounds: 5) {
            [Self.steelCustom(steelKernel, a, b)]
        }
        print(String(format: "  %-20@  %5d  %9.0f  %6.1f    %6.1f    %.2fx  %@",
                     "Steel 64x64x16 1x2sg", 32, 32.0,
                     steelSeconds * 1000, Self.rate(seconds: steelSeconds),
                     Self.rate(seconds: steelSeconds) / Self.rate(seconds: mlxSeconds),
                     steelExact ? "yes" : "NO"))
        #expect(steelExact, "Steel-specialised kernel does not compute the same product")

        if ProcessInfo.processInfo.environment["H3_STEEL_ONLY"] != nil {
            return
        }

        for t in tilings {
            let kernel = Self.kernel(t)
            let mine = Self.custom(kernel, t, a, b)
            MLX.eval(mine)
            let exact = MLX.all(mine .== reference).item(Bool.self)
            let seconds = BenchmarkSupport.medianArrays(rounds: 5) {
                [Self.custom(kernel, t, a, b)]
            }
            print(String(format: "  %-20@  %5d  %9.0f  %6.1f    %6.1f    %.2fx  %@",
                         t.label, t.accumulators, t.intensity, seconds * 1000,
                         Self.rate(seconds: seconds),
                         Self.rate(seconds: seconds) / Self.rate(seconds: mlxSeconds),
                         exact ? "yes" : "NO"))
            #expect(exact, "\(t.label) does not compute the same product")
        }
    }
}
