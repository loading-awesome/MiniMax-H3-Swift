// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import H3Foundation
@testable import H3Modules

/// What MLX can actually do on this machine, at the shapes this model uses.
///
/// **This is the measurement that decides whether the remaining roadmap items
/// exist**, and it costs seconds. The production forward achieves 16.0 TFLOP/s
/// — 961 TFLOP in 60.0 s, measured from the control sweep. Nobody has measured
/// what MLX reaches on an isolated GEMM of the same shape, so nobody knows
/// whether that 16 is MLX's ceiling or the model's overhead.
///
/// The two answers lead to different work:
///
///  * **Isolated GEMM much faster than 16.** The gap is in shapes, layout,
///    dispatch and everything between the matmuls — worth attacking, and the
///    only untried item with a multiple in it.
///  * **Isolated GEMM also near 16.** MLX's GEMM is the ceiling. Layout work
///    is dead on arrival too, and only fewer bits or a hand-written kernel can
///    move anything.
///
/// Doing this *after* building something is how 6B happened: a kernel whose
/// ceiling was about 1%, discovered afterwards. The whole of `docs/PERF_ROADMAP.md`
/// now leads with the arithmetic for that reason.
///
///     H3_BIG=1 swift test --filter gemmCeiling
///
/// Never part of the normal suite: it is a benchmark, it is meaningless under
/// contention, and a timing assertion that fails when another process is busy
/// is a test that teaches people to ignore failures.
@Suite("GEMM ceiling", .serialized)
struct GEMMCeilingTests {

    private func timeIt(_ n: Int = 5, _ body: () -> MLXArray) -> Double {
        MLX.eval(body())                       // warm up: compile, allocate
        let t0 = Date()
        for _ in 0 ..< n { MLX.eval(body()) }
        return Date().timeIntervalSince(t0) / Double(n)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func gemmCeiling() {
        let cfg = H3Config()
        let s = 15_731                          // 864x480x124, the control shape
        let h = cfg.hiddenSize                  // 5376
        let inner = cfg.innerDim                // 7168
        let ffn = cfg.ffnHidden                 // 14336

        // The four GEMMs a block actually runs, at production width, plus a
        // large square one as a clean upper bound on what the hardware gives.
        let shapes: [(String, Int, Int, Int)] = [
            ("qkv      [S,H]x[H,3I]", s, h, 3 * inner),
            ("attn out [S,I]x[I,H]", s, inner, h),
            ("mlp fc1  [S,H]x[H,2F]", s, h, 2 * ffn),
            ("mlp fc2  [S,F]x[F,H]", s, ffn, h),
            ("square   8192^3", 8_192, 8_192, 8_192)
        ]

        print("\n  the model achieves 16.0 TFLOP/s overall (961 TFLOP in 60.0 s)\n")
        print("  \(("shape" as NSString).padding(toLength: 24, withPad: " ", startingAt: 0))"
              + "     ms    TFLOP/s")
        for (name, m, k, n) in shapes {
            let a = MLXRandom.normal([m, k]).asType(.bfloat16)
            let b = MLXRandom.normal([k, n]).asType(.bfloat16)
            MLX.eval(a, b)
            let ms = timeIt { MLX.matmul(a, b) }
            let flops = 2.0 * Double(m) * Double(k) * Double(n)
            print(String(format: "  %@ %6.1f %10.1f",
                         (name as NSString).padding(toLength: 24, withPad: " ",
                                                    startingAt: 0), ms * 1000,
                         flops / ms / 1e12))
        }

        // Attention is 36.9% of the forward's FLOPs and is not a plain GEMM,
        // so it gets measured on its own terms rather than inferred.
        let heads = cfg.numHeads, d = cfg.headDim
        let q = MLXRandom.normal([1, heads, s, d]).asType(.bfloat16)
        let k = MLXRandom.normal([1, heads, s, d]).asType(.bfloat16)
        let v = MLXRandom.normal([1, heads, s, d]).asType(.bfloat16)
        MLX.eval(q, k, v)
        let scale = 1.0 / Float(d).squareRoot()
        let ms = timeIt(3) {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                              scale: scale, mask: nil)
        }
        let attnFlops = 4.0 * Double(s) * Double(s) * Double(heads * d)
        print(String(format: "\n  attention  S=%d H=%d D=%d   %6.1f ms  %6.1f TFLOP/s",
                     s, heads, d, ms * 1000, attnFlops / ms / 1e12))
        print(String(format: "  one block's share of a 60.0 s forward: %.1f ms\n",
                     60_000.0 / 50.0))
    }
}
