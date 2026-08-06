// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import MLXRandom
@testable import H3Attention

/// What the kernel has to beat, measured on this machine at the verified shape.
///
/// Not a correctness test and deliberately not asserting a time: it prints the
/// budget so the sparse path can be judged against dense attention as actually
/// implemented here, rather than against the CUDA figures in docs/SOL_ATTN.md,
/// which were measured on an RTX PRO 6000 with a Triton kernel and do not
/// transfer.
///
/// Run with:
///     swift test -c release --filter attentionBudgetAtProductionShape
@Suite("Sol-Attn budget", .serialized)
struct SolAttnBudgetTests {

    private func time(_ label: String, _ n: Int = 5, _ body: () -> MLXArray) -> Double {
        _ = body()                     // warm the JIT and the allocator
        MLX.eval(body())
        let t0 = Date()
        for _ in 0 ..< n { MLX.eval(body()) }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(n)
        print(String(format: "  %-28s %8.2f ms", (label as NSString).utf8String!, ms))
        return ms
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BUDGET"] != nil))
    func attentionBudgetAtProductionShape() {
        // The verified shape: 864x480x124 packs to S = 15,731 with 56 heads.
        let heads = 56, s = 15_731, d = 128
        MLXRandom.seed(0)
        let q = (MLXRandom.normal([heads, s, d]) * 0.5).asType(.bfloat16)
        let k = (MLXRandom.normal([heads, s, d]) * 0.5).asType(.bfloat16)
        let v = (MLXRandom.normal([heads, s, d]) * 0.5).asType(.bfloat16)
        let scale = 1.0 / Float(d).squareRoot()
        MLX.eval(q, k, v)

        print("\nSol-Attn budget — H=\(heads) S=\(s) D=\(d), bf16")

        let denseMs = time("dense sdpa") {
            MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
                values: v.expandedDimensions(axis: 0), scale: scale, mask: nil)
        }

        let config = SolAttnConfig()
        let routeMs = time("routing (pool+proxy+tau)") {
            let pooling = SolAttnRouting.pool(keys: k, values: v, queryCount: s,
                                              blockSize: config.blockSize)
            let pq = SolAttnRouting.poolQueries(q, blockSize: config.blockSize)
            return SolAttnRouting.select(pooledQueries: pq, pooling: pooling,
                                         beta: config.beta, sinkKeyBlocks: 16)
                .asType(.float32)
        }

        // What fraction of blocks the router keeps decides the kernel's ceiling.
        let pooling = SolAttnRouting.pool(keys: k, values: v, queryCount: s,
                                          blockSize: config.blockSize)
        let pq = SolAttnRouting.poolQueries(q, blockSize: config.blockSize)
        let sel = SolAttnRouting.select(pooledQueries: pq, pooling: pooling,
                                        beta: config.beta, sinkKeyBlocks: 16)
        let density = MLX.mean(sel.asType(.float32)).item(Float.self)

        let ceiling = denseMs / (Double(density) * denseMs + routeMs)
        print(String(format: "  density %.3f  routing is %.1f%% of dense",
                     density, routeMs / denseMs * 100))
        print(String(format: "  ceiling if the kernel were free at that density: %.2fx", ceiling))

        // The whole sparse path, routing included, which is what actually
        // replaces the dense call.
        let solMs = time("sol-attn (routing+kernel)") {
            SolAttnMetalKernel.attend(queries: q, keys: k, values: v, scale: scale,
                                      config: config, videoSpan: 1_000 ..< s)!
        }
        print(String(format: "  end-to-end on the attention call: %.2fx", denseMs / solMs))
        print(String(format: "  kernel efficiency against the ceiling: %.0f%%",
                     (denseMs / solMs) / ceiling * 100))
        print("")
    }

    /// Routing block size, swept until something gets worse.
    ///
    /// Three quantities move independently and only one of them is a fact about
    /// this machine:
    ///
    ///  * **speed** saturates for hardware reasons — fewer, longer contiguous
    ///    runs of exact keys, against a coarser selection that eventually keeps
    ///    blocks it did not need. Data-independent, and the honest reason to run
    ///    this sweep.
    ///  * **density** is `1 - Phi(beta)` plus the sink, and the Gaussian tail is
    ///    a property of the score distribution rather than of how many blocks it
    ///    is cut into, so it should barely move.
    ///  * **quality** degrades once a block is wider than the input's own
    ///    locality — and on synthetic input *the locality scale is something I
    ///    chose*. Where the quality knee lands here is therefore an artifact of
    ///    the generator, not a fact about H3. The shape of the curve transfers;
    ///    the position of the knee does not.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BUDGET"] != nil))
    func blockSizeSweptToDegradation() {
        for segment in [256, 1_024] { sweep(locality: segment) }
    }

    /// One sweep at a chosen locality scale. Run at two scales, because a knee
    /// that sits at whatever `segment` was set to is a statement about the
    /// generator; a knee that *moves with* it is the structural result.
    private func sweep(locality segment: Int) {
        let heads = 56, s = 15_731, d = 128
        let clusters = 12
        MLXRandom.seed(4)
        let centres = MLXRandom.normal([clusters, d]) * 0.8
        let ids = MLXArray((0 ..< s).map { Int32(($0 / segment) % clusters) })
        let q = (centres[ids].reshaped([1, s, d])
                 + MLXRandom.normal([heads, s, d]) * 0.25).asType(.bfloat16)
        let k = (centres[ids].reshaped([1, s, d])
                 + MLXRandom.normal([heads, s, d]) * 0.25).asType(.bfloat16)
        let v = MLXRandom.normal([heads, s, d]).asType(.bfloat16)
        let scale = 1.0 / Float(d).squareRoot()
        MLX.eval(q, k, v)

        let want = MLXFast.scaledDotProductAttention(
            queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
            values: v.expandedDimensions(axis: 0), scale: scale, mask: nil).squeezed(axis: 0)
        MLX.eval(want)
        let denseMs = time("dense sdpa") {
            MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
                values: v.expandedDimensions(axis: 0), scale: scale, mask: nil)
        }

        print("\n  block  density   speedup   relRMS vs dense   (locality scale \(segment))")
        for bs in [64, 128, 256, 512, 1_024, 2_048, 4_096] {
            var c = SolAttnConfig(); c.blockSize = bs
            guard let got = SolAttnMetalKernel.attend(queries: q, keys: k, values: v,
                                                      scale: scale, config: c,
                                                      videoSpan: 1_000 ..< s) else {
                print("  \(bs)  declined"); continue
            }
            MLX.eval(got)
            let density = SolAttnReference.density(queries: q, keys: k, values: v,
                                                   config: c, videoSpan: 1_000 ..< s)
            let diff = got.asType(.float32) - want.asType(.float32)
            let err = MLX.sqrt(MLX.mean(diff * diff)).item(Float.self)
                    / MLX.sqrt(MLX.mean(want.asType(.float32) * want.asType(.float32))).item(Float.self)
            let ms = time("  blockSize \(bs)", 3) {
                SolAttnMetalKernel.attend(queries: q, keys: k, values: v, scale: scale,
                                          config: c, videoSpan: 1_000 ..< s)!
            }
            print(String(format: "  %5d  %.3f     %.2fx      %.4f", bs, density,
                         denseMs / ms, err))
        }

        print("")
    }
}
