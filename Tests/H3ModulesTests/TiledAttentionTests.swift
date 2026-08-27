// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXRandom
import H3Foundation
@testable import H3Modules

/// Query tiling: the same attention, computed a slice of the queries at a time.
///
/// Softmax runs over the key axis, independently for every query row, so
/// splitting the queries and keeping the whole KV is the one decomposition of
/// attention that is arithmetically free — no rescaling, no running maximum,
/// nothing to recombine. Each tile is a smaller attention with the same keys.
///
/// **The entire plan rests on that being bit-identical and not merely close.**
/// A block's arithmetic is pinned by contract, and a tiled attention that
/// reassociates its accumulation is a different sample, not a faster one. It is
/// asserted here rather than assumed, because `MLXFast.scaledDotProductAttention`
/// is free to pick a different kernel or a different tiling when the query
/// count changes, and nothing in its contract promises it will not.
///
/// What tiling buys is a **seam inside the attention window**. Everything after
/// attention in a block — the output projection, the residual and gate, the
/// second norm, the MLP — is row-wise, so once tile `i-1` has attended, its
/// whole post-attention chain can run on the engine while the GPU attends tile
/// `i`. Without tiling the engine has nothing to do for the 37.5% of a block
/// that is attention, because `out` depends on all of it.
///
///     H3_BIG=1 swift test --filter tiledAttention
@Suite("query-tiled attention", .serialized)
struct TiledAttentionTests {

    static let s = 15_731
    static let heads = 56
    static let headDim = 128

    /// Dense attention over a query slice, with the keys and values whole.
    static func tiled(q: MLXArray, k: MLXArray, v: MLXArray, tiles: Int) -> MLXArray {
        let rows = q.dim(0)
        let span = (rows + tiles - 1) / tiles
        var parts: [MLXArray] = []
        parts.reserveCapacity(tiles)
        var start = 0
        while start < rows {
            let stop = min(start + span, rows)
            parts.append(AttentionLayer.sdpa(q: q[start ..< stop], k: k, v: v,
                                             headDim: headDim))
            start = stop
        }
        return concatenated(parts, axis: 0)
    }

    /// What does one routed projection cost at tile size, against the same
    /// work done whole?
    ///
    /// Tiling multiplies the number of engine round-trips by T. Each one
    /// converts and transposes an activation, blocks on `eval`, memcpies into
    /// an IOSurface, waits on a semaphore, adopts two output surfaces, rescales
    /// them and concatenates three pieces. None of that scales down with the
    /// tile — several parts are fixed — so if `T` tiles cost materially more
    /// than one whole call, the window tiling opens is being spent on the tax
    /// of opening it.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func tilingTax() {
        let k = 5376, n = 28_672            // fc1, the biggest routed projection
        let rows = Self.s
        ANELinearBackend.postShare = 0.45
        let weight = (MLXRandom.normal([n, k]) * 0.02).asType(.bfloat16)
        MLX.eval(weight)

        func time(_ body: () -> Void) -> Double {
            body()
            var v: [Double] = []
            for _ in 0 ..< 5 { let t = Date(); body(); v.append(Date().timeIntervalSince(t)) }
            return v.sorted()[2] * 1000
        }

        print("\n  fc1 [S,5376]x[5376,28672], engine share 0.45\n")
        print("  tiles   rows/tile      total ms     per call     vs whole")
        var whole = 0.0
        for tiles in [1, 2, 4, 8, 16] {
            let spans = QueryTiling.spans(count: rows, tiles: tiles)
            let xs = spans.map { span in
                MLX.contiguous((MLXRandom.normal([span.count, k]) * 0.05).asType(.bfloat16))
            }
            MLX.eval(xs)
            let ms = time {
                var out: [MLXArray] = []
                for x in xs {
                    out.append(ANELinearBackend.project(x: x, weight: weight, label: "fc1"))
                }
                MLX.eval(out)
            }
            if tiles == 1 { whole = ms }
            print(String(format: "  %5d   %9d   %10.1f   %10.1f     %6.3fx",
                         tiles, spans[0].count, ms, ms / Double(spans.count), ms / whole))
        }
        print("")
    }

    /// **The gate.** One production block, tiled, against the block that ships.
    ///
    /// One configuration per process, and the baseline re-measured inside it.
    /// Sweeping T and the share in a single run drifted the *untiled* number by
    /// 10% between runs — every share compiles new engine programs and every
    /// session holds surfaces, and by the end of a sweep the engine is carrying
    /// a dozen of them. A benchmark whose control moves is not a benchmark.
    ///
    ///     H3_ANE=experimental H3_BIG=1 H3_ANE_TILES_T=4 H3_ANE_SHARE_POST=0.45 \
    ///       swift test --filter tiledBlock
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func tiledBlock() {
        let f = CompiledBlockTests.productionBlock()
        let tiles = QueryTiling.tiles
        let post = ANELinearBackend.postShare
        print("\n  ANE \(ANELinearBackend.isEnabled ? "ON" : "off")   T=\(tiles)"
              + "   qkv share \(ANELinearBackend.share)   post share \(post)")

        // **The share does not change inside a process.** Flipping it between
        // interleaved samples re-plans the shards, which re-uploads both weight
        // surfaces — a transpose and a memcpy of the whole weight — on every
        // sample, and the control drifted 25% because of it. So the untiled arm
        // runs at the same share as the tiled one, and the shipping baseline is
        // its own run with `H3_ANE_TILES_T=1 H3_ANE_SHARE_POST=0.286`.
        let shipping = { () -> [MLXArray] in
            [f.block(f.x, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)]
        }
        let tiled = { () -> [MLXArray] in
            [QueryTiling.block(f.block, f.x, tEmb: f.tEmb, index: f.index,
                               ropeTable: f.rope, tiles: tiles)]
        }

        // **Correctness is against the same share, not against shipping.**
        // Moving columns between the engine and the GPU changes the arithmetic
        // by design — fp16 with a wide accumulator against bf16 — so a tiled
        // block at 0.45 is not supposed to match an untiled one at 0.286. What
        // tiling must not change is the block at its own share.
        let sameShare = shipping()
        let b = tiled()
        MLX.eval(sameShare, b)
        #expect(MLX.abs(sameShare[0] - b[0]).max().item(Float.self) == 0,
                "T=\(tiles) at post=\(post) changed the block's arithmetic")

        let m = BenchmarkSupport.interleavedArrays(rounds: 5, first: shipping, second: tiled)
        let shippingMs = m.first * 1000, tiledMs = m.second * 1000
        print(String(format: "  untiled, same share        %8.1f ms", shippingMs))
        print(String(format: "  tiled T=%d post=%.3f        %8.1f ms", tiles, post, tiledMs))
        print(String(format: "  gain                       %8.3fx   (gate: 1011 ms)\n",
                     shippingMs / tiledMs))
    }

    /// **The gate, with the control measured twice.**
    ///
    /// The shipping route is untiled at 0.286. The candidate is the tiled,
    /// native-I/O route at whatever T and post share are configured. Changing
    /// the share re-plans the shards and re-uploads both weight surfaces, so
    /// these cannot be interleaved sample by sample; instead the control runs
    /// before *and* after the candidate, and if the two controls disagree the
    /// run is drifting and the comparison is void.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func tileGate() {
        let f = CompiledBlockTests.productionBlock()
        let tiles = QueryTiling.tiles
        let post = ANELinearBackend.postShare

        func median(_ body: () -> [MLXArray]) -> Double {
            MLX.eval(body())
            var v: [Double] = []
            for _ in 0 ..< 5 {
                let t = Date(); MLX.eval(body()); v.append(Date().timeIntervalSince(t))
            }
            return v.sorted()[2] * 1000
        }
        func control() -> Double {
            ANELinearBackend.nativeIOEnabled = false
            ANELinearBackend.postShare = 0.286
            return median { [f.block(f.x, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)] }
        }

        let before = control()
        ANELinearBackend.nativeIOEnabled = true
        ANELinearBackend.postShare = post
        let candidate = median {
            [QueryTiling.block(f.block, f.x, tEmb: f.tEmb, index: f.index,
                               ropeTable: f.rope, tiles: tiles)]
        }
        let after = control()

        let drift = abs(before - after) / min(before, after)
        print(String(format: """

          shipping control   %8.1f ms   (re-measured %8.1f ms, drift %.1f%%)
          tiled T=%d post=%.3f native   %8.1f ms
          against the gate   %8.1f ms   %@

        """, before, after, 100 * drift, tiles, post, candidate, 1011.0,
             candidate < 1011 ? "CLEARS" : "MISSES" as NSString))
        #expect(drift < 0.03, "the control moved \(100 * drift)%; this comparison is void")
    }

    /// Does the native Metal pack and IOSurface merge actually buy anything?
    ///
    /// It replaces a bf16->fp16 convert, a transpose, a CPU-visible `asData`
    /// and a memcpy with one kernel writing straight into the engine's
    /// activation surface, and replaces the three-way MLX join with a kernel
    /// reading both output surfaces. Both seams are per routed projection, so
    /// tiling multiplies them by T — which is exactly the argument for building
    /// them, and exactly why it has to be measured on the untiled route too.
    ///
    /// Interleaved in one process, because the effect is smaller than the
    /// cross-process drift.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func nativeSeam() {
        let f = CompiledBlockTests.productionBlock()
        let tiles = QueryTiling.tiles
        let post = ANELinearBackend.postShare
        print("\n  ANE \(ANELinearBackend.isEnabled ? "ON" : "off")   T=\(tiles)"
              + "   qkv \(ANELinearBackend.share)   post \(post)")

        func untiled() -> [MLXArray] {
            [f.block(f.x, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)]
        }
        func tiled() -> [MLXArray] {
            [QueryTiling.block(f.block, f.x, tEmb: f.tEmb, index: f.index,
                               ropeTable: f.rope, tiles: tiles)]
        }
        func with(_ native: Bool, _ body: () -> [MLXArray]) -> [MLXArray] {
            ANELinearBackend.nativeIOEnabled = native
            defer { ANELinearBackend.nativeIOEnabled = false }
            let out = body()
            MLX.eval(out)
            return out
        }

        // The seam must not move the numbers. It changes where bytes are
        // converted, not what is computed.
        let cpu = with(false, untiled), native = with(true, untiled)
        #expect(MLX.abs(cpu[0] - native[0]).max().item(Float.self) == 0,
                "the native pack/merge seam changed the block's arithmetic")

        for (name, body) in [("untiled", untiled), ("tiled T=\(tiles)", tiled)] {
            let m = BenchmarkSupport.interleavedArrays(
                rounds: 5,
                first: { with(false, body) }, second: { with(true, body) })
            let cpuMs = m.first * 1000, nativeMs = m.second * 1000
            print(String(format: "  %-14@ CPU seam %8.1f ms   native %8.1f ms   %6.3fx",
                         name as NSString, cpuMs, nativeMs, cpuMs / nativeMs))
        }
        print("")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func tiledAttention() {
        MLXRandom.seed(11)
        let shape = [Self.s, Self.heads, Self.headDim]
        let q = (MLXRandom.normal(shape) * 0.05).asType(.bfloat16)
        let k = (MLXRandom.normal(shape) * 0.05).asType(.bfloat16)
        let v = (MLXRandom.normal(shape) * 0.05).asType(.bfloat16)
        MLX.eval(q, k, v)

        let dense = AttentionLayer.sdpa(q: q, k: k, v: v, headDim: Self.headDim)
        MLX.eval(dense)

        func time(_ body: () -> MLXArray) -> Double {
            MLX.eval(body())
            var samples: [Double] = []
            for _ in 0 ..< 5 {
                let t = Date()
                MLX.eval(body())
                samples.append(Date().timeIntervalSince(t))
            }
            return samples.sorted()[2] * 1000
        }

        let denseMs = time { AttentionLayer.sdpa(q: q, k: k, v: v, headDim: Self.headDim) }
        print(String(format: "\n  dense  S=%d heads=%d headDim=%d   %.1f ms\n",
                     Self.s, Self.heads, Self.headDim, denseMs))
        print("  tiles   rows/tile      ms    vs dense   max |diff|   bit-identical")

        for tiles in [2, 4, 8, 16] {
            let out = Self.tiled(q: q, k: k, v: v, tiles: tiles)
            MLX.eval(out)
            let diff = MLX.abs(out.asType(.float32) - dense.asType(.float32))
                .max().item(Float.self)
            let identical = MLX.all(out .== dense).item(Bool.self)
            let ms = time { Self.tiled(q: q, k: k, v: v, tiles: tiles) }
            print(String(format: "  %5d   %9d  %6.1f    %6.3fx   %10.3e   %@",
                         tiles, (Self.s + tiles - 1) / tiles, ms, denseMs / ms, diff,
                         identical ? "yes" : "NO"))
            // Tiling must not cost the block more than it can win back. The
            // engine has about 390 ms of attention to hide in; a tiling that
            // makes attention itself 10% slower has spent the window it opened.
            #expect(ms < denseMs * 1.10,
                    "query tiling at T=\\(tiles) made attention materially slower")
            #expect(identical,
                    "query tiling at T=\\(tiles) changed the arithmetic")
        }
        print("")
    }
}
