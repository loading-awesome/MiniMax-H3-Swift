// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import H3Foundation
@testable import H3Modules

/// Does compiling a whole DiT block make it faster, and does it change the math?
///
/// **Measured at the block, not at the operation.** An earlier estimate struck
/// this idea by costing kernel-launch overhead alone — about a thousand
/// dispatches at ~30 µs against a 60 s forward, so 0.05% — and that was the
/// wrong instrument for the question. Compilation also removes intermediate
/// materialisation and gives the optimiser visibility across the attention and
/// MLP boundaries, neither of which a launch count captures. The only honest
/// measurement is the latency of the thing itself.
///
/// The block is the right unit because it is the repeated one: fifty identical
/// blocks per forward, so a per-block saving multiplies by fifty, and it
/// applies to **every full step** including the refresh steps the cross-step
/// cache is required to run in full.
///
///     H3_BIG=1 swift test --filter compiledBlock
///
/// Out of the normal suite: it allocates roughly 1.3 GB of block weights at
/// production width, and a timing assertion that fails under contention teaches
/// people to ignore failures.
@Suite("compiled block", .serialized)
struct CompiledBlockTests {

    static func productionBlock(seed: UInt64 = 11)
        -> (block: DiTBlock, x: MLXArray, tEmb: MLXArray,
            index: ModulationIndex, rope: MLXArray?) {
        let cfg = H3Config()
        let s = 15_731
        let h = cfg.hiddenSize, inner = cfg.innerDim, ffn = cfg.ffnHidden
        MLXRandom.seed(seed)
        func w(_ shape: [Int]) -> MLXArray {
            (MLXRandom.normal(shape) * 0.02).asType(.bfloat16)
        }
        let block = DiTBlock(
            norm1: H3RMSNorm(weight: w([h]), eps: cfg.normEps),
            norm2: H3RMSNorm(weight: w([h]), eps: cfg.normEps),
            attn: AttentionLayer(qkvWeight: w([3 * inner, h]),
                                 outWeight: w([h, inner]),
                                 qNormWeight: w([cfg.headDim]),
                                 kNormWeight: w([cfg.headDim]),
                                 heads: cfg.numHeads, headDim: cfg.headDim,
                                 eps: cfg.qkNormEps),
            mlp: H3MLP(fc1: w([2 * ffn, h]), fc2: w([h, ffn])),
            adaln: AdalnProj(weight: w([cfg.adalnOutFeatures, cfg.timeEmbedDim]),
                             bias: nil, expand: 6, modalities: 3, hidden: h,
                             // fp32 as production runs it. Switching it off
                             // would remove a cast pass that compilation might
                             // otherwise fuse, and measuring a configuration
                             // nobody renders with is how 6B happened.
                             computeFP32: true))
        // Three timestep rows and three modalities, as a real render has.
        let segs = [ModSegment(start: 0, stop: 746, row: 0),
                    ModSegment(start: 746, stop: s, row: 6)]
        // **RoPE included.** Leaving it out understated compilation's case: the
        // split-half rotation is a chain of elementwise multiplies, adds and a
        // concatenate on `[S, heads, headDim]`, applied twice per block, and
        // fusing exactly that chain is the thing being tested. Angles are
        // `[S, 96]` — t|h|w concatenated with itself — so the table is
        // `[1, S, 1, 48, 2, 2]`.
        let rope = H3RoPE.rotationTable(
            angles: MLXRandom.normal([s, 96]).asType(.float32)).asType(.bfloat16)
        return (block, w([s, h]), w([3, cfg.timeEmbedDim]),
                ModulationIndex(segments: segs, tokenCount: s), rope)
    }

    private static func time(_ n: Int, _ body: () -> MLXArray) -> Double {
        MLX.eval(body())                 // warm up: compile, allocate, JIT
        let t0 = Date()
        for _ in 0 ..< n { MLX.eval(body()) }
        return Date().timeIntervalSince(t0) / Double(n)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func compiledBlock() {
        let f = Self.productionBlock()
        let blocks = H3Config().numLayers

        let plain = { f.block(f.x, tEmb: f.tEmb, index: f.index, ropeTable: f.rope) }

        // The weights, modulation rows and RoPE table are captured constants.
        // **`tEmb` is an input, not a capture.** It is derived from sigma and
        // therefore changes at every sampler step, so compiling it in as a
        // constant would measure a configuration that cannot ship — which is
        // exactly how 6B came to report a number for something nobody would
        // run. The weights, the modulation rows and the RoPE table are genuinely
        // fixed for a render and stay captured.
        let compiled = MLX.compile { (arrays: [MLXArray]) -> [MLXArray] in
            [f.block(arrays[0], tEmb: arrays[1], index: f.index, ropeTable: f.rope)]
        }
        let compiledCall = { compiled([f.x, f.tEmb])[0] }

        // **Math first.** A speed-up that changes the result is not a speed-up,
        // and compilation reassociating a reduction would show up here rather
        // than in a render three hours later.
        let a = plain(), b = compiledCall()
        MLX.eval(a, b)
        let d = (a.asType(.float32) - b.asType(.float32))
        let rel = MLX.sqrt(MLX.mean(d * d)).item(Float.self)
            / MLX.sqrt(MLX.mean(a.asType(.float32) * a.asType(.float32))).item(Float.self)
        let identical = MLX.all(a .== b).item(Bool.self)
        print("\n  math: relative RMS \(rel), bit-identical: \(identical)")

        let measured = BenchmarkSupport.interleaved(first: plain, second: compiledCall)
        let plainMs = measured.first * 1000
        let compiledMs = measured.second * 1000
        print(String(format: "  per block   plain %.1f ms   compiled %.1f ms   %.3fx",
                     plainMs, compiledMs, plainMs / compiledMs))
        print(String(format: "  x%d blocks  plain %.2f s    compiled %.2f s    "
                     + "saving %.2f s per full step",
                     blocks, plainMs * Double(blocks) / 1000,
                     compiledMs * Double(blocks) / 1000,
                     (plainMs - compiledMs) * Double(blocks) / 1000))
        // A full step measured 60.0 s in the controls; 11 of 20 steps run full
        // at the shipping cache setting, so a per-step saving is worth 11x that
        // over a render.
        let perStepSaving = (plainMs - compiledMs) * Double(blocks) / 1000
        print(String(format: "  against a 60.0 s full step: %.2f%% — and %.1f s "
                     + "over the 11 full steps of a 20-step render\n",
                     100 * perStepSaving / 60.0, perStepSaving * 11))

        #expect(rel < 1e-3, "compilation must not change the block's math")
    }

    /// What does the AdaLN projection alone cost, per block?
    ///
    /// **Found by accident, and it revives an item that was struck.** Compiling
    /// a block with `tEmb` captured as a constant measured 2.09% of a full
    /// step; an initial non-interleaved run with `tEmb` as a real input — the
    /// only shippable form — measured 0.28%. That unstable comparison prompted
    /// the direct measurement below; the final compiler decision uses the
    /// interleaved benchmark above instead.
    ///
    /// It cannot be folded at compile time, but it can be precomputed: `tEmb`
    /// takes exactly `steps` distinct values in a render, all of them known
    /// before the first block runs. That is roadmap item 3, which an earlier
    /// estimate struck at 0.05% by counting its FLOPs — 0.008% of the forward —
    /// while ignoring that it reads a 520 MB weight matrix and upcasts it to
    /// fp32 on every call. FLOPs were the wrong unit twice.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func adalnProjectionCost() {
        let cfg = H3Config()
        MLXRandom.seed(11)
        let tEmb = (MLXRandom.normal([3, cfg.timeEmbedDim]) * 0.02).asType(.bfloat16)
        for fp32 in [true, false] {
            let proj = AdalnProj(
                weight: (MLXRandom.normal([cfg.adalnOutFeatures, cfg.timeEmbedDim]) * 0.02)
                    .asType(.bfloat16),
                bias: nil, expand: 6, modalities: 3, hidden: cfg.hiddenSize,
                computeFP32: fp32)
            let ms = Self.time(5) { proj(tEmb)[0] } * 1000
            print(String(format: "  adaln projection (fp32 %@)  %.1f ms/block  "
                         + "%.1f s per forward  %.2f%% of a 60.0 s step",
                         (fp32 ? "on" : "off") as NSString, ms,
                         ms * Double(cfg.numLayers) / 1000,
                         100 * ms * Double(cfg.numLayers) / 1000 / 60.0))
        }
        print("")
    }

    /// Does projecting the complete known schedule once per block outperform
    /// twenty small projections, after paying for cache-skipped blocks and the
    /// persistent modulation tables? This is the shippable unit for roadmap
    /// item 3; a single-call microbenchmark cannot answer it.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func adalnScheduleBatching() {
        let cfg = H3Config()
        let steps = 20
        MLXRandom.seed(11)
        let projection = AdalnProj(
            weight: (MLXRandom.normal([cfg.adalnOutFeatures, cfg.timeEmbedDim]) * 0.02)
                .asType(.bfloat16),
            bias: nil, expand: 6, modalities: 3, hidden: cfg.hiddenSize,
            computeFP32: true)

        // Plain generation has one deduplicated timestep at sigma=1 and two
        // thereafter. A paired visual+audio reference adds the pinned .999 and
        // 1.0 rows: three at sigma=1, then four. Measure both supported shapes.
        let scenarios: [(String, [Int])] = [
            ("t2va", [1] + Array(repeating: 2, count: steps - 1)),
            ("ref2va", [3] + Array(repeating: 4, count: steps - 1))
        ]
        for (label, rowsPerStep) in scenarios {
            let perStep = rowsPerStep.map { rows in
                (MLXRandom.normal([rows, cfg.timeEmbedDim]) * 0.009).asType(.bfloat16)
            }
            let wholeSchedule = concatenated(perStep, axis: 0)
            MLX.eval(wholeSchedule)

            let sequential = { perStep.flatMap { projection($0) } }
            let batched = { projection(wholeSchedule) }

            // Preserve the row order consumed by ModulationIndex: timestep,
            // then modality, with the six expansion tables kept separate.
            let byStep = perStep.map { projection($0) }
            let expected = (0 ..< 6).map { expansion in
                concatenated(byStep.map { $0[expansion] }, axis: 0)
            }
            let actual = batched()
            MLX.eval(expected)
            MLX.eval(actual)
            var worstRel: Float = 0
            var identical = true
            for (a, b) in zip(expected, actual) {
                let delta = a.asType(.float32) - b.asType(.float32)
                let rel = MLX.sqrt(MLX.mean(delta * delta)).item(Float.self)
                    / MLX.sqrt(MLX.mean(a.asType(.float32) * a.asType(.float32)))
                        .item(Float.self)
                worstRel = max(worstRel, rel)
                identical = identical && MLX.all(a .== b).item(Bool.self)
            }

            let measured = BenchmarkSupport.interleavedArrays(first: sequential,
                                                               second: batched)
            let sequentialMs = measured.first * 1000
            let batchedMs = measured.second * 1000
            let denseSaving = (sequentialMs - batchedMs) * Double(cfg.numLayers) / 1000
            let currentCalls = steps + (cfg.numLayers - 1) * 10 // cap 5: ten full
            let currentCached = sequentialMs / Double(steps) * Double(currentCalls) / 1000
            let batchedCached = batchedMs * Double(cfg.numLayers) / 1000
            let cachedSaving = currentCached - batchedCached
            let tableMiB = Double(wholeSchedule.dim(0) * cfg.adalnOutFeatures * 2
                                  * cfg.numLayers) / 1_048_576

            print(String(format: "\n  AdaLN %@ schedule (%d steps, %d input rows)",
                         label as NSString, steps, wholeSchedule.dim(0)))
            print(String(format: "  per block   sequential %.1f ms   batched %.1f ms   %.2fx",
                         sequentialMs, batchedMs, sequentialMs / batchedMs))
            print(String(format: "  dense render saving %.2f s; cap-5 estimate %.2f s",
                         denseSaving, cachedSaving))
            print(String(format: "  persistent tables %.0f MiB; rel RMS %.3e; bit-identical %@\n",
                         tableMiB, worstRel, (identical ? "yes" : "no") as NSString))

            // A larger M dimension selects a different GEMM accumulation path.
            // A qualified speed result would therefore require a render gate.
            #expect(worstRel < 1e-4,
                    "schedule batching exceeded the block-level error bound")
        }
    }

    /// Does a larger compiled subgraph do better than one block at a time?
    ///
    /// The block boundary is a real barrier: `x1 = x + gate * attn(...)` ends
    /// one compiled region and begins the next, so the residual add and the
    /// following RMSNorm cannot be fused across it. Chaining two blocks inside
    /// one compiled function removes one such boundary. If the per-block gain
    /// rises, the boundary is costing something and the right unit is the
    /// stack rather than the block; if it does not, the block is already the
    /// largest useful region and there is nothing more to take.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func compiledPair() {
        let a = Self.productionBlock(seed: 11)
        let b = Self.productionBlock(seed: 12)

        let plain = {
            let h = a.block(a.x, tEmb: a.tEmb, index: a.index, ropeTable: a.rope)
            return b.block(h, tEmb: a.tEmb, index: b.index, ropeTable: b.rope)
        }
        let compiled = MLX.compile { (arrays: [MLXArray]) -> [MLXArray] in
            let h = a.block(arrays[0], tEmb: arrays[1], index: a.index, ropeTable: a.rope)
            return [b.block(h, tEmb: arrays[1], index: b.index, ropeTable: b.rope)]
        }
        let compiledCall = { compiled([a.x, a.tEmb])[0] }

        let expected = plain(), actual = compiledCall()
        MLX.eval(expected, actual)
        let delta = expected.asType(.float32) - actual.asType(.float32)
        let rel = MLX.sqrt(MLX.mean(delta * delta)).item(Float.self)
            / MLX.sqrt(MLX.mean(expected.asType(.float32)
                * expected.asType(.float32))).item(Float.self)

        let measured = BenchmarkSupport.interleaved(first: plain, second: compiledCall)
        let plainMs = measured.first * 1000
        let compiledMs = measured.second * 1000
        print(String(format: "\n  two blocks  plain %.1f ms   compiled %.1f ms   %.3fx",
                     plainMs, compiledMs, plainMs / compiledMs))
        print(String(format: "  per block   plain %.1f ms   compiled %.1f ms   "
                     + "%.2f%% of a full step",
                     plainMs / 2, compiledMs / 2,
                     100 * (plainMs - compiledMs) / 2 * 50 / 1000 / 60.0))
        print("  math: relative RMS \(rel)\n")
        #expect(rel < 1e-3, "two-block compilation must stay inside the block oracle")
    }
}
