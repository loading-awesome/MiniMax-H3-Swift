// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import MLXRandom
import H3Foundation
@testable import H3Attention

/// Relative RMS, the quantity the whole equivalence-class contract is stated in.
private func relRMS(_ a: MLXArray, _ b: MLXArray) -> Float {
    let d = (a.asType(.float32) - b.asType(.float32))
    let num = MLX.sqrt(MLX.mean(d * d)).item(Float.self)
    let den = MLX.sqrt(MLX.mean(b.asType(.float32) * b.asType(.float32))).item(Float.self)
    return num / den
}

private func dense(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
    MLXFast.scaledDotProductAttention(
        queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
        values: v.expandedDimensions(axis: 0), scale: scale, mask: nil).squeezed(axis: 0)
}

/// `[H, S, D]` with block structure, because unstructured input is the one case
/// where every sparse method is worthless and measuring against it says nothing.
///
/// §8 recorded rel_rms 0.17 at beta = 0 on Gaussian q/k/v — not because the
/// method is broken but because attention with no dominant blocks has nothing
/// for a sparse method to keep.
///
/// **The structure has to be block-*contiguous*, and getting that wrong is
/// instructive.** A first version of this generator drew each row's cluster
/// independently, which scatters every cluster uniformly across every block: a
/// pooled key then averages all eight centres to roughly the origin, every
/// block looks identical to the router, and the rank-1 correction stands in for
/// keys whose true scores it misses by orders of magnitude. It measured 2.67.
/// That is the honest answer for input with no block structure, and it is not
/// what video tokens look like — neighbouring latents are neighbouring pixels.
///
/// Cluster identity therefore runs in contiguous segments, deliberately not
/// aligned to `blockSize`, so blocks straddle boundaries the way real ones do.
///
/// This remains *synthetic* structure chosen to be tractable. It is enough to
/// show the method works and nowhere near enough to state an equivalence class;
/// that needs q/k/v captured from a real render.
private func structured(heads: Int, s: Int, d: Int, clusters: Int, seed: UInt64)
    -> (MLXArray, MLXArray, MLXArray) {
    MLXRandom.seed(seed)
    // Scaled so that scale * q.k lands at single digits, as attention logits do.
    // At 2.0 the softmax degenerates to an argmax over exp(32) and every method
    // that is not exact looks catastrophic.
    let centres = MLXRandom.normal([clusters, d]) * 0.8
    let segment = 96
    let ids = MLXArray((0 ..< s).map { Int32(($0 / segment) % clusters) })
    let k = centres[ids].reshaped([1, s, d]) + MLXRandom.normal([heads, s, d]) * 0.25
    let q = centres[ids].reshaped([1, s, d]) + MLXRandom.normal([heads, s, d]) * 0.25
    let v = MLXRandom.normal([heads, s, d])
    return (q, k, v)
}

@Suite("Sol-Attn reference")
struct SolAttnReferenceTests {

    /// With nothing rejected there is no approximation left, so the reference
    /// must reproduce dense attention — not approximately, but to float32 noise.
    ///
    /// This is the test that catches the whole family of softmax-accounting
    /// bugs: a wrong running max, a denominator that forgets a term, a
    /// correction that fires when it should not. All of them survive a
    /// "looks about right" comparison and none of them survive this one.
    @Test func allBlocksSelectedIsExactlyDense() {
        let (q, k, v) = structured(heads: 4, s: 512, d: 64, clusters: 6, seed: 1)
        let scale = 1.0 / Float(64).squareRoot()
        // beta far below any standardised score puts every block over threshold.
        var config = SolAttnConfig(beta: -50, blockSize: 64, exactConditioningKV: false)
        config.warmupFraction = 0

        let got = SolAttnReference.attend(queries: q, keys: k, values: v,
                                          scale: scale, config: config, videoSpan: nil)
        let want = dense(q, k, v, scale: scale)
        #expect(relRMS(got, want) < 1e-6)
    }

    /// The tail block is short at every real H3 shape — 15,731 is not a multiple
    /// of 64 — and a pooling that divides it by the full block size moves its
    /// proxy score. Exercised at a length chosen to leave 13 rows over.
    @Test func shortTailBlockStaysExact() {
        let (q, k, v) = structured(heads: 3, s: 525, d: 64, clusters: 5, seed: 2)
        let scale = 1.0 / Float(64).squareRoot()
        var config = SolAttnConfig(beta: -50, blockSize: 64, exactConditioningKV: false)
        config.warmupFraction = 0

        let got = SolAttnReference.attend(queries: q, keys: k, values: v,
                                          scale: scale, config: config, videoSpan: nil)
        #expect(relRMS(got, dense(q, k, v, scale: scale)) < 1e-6)
    }

    /// The correction is exact when the block it stands in for is constant, and
    /// this pins it independently of any selection.
    ///
    /// If every key in a block equals that block's pooled key, then
    /// `sum_t exp(s_qt) v_t` **is** `exp(p_qj) * V_j` and `sum_t exp(s_qt)` is
    /// `n_j * exp(p_qj)` — no approximation anywhere. So the output must equal
    /// dense attention no matter which blocks the router rejected, at any beta.
    ///
    /// This separates the two things a bad quality number could mean. Dense
    /// agreement here says the correction's algebra is right and any error at
    /// beta 1.2 on other input is the approximation doing its job; a failure
    /// here says the correction itself is wrong, and no amount of better test
    /// data would hide it.
    @Test func correctionIsExactForConstantBlocks() {
        MLXRandom.seed(11)
        let heads = 3, blocks = 8, bs = 64, d = 64
        let s = blocks * bs
        // One key vector per block, repeated across the block's rows.
        let perBlock = MLXRandom.normal([heads, blocks, 1, d]) * 0.5
        let k = MLX.repeated(perBlock, count: bs, axis: 2).reshaped([heads, s, d])
        let q = MLXRandom.normal([heads, s, d]) * 0.5
        let v = MLXRandom.normal([heads, s, d])
        let scale = 1.0 / Float(d).squareRoot()

        // A beta that rejects nearly everything, so the correction carries the
        // answer rather than the exact path.
        let config = SolAttnConfig(beta: 1.5, blockSize: bs, exactConditioningKV: false)
        let density = SolAttnReference.density(queries: q, keys: k, values: v,
                                               config: config, videoSpan: nil)
        #expect(density < 0.5)   // the correction is actually being exercised

        let got = SolAttnReference.attend(queries: q, keys: k, values: v,
                                          scale: scale, config: config, videoSpan: nil)
        #expect(relRMS(got, dense(q, k, v, scale: scale)) < 1e-5)
    }

    /// Density has to fall as beta rises, and land near `1 - Phi(beta)`.
    ///
    /// Not asserted tightly: §8 measured H3's proxy distribution as clearly
    /// heavy-tailed (excess kurtosis +2.01), so the Gaussian formula predicts
    /// density only to within 8-18%. A tight assertion here would be asserting
    /// that the model is Gaussian, which it is not.
    @Test func densityFallsWithBeta() {
        let (q, k, v) = structured(heads: 4, s: 1024, d: 64, clusters: 8, seed: 3)
        var last = Float(1.0)
        for beta in [Float(0.5), 1.0, 1.5, 2.0] {
            let config = SolAttnConfig(beta: beta, blockSize: 64, exactConditioningKV: false)
            let d = SolAttnReference.density(queries: q, keys: k, values: v,
                                             config: config, videoSpan: nil)
            #expect(d < last)
            #expect(d > 0)
            last = d
        }
        // At beta = 2 the Gaussian tail is 2.3%; heavy tails put the real figure
        // above it, never near a half.
        #expect(last < 0.25)
    }

    /// The sink is a floor on the selection: conditioning blocks are exact for
    /// every query however the router scored them.
    @Test func sinkKeepsConditioningExact() {
        let (q, k, v) = structured(heads: 2, s: 512, d: 64, clusters: 4, seed: 4)
        let config = SolAttnConfig(beta: 3.0, blockSize: 64, exactConditioningKV: true)
        let pooling = SolAttnRouting.pool(keys: k, values: v, queryCount: 512, blockSize: 64)
        let pooledQ = SolAttnRouting.poolQueries(q, blockSize: 64)
        let sink = SolAttnRouting.sinkKeyBlocks(videoSpan: 128 ..< 512, blockSize: 64,
                                                enabled: config.exactConditioningKV)
        #expect(sink == 2)

        let selected = SolAttnRouting.select(pooledQueries: pooledQ, pooling: pooling,
                                             beta: config.beta, sinkKeyBlocks: sink)
        // Every query block, every head, keeps both conditioning blocks.
        let sinkCols = selected[0..., 0..., 0 ..< 2]
        #expect(MLX.all(sinkCols).item(Bool.self))
    }

    /// A boundary that lands mid-block rounds up, because a block holding
    /// conditioning rows must be exact even if it also holds video rows.
    @Test func sinkRoundsUpAcrossTheBoundary() {
        #expect(SolAttnRouting.sinkKeyBlocks(videoSpan: 100 ..< 512, blockSize: 64,
                                             enabled: true) == 2)
        #expect(SolAttnRouting.sinkKeyBlocks(videoSpan: 0 ..< 512, blockSize: 64,
                                             enabled: true) == 0)
        #expect(SolAttnRouting.sinkKeyBlocks(videoSpan: nil, blockSize: 64,
                                             enabled: true) == 0)
        #expect(SolAttnRouting.sinkKeyBlocks(videoSpan: 100 ..< 512, blockSize: 64,
                                             enabled: false) == 0)
    }

    /// Rejected blocks are corrected, not dropped. With everything rejected the
    /// output is entirely rank-1 — it must still be finite, still normalised,
    /// and it must not equal dense.
    @Test func everythingRejectedStaysFiniteAndNormalised() {
        let (q, k, v) = structured(heads: 2, s: 512, d: 64, clusters: 4, seed: 5)
        let scale = 1.0 / Float(64).squareRoot()
        let config = SolAttnConfig(beta: 50, blockSize: 64, exactConditioningKV: false)

        let got = SolAttnReference.attend(queries: q, keys: k, values: v,
                                          scale: scale, config: config, videoSpan: nil)
        #expect(MLX.all(MLX.isFinite(got)).item(Bool.self))
        // Attention output is a convex combination of values, so it cannot leave
        // their range however coarse the approximation gets.
        let vmax = MLX.max(v.asType(.float32)).item(Float.self)
        let vmin = MLX.min(v.asType(.float32)).item(Float.self)
        #expect(MLX.max(got.asType(.float32)).item(Float.self) <= vmax + 1e-4)
        #expect(MLX.min(got.asType(.float32)).item(Float.self) >= vmin - 1e-4)
    }

    /// The correction must beat masking — same selection, rejected blocks simply
    /// dropped and the softmax renormalised over what is left.
    ///
    /// **This is the assertion worth making here, and an absolute threshold is
    /// not.** How close Sol-Attn lands to dense depends entirely on how much
    /// block structure the input has; on synthetic clusters that number says
    /// something about the generator, not about H3. What *is* input-independent
    /// is that keeping the rejected mass as a rank-1 estimate should beat
    /// throwing it away, because throwing it away inflates the surviving blocks
    /// by exactly the weight it deleted.
    ///
    /// The equivalence class this backend ships with cannot come from here. It
    /// has to be measured on q/k/v captured from a real render — see
    /// `Tools/solattn-measure`.
    @Test func correctionBeatsDroppingTheRejectedBlocks() {
        let (q, k, v) = structured(heads: 4, s: 1024, d: 64, clusters: 8, seed: 6)
        let s = 1024, bs = 64
        let scale = 1.0 / Float(64).squareRoot()
        let config = SolAttnConfig(beta: 1.2, blockSize: bs, exactConditioningKV: false)
        let want = dense(q, k, v, scale: scale)

        let corrected = SolAttnReference.attend(queries: q, keys: k, values: v,
                                                scale: scale, config: config, videoSpan: nil)

        // The same routing, expressed as an additive mask so dense attention
        // renormalises over the kept blocks alone.
        let pooling = SolAttnRouting.pool(keys: k, values: v, queryCount: s, blockSize: bs)
        let selected = SolAttnRouting.select(
            pooledQueries: SolAttnRouting.poolQueries(q, blockSize: bs),
            pooling: pooling, beta: config.beta, sinkKeyBlocks: 0)
        let rows = MLX.repeated(MLX.repeated(selected, count: bs, axis: -1),
                                count: bs, axis: 1)[0..., 0 ..< s, 0 ..< s]
        let additive = MLX.where(rows, MLXArray(Float(0)),
                                 MLXArray(-Float.greatestFiniteMagnitude))
        let masked = MLXFast.scaledDotProductAttention(
            queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
            values: v.expandedDimensions(axis: 0), scale: scale,
            mask: additive.expandedDimensions(axis: 0)).squeezed(axis: 0)

        let withCorrection = relRMS(corrected, want)
        let withoutCorrection = relRMS(masked, want)
        #expect(withCorrection < withoutCorrection)
        // Sanity floor: the output is still attention, not noise.
        #expect(withCorrection < 0.5)
    }
}

@Suite("Sol-Attn policy")
struct SolAttnPolicyTests {

    private func context(block: Int, of count: Int = 50, progress: Double = 0.5,
                         length: Int = 15_750) -> AttentionContext {
        AttentionContext(blockIndex: block, blockCount: count, scheduleProgress: progress,
                         sequenceLength: length, videoSpan: 1_000 ..< length)
    }

    /// The policy is pure arithmetic on the context and must be checkable
    /// without a GPU — the same reason `StepCachePolicy` was split out of the
    /// cache it governs.
    @Test func declinesDuringWarmup() {
        let c = SolAttnConfig()
        #expect(!c.admits(context(block: 25, progress: 0.0)))
        #expect(!c.admits(context(block: 25, progress: 0.19)))
        #expect(c.admits(context(block: 25, progress: 0.20)))
    }

    @Test func declinesAtTheEndsOfTheStack() {
        let c = SolAttnConfig()
        #expect(!c.admits(context(block: 0)))
        #expect(c.admits(context(block: 1)))
        #expect(c.admits(context(block: 48)))
        #expect(!c.admits(context(block: 49)))
    }

    @Test func declinesShortSequences() {
        let c = SolAttnConfig()
        #expect(!c.admits(context(block: 25, length: 4_095)))
        #expect(c.admits(context(block: 25, length: 4_096)))
    }
}

@Suite("Sol-Attn overrides")
struct SolAttnOverrideTests {

    /// Parsing takes a dictionary rather than the process environment, so this
    /// runs without mutating global state — and so a typo in a variable name
    /// cannot silently leave a default in place unnoticed.
    @Test func environmentOverridesParse() {
        let c = SolAttnConfig(environment: [
            "H3_SOL_BETA": "0.8", "H3_SOL_BLOCK": "128", "H3_SOL_SINK": "0",
            "H3_SOL_EDGES": "2", "H3_SOL_WARMUP": "0.4",
            "H3_SOL_DENSE_BLOCKS": "23,24,25",
        ])
        #expect(c.beta == 0.8)
        #expect(c.blockSize == 128)
        #expect(c.exactConditioningKV == false)
        #expect(c.denseEdgeBlocks == 2)
        #expect(c.warmupFraction == 0.4)
        #expect(c.denseBlocks == [23, 24, 25])
    }

    /// An empty environment must leave every default exactly where it was, or
    /// simply having the mechanism present changes what everyone renders.
    @Test func emptyEnvironmentIsTheDefault() {
        #expect(SolAttnConfig(environment: [:]) == SolAttnConfig())
        #expect(SolAttnConfig(environment: [:]).overridesDescription == nil)
    }

    /// Garbage must not silently become a default. A misread beta that fell
    /// back to 1.2 would look exactly like a successful run at 1.2.
    @Test func unparseableValuesLeaveTheDefault() {
        let c = SolAttnConfig(environment: ["H3_SOL_BETA": "banana"])
        #expect(c.beta == SolAttnConfig().beta)
    }

    /// Whatever is in force has to reach the log.
    @Test func overridesAreDescribed() {
        var c = SolAttnConfig(); c.beta = 0.8; c.denseBlocks = [24]
        let d = try! #require(c.overridesDescription)
        #expect(d.contains("beta 0.8"))
        #expect(d.contains("denseBlocks 24"))
    }

    /// `denseBlocks` must actually decline, not merely be stored.
    @Test func denseBlocksDecline() {
        var c = SolAttnConfig(); c.denseBlocks = [24]
        func ctx(_ b: Int) -> AttentionContext {
            AttentionContext(blockIndex: b, blockCount: 50, scheduleProgress: 0.5,
                             sequenceLength: 15_731, videoSpan: 1_000 ..< 15_731)
        }
        #expect(!c.admits(ctx(24)))
        #expect(c.admits(ctx(23)))
        #expect(c.admits(ctx(25)))
    }
}
