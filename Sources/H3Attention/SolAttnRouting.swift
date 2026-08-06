// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Foundation

/// Sol-Attn's knobs, and the policy that decides when it declines a call.
///
/// Defaults are the configuration measured in `docs/SOL_ATTN.md` §8 against
/// q/k/v captured from a real render — beta 1.2 with the exact-KV sink — not
/// the paper's, which were never fitted to H3.
package struct SolAttnConfig: Sendable, Equatable {

    /// Gaussian tail cutoff. The density of blocks kept exact is `1 - Phi(beta)`,
    /// so 1.0 keeps ~16%, 1.5 ~7%, 2.0 ~2.7%.
    ///
    /// **This is the paper's beta, not kijai's `tau`.** That node exposes a
    /// parameter named `tau` whose tooltip reads "Threshold beta"; it is this
    /// quantity. The paper's tau is the resulting per-row threshold, which is
    /// derived rather than set.
    package var beta: Float

    /// Query and key block size, in tokens.
    ///
    /// Routing is per block in both axes, so this trades routing cost against
    /// how precisely the selection can follow the attention structure.
    package var blockSize: Int

    /// Keep the packed conditioning rows exact as keys for every query.
    ///
    /// Not optional for H3 in practice. NVIDIA's own notes record a text-only
    /// sink where "the picture scored best of its set while its dialogue fell
    /// apart" — the conditioning rows carry text, audio and lip-sync, and
    /// approximating them is heard before it is seen.
    package var exactConditioningKV: Bool

    /// Transformer blocks at each end of the stack that run dense.
    ///
    /// The convention — the paper's first-layer warm-up, kijai's "first and
    /// last" — is about *propagation*: error in the last block reaches the
    /// output undamped. Note that §8 measured the *middle* of the stack as
    /// hardest to approximate (block 24 at rel_rms 0.245 against 0.132 and
    /// 0.142 at the ends), so this default deliberately leaves the hardest
    /// block sparsified. Both facts are true at once and the right policy
    /// weights one by the other; this is the measured convention, not a
    /// finished answer.
    package var denseEdgeBlocks: Int

    /// Fraction of the sampling schedule that runs dense before sparsity starts.
    ///
    /// The paper specifies a dense warm-up and attributes it to prior work
    /// without ablating it. It specifies no late-schedule dense phase; the
    /// ComfyUI port's `end_percent` is the implementer's addition and is not
    /// reproduced here.
    package var warmupFraction: Double

    /// Below this packed length the routing costs more than the sparsity saves.
    ///
    /// Routing is O(S/blockSize) squared in the proxy map plus two poolings; at
    /// a few thousand tokens that is a real fraction of an attention call that
    /// was already cheap.
    package var minSequenceLength: Int

    /// Individual blocks forced dense, on top of `denseEdgeBlocks`.
    ///
    /// Exists because the measurement disagrees with the convention. The
    /// published policy excludes the *ends* of the stack, on a propagation
    /// argument: error in the last block reaches the output undamped. But both
    /// §8 on CUDA and §11 here measure the **middle** as hardest to approximate
    /// — block 24 at 0.287 against 0.193 and 0.222 at the ends — so excluding
    /// only the ends leaves the single worst block sparsified. This is how that
    /// gets tested without inventing a policy first.
    package var denseBlocks: Set<Int>

    package init(beta: Float = 1.2,
                 blockSize: Int = 64,
                 exactConditioningKV: Bool = true,
                 denseEdgeBlocks: Int = 1,
                 warmupFraction: Double = 0.20,
                 minSequenceLength: Int = 4096,
                 denseBlocks: Set<Int> = []) {
        self.beta = beta
        self.blockSize = blockSize
        self.exactConditioningKV = exactConditioningKV
        self.denseEdgeBlocks = denseEdgeBlocks
        self.warmupFraction = warmupFraction
        self.minSequenceLength = minSequenceLength
        self.denseBlocks = denseBlocks
    }

    /// Overrides read from the environment, for experiments on a real render.
    ///
    /// **Deliberately not CLI options.** This backend is opt-in and not yet
    /// shippable; `h3 render` should not grow five tuning flags for a
    /// configuration nobody can select by default, and threading them through
    /// `RenderRequest` and the pipeline would put experimental surface into the
    /// public API. Same reasoning as `DiTBlock.qkvCapture`. If Sol-Attn is ever
    /// promoted ahead of dense, these become real options with real help text.
    ///
    /// Takes the environment as an argument rather than reading the process's,
    /// so the parsing is testable without mutating global state.
    ///
    ///     H3_SOL_BETA=0.8 H3_SOL_BLOCK=128 H3_SOL_WARMUP=0.4 \
    ///     H3_SOL_DENSE_BLOCKS=23,24,25 H3_SOL_SINK=1 H3_SOL_EDGES=1
    package init(environment env: [String: String]) {
        self.init()
        if let v = env["H3_SOL_BETA"], let x = Float(v) { beta = x }
        if let v = env["H3_SOL_BLOCK"], let x = Int(v) { blockSize = x }
        if let v = env["H3_SOL_SINK"] { exactConditioningKV = (v != "0") }
        if let v = env["H3_SOL_EDGES"], let x = Int(v) { denseEdgeBlocks = x }
        if let v = env["H3_SOL_WARMUP"], let x = Double(v) { warmupFraction = x }
        if let v = env["H3_SOL_MINLEN"], let x = Int(v) { minSequenceLength = x }
        if let v = env["H3_SOL_DENSE_BLOCKS"] {
            denseBlocks = Set(v.split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
        }
    }

    /// Everything that differs from the defaults, for the render log. A run
    /// tuned by environment variable and not saying so is the same failure as a
    /// silent backend fallback.
    package var overridesDescription: String? {
        let d = SolAttnConfig()
        var parts: [String] = []
        if beta != d.beta { parts.append("beta \(beta)") }
        if blockSize != d.blockSize { parts.append("blockSize \(blockSize)") }
        if exactConditioningKV != d.exactConditioningKV {
            parts.append("sink \(exactConditioningKV ? "on" : "OFF")")
        }
        if denseEdgeBlocks != d.denseEdgeBlocks { parts.append("denseEdges \(denseEdgeBlocks)") }
        if warmupFraction != d.warmupFraction { parts.append("warmup \(warmupFraction)") }
        if minSequenceLength != d.minSequenceLength { parts.append("minLen \(minSequenceLength)") }
        if !denseBlocks.isEmpty {
            parts.append("denseBlocks \(denseBlocks.sorted().map(String.init).joined(separator: ","))")
        }
        return parts.isEmpty ? nil : parts.joined(separator: ", ")
    }

    /// Whether this call should be sparsified at all.
    ///
    /// Separated from the routing so the policy is testable without building
    /// tensors, in the same spirit as `StepCachePolicy`: the rules are the part
    /// that gets argued about, and they should not need a GPU to check.
    package func admits(_ context: AttentionContext) -> Bool {
        guard context.sequenceLength >= minSequenceLength else { return false }
        guard context.scheduleProgress >= warmupFraction else { return false }
        guard context.blockIndex >= denseEdgeBlocks,
              context.blockIndex < context.blockCount - denseEdgeBlocks else { return false }
        guard !denseBlocks.contains(context.blockIndex) else { return false }
        return true
    }
}

/// The block-level quantities Sol-Attn needs, computed once per attention call.
///
/// The pooled keys serve twice — as the router's proxy for a block, and as the
/// rank-1 stand-in for every key in a block the router rejected — which is why
/// they are computed here rather than inside either stage.
package struct SolAttnPooling {
    /// `[H, Nk, D]` — mean key per key block.
    package let pooledKeys: MLXArray
    /// `[H, Nk, D]` — **sum**, not mean, of values per key block.
    ///
    /// The correction multiplies one exponential by the block's total value
    /// mass, so the row count belongs inside this sum rather than in a separate
    /// multiply the kernel would have to carry.
    package let summedValues: MLXArray
    /// `[Nk]` — rows in each key block. The last block is short whenever the
    /// packed length is not a multiple of `blockSize`, which for H3 it never is.
    package let counts: MLXArray
    package let keyBlocks: Int
    package let queryBlocks: Int
}

/// Block pooling, the proxy map, and the per-row threshold.
///
/// All of this is ordinary MLX: the proxy map is `Nq x Nk` per head — 247^2 at
/// the verified shape — which is four orders of magnitude smaller than the
/// score matrix it stands in for. There is nothing here worth a custom kernel,
/// and the paper's O(d) variance estimator (Eq. 15) exists to avoid
/// materialising a map that at *this* sequence length costs 13 MB. It is not
/// needed until the map itself is the problem.
package enum SolAttnRouting {

    /// Mean-pools keys and sums values per block, handling a short tail block.
    ///
    /// Zero-padding to a whole number of blocks and then dividing by the *true*
    /// count is what keeps the tail honest: padding with zeros and dividing by
    /// `blockSize` would pull the last pooled key toward the origin in
    /// proportion to how short it is, which moves its proxy score and so
    /// changes which blocks get selected.
    package static func pool(keys: MLXArray, values: MLXArray,
                             queryCount: Int, blockSize: Int) -> SolAttnPooling {
        let heads = keys.dim(0)
        let s = keys.dim(1)
        let d = keys.dim(2)
        let nk = (s + blockSize - 1) / blockSize
        let nq = (queryCount + blockSize - 1) / blockSize
        let padded = nk * blockSize - s

        var k = keys
        var v = values
        if padded > 0 {
            k = MLX.padded(k, widths: [.init((0, 0)), .init((0, padded)), .init((0, 0))])
            v = MLX.padded(v, widths: [.init((0, 0)), .init((0, padded)), .init((0, 0))])
        }

        let kb = k.reshaped([heads, nk, blockSize, d])
        let vb = v.reshaped([heads, nk, blockSize, d])

        var counts = MLXArray.full([nk], values: MLXArray(Float(blockSize)))
        if padded > 0 {
            counts[nk - 1] = MLXArray(Float(blockSize - padded))
        }

        let summed = MLX.sum(kb, axis: 2)                      // [H, Nk, D]
        let pooledKeys = summed / counts.reshaped([1, nk, 1])
        let summedValues = MLX.sum(vb, axis: 2)                // [H, Nk, D]

        return SolAttnPooling(pooledKeys: pooledKeys, summedValues: summedValues,
                              counts: counts, keyBlocks: nk, queryBlocks: nq)
    }

    /// The proxy map and the resulting per-(head, query block) selection.
    ///
    /// Returns `[H, Nq, Nk]` of `true` where the block is kept exact.
    ///
    /// **The scale factor is deliberately not applied.** `tau = mu + beta*sigma`
    /// is computed from the same row it is compared against, so any positive
    /// scaling multiplies score, mean and standard deviation alike and cancels.
    /// Applying it would cost a full pass over the map to produce an identical
    /// selection.
    package static func select(pooledQueries: MLXArray, pooling: SolAttnPooling,
                               beta: Float, sinkKeyBlocks: Int) -> MLXArray {
        // [H, Nq, D] x [H, D, Nk] -> [H, Nq, Nk], in float32: the router's
        // statistics are means and standard deviations over a few hundred
        // values, and bf16 has seven mantissa bits to spend on them.
        let proxy = MLX.matmul(pooledQueries.asType(.float32),
                               pooling.pooledKeys.asType(.float32).transposed(0, 2, 1))

        let mu = MLX.mean(proxy, axis: -1, keepDims: true)
        let centred = proxy - mu
        let sigma = MLX.sqrt(MLX.mean(centred * centred, axis: -1, keepDims: true))
        let tau = mu + MLXArray(beta) * sigma

        var selected = proxy .> tau

        // The sink is a floor on the selection, not a separate code path: the
        // conditioning blocks are simply always in it. Expressing it here means
        // the kernel has one uniform notion of "selected" and never learns what
        // conditioning is.
        if sinkKeyBlocks > 0 {
            let nk = pooling.keyBlocks
            let index = MLXArray(0 ..< Int32(nk)).reshaped([1, 1, nk])
            selected = selected .|| (index .< MLXArray(Int32(sinkKeyBlocks)))
        }
        return selected
    }

    /// How many leading key blocks are covered by the conditioning span.
    ///
    /// Rounded **up**: a block straddling the boundary holds conditioning rows,
    /// and a sink that half-covers the audio rows is the failure the sink
    /// exists to prevent. The cost of rounding up is at most one extra exact
    /// block out of ~247.
    package static func sinkKeyBlocks(videoSpan: Range<Int>?, blockSize: Int,
                                      enabled: Bool) -> Int {
        guard enabled, let videoSpan, videoSpan.lowerBound > 0 else { return 0 }
        return (videoSpan.lowerBound + blockSize - 1) / blockSize
    }

    /// Mean-pools queries per block. Split out because the query tail needs the
    /// same true-count division the key tail does.
    package static func poolQueries(_ queries: MLXArray, blockSize: Int) -> MLXArray {
        let heads = queries.dim(0)
        let s = queries.dim(1)
        let d = queries.dim(2)
        let nq = (s + blockSize - 1) / blockSize
        let padded = nq * blockSize - s

        var q = queries
        if padded > 0 {
            q = MLX.padded(q, widths: [.init((0, 0)), .init((0, padded)), .init((0, 0))])
        }
        var counts = MLXArray.full([nq], values: MLXArray(Float(blockSize)))
        if padded > 0 {
            counts[nq - 1] = MLXArray(Float(blockSize - padded))
        }
        let summed = MLX.sum(q.reshaped([heads, nq, blockSize, d]), axis: 2)
        return summed / counts.reshaped([1, nq, 1])
    }
}
