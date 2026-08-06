// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Foundation

/// Sol-Attn written in plain MLX ops, as the definition of what the kernel must
/// compute.
///
/// **This is an oracle, not a product, and it is slower than dense attention.**
/// It materialises `[H, blockSize, S]` scores per query block and then throws
/// most of them away — it does the full quadratic work and adds the correction
/// on top. Its entire job is to be obviously correct so that
/// `SolAttnMetalKernel` has something to be wrong against.
///
/// Keeping it in the shipped target rather than the test target is deliberate:
/// conformance gates per backend, and a backend's own definition of correct
/// should not live somewhere it can drift out of sync with what ships.
///
/// ## The maths
///
/// For query row `q` with per-block selection `S_i`, over key blocks `j`:
///
///     out = [ sum_{j in S_i} sum_{t in j} exp(s_qt - m) v_t
///           + sum_{j not in S_i}         exp(p_qj - m) * V_j ]
///         / [ sum_{j in S_i} sum_{t in j} exp(s_qt - m)
///           + sum_{j not in S_i}         exp(p_qj - m) * n_j ]
///
/// where `s_qt` is the true score, `p_qj = scale * q . Kbar_j` is the score
/// against the block's pooled key, `V_j` is the block's **summed** values and
/// `n_j` its row count.
///
/// **The rejected blocks are corrected, not dropped**, which is the whole
/// difference between Sol-Attn and a block mask. A mask renormalises the
/// softmax over the kept blocks alone and silently inflates their weight; the
/// correction keeps every block's mass in the denominator, standing in for the
/// rejected ones with a rank-1 estimate. Dropping them instead is a one-line
/// change here and it is not the same method.
package enum SolAttnReference {

    /// `[H, S, D]` in, `[H, S, D]` out.
    ///
    /// - Parameter scale: `1/sqrt(headDim)`, already applied to the true scores
    ///   and to the proxy scores alike. Unlike in the router, the scale matters
    ///   here: these exponentials are compared against each other.
    package static func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                               scale: Float, config: SolAttnConfig,
                               videoSpan: Range<Int>?) -> MLXArray {
        let heads = queries.dim(0)
        let s = queries.dim(1)
        let d = queries.dim(2)
        let bs = config.blockSize

        let pooling = SolAttnRouting.pool(keys: keys, values: values,
                                          queryCount: s, blockSize: bs)
        let pooledQ = SolAttnRouting.poolQueries(queries, blockSize: bs)
        let sink = SolAttnRouting.sinkKeyBlocks(videoSpan: videoSpan, blockSize: bs,
                                                enabled: config.exactConditioningKV)
        let selected = SolAttnRouting.select(pooledQueries: pooledQ, pooling: pooling,
                                             beta: config.beta, sinkKeyBlocks: sink)

        let k32 = keys.asType(.float32)
        let v32 = values.asType(.float32)
        let pk32 = pooling.pooledKeys.asType(.float32)
        let sv32 = pooling.summedValues.asType(.float32)
        let counts = pooling.counts.reshaped([1, 1, pooling.keyBlocks])

        // Row-level expansion of the block selection, clipped back to S. Built
        // once for the whole call and sliced per query block.
        let rowSelected = MLX.repeated(selected, count: bs, axis: -1)
            .reshaped([heads, pooling.queryBlocks, pooling.keyBlocks * bs])[0..., 0..., 0 ..< s]

        let negInf = MLXArray(-Float.greatestFiniteMagnitude)
        var out = [MLXArray]()
        out.reserveCapacity(pooling.queryBlocks)

        for i in 0 ..< pooling.queryBlocks {
            let lo = i * bs
            let hi = Swift.min(lo + bs, s)
            let q = queries[0..., lo ..< hi, 0...].asType(.float32)   // [H, b, D]

            // Exact half: true scores, everything outside the selection removed.
            let scores = MLX.matmul(q, k32.transposed(0, 2, 1)) * scale   // [H, b, S]
            let keep = rowSelected[0..., i, 0...].reshaped([heads, 1, s])
            let exactScores = MLX.where(keep, scores, negInf)

            // Approximate half: one proxy score per rejected block.
            let proxy = MLX.matmul(q, pk32.transposed(0, 2, 1)) * scale   // [H, b, Nk]
            let reject = MLX.logicalNot(selected[0..., i, 0...]).reshaped([heads, 1, pooling.keyBlocks])
            let approxScores = MLX.where(reject, proxy, negInf)

            // One running max across both halves. Taking them separately would
            // renormalise the two contributions against different maxima, which
            // is exactly the bug that makes a sparse attention look plausible
            // and score badly.
            let m = MLX.maximum(MLX.max(exactScores, axis: -1, keepDims: true),
                                MLX.max(approxScores, axis: -1, keepDims: true))

            let pe = MLX.exp(exactScores - m)             // [H, b, S]
            let pa = MLX.exp(approxScores - m)            // [H, b, Nk]

            let num = MLX.matmul(pe, v32) + MLX.matmul(pa, sv32)
            let den = MLX.sum(pe, axis: -1, keepDims: true)
                    + MLX.sum(pa * counts, axis: -1, keepDims: true)

            out.append(num / den)
        }

        return MLX.concatenated(out, axis: 1).asType(queries.dtype)
            .reshaped([heads, s, d])
    }

    /// The density the router actually selected, as a fraction in `[0, 1]`.
    ///
    /// Reported rather than assumed: `1 - Phi(beta)` predicts this to within
    /// 8-18% on H3 (§8), and the gap is the heavy tail. A backend that believes
    /// the formula over the measurement will mis-report its own cost.
    package static func density(queries: MLXArray, keys: MLXArray, values: MLXArray,
                                config: SolAttnConfig, videoSpan: Range<Int>?) -> Float {
        let pooling = SolAttnRouting.pool(keys: keys, values: values,
                                          queryCount: queries.dim(1), blockSize: config.blockSize)
        let pooledQ = SolAttnRouting.poolQueries(queries, blockSize: config.blockSize)
        let sink = SolAttnRouting.sinkKeyBlocks(videoSpan: videoSpan, blockSize: config.blockSize,
                                                enabled: config.exactConditioningKV)
        let selected = SolAttnRouting.select(pooledQueries: pooledQ, pooling: pooling,
                                             beta: config.beta, sinkKeyBlocks: sink)
        return MLX.mean(selected.asType(.float32)).item(Float.self)
    }
}
