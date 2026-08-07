// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXFast
import H3Foundation

/// The numerical definition of axial attention. Slow on purpose.
///
/// **Sliced SDPA cannot be the oracle, and that is the trap this file exists to
/// avoid.** The obvious implementation runs one attention over the same frame,
/// another over the same spatial column, another over the prefix, and combines
/// the three. It cannot work: each of those is normalised by *its own* softmax
/// denominator, so adding or averaging them produces something that is not
/// attention over the union. The result would look plausible, be wrong
/// everywhere, and disagree with dense in the degenerate case where it must not
/// — which is precisely why the identity test below is the first one written.
///
/// There must be **one softmax over the union of allowed keys**. Two
/// implementations here do that, and each exists to check something different:
///
///  * ``attend(queries:keys:values:scale:topology:queryChunk:)`` chunks over
///    *queries*, which is exact by construction — every query's softmax is
///    complete within its chunk — and applies an additive mask. This is the
///    definition. It is simple enough to be obviously right, and slower than
///    dense.
///  * ``attendStreamingKeys(queries:keys:values:scale:topology:keyTile:)``
///    chunks over *keys* with a running `(m, l, acc)`. This is the arithmetic a
///    Metal kernel would have to implement, and getting it wrong is silent. It
///    is validated against the first one rather than trusted.
///
/// If those two ever disagree, the streaming recurrence is broken and 6E has no
/// foundation.
package enum AxialReference {

    /// Additive mask for a block of query rows: 0 where allowed, -inf elsewhere.
    ///
    /// Built from `AxialTopology.allows` semantics but vectorised. The
    /// per-pair definition is the check, not the implementation — see
    /// `AxialTopologyTests.bulkMaskMatchesThePairwiseDefinition`.
    package static func maskRows(_ rows: Range<Int>, topology t: AxialTopology,
                                 dtype: DType = .float32) -> MLXArray {
        let s = t.videoSpan.upperBound
        let per = t.tokensPerFrame
        let vLo = t.videoSpan.lowerBound

        // Key-side attributes, computed once for the whole sequence.
        let keyIdx = MLXArray(Array(Int32(0) ..< Int32(s)))
        let isPrefixKey = keyIdx .< Int32(vLo)
        let kv = keyIdx - Int32(vLo)
        let kFrame = MLX.floor(kv.asType(.float32) / Float(per)).asType(.int32)
        let kPos = kv - kFrame * Int32(per)
        var isLandmarkKey = MLXArray.zeros([s], type: Bool.self)
        for f in t.landmarkFrames {
            isLandmarkKey = MLX.logicalOr(isLandmarkKey, kFrame .== Int32(f))
        }
        isLandmarkKey = MLX.logicalAnd(isLandmarkKey, MLX.logicalNot(isPrefixKey))

        var allowed: [MLXArray] = []
        allowed.reserveCapacity(rows.count)
        for q in rows {
            if q < vLo {
                allowed.append(MLXArray.ones([s], type: Bool.self))       // prefix query: dense
                continue
            }
            let qv = q - vLo
            let qFrame = Int32(qv / per), qPos = Int32(qv % per)
            var row = isPrefixKey                          // the prefix, always
            row = MLX.logicalOr(row, MLX.logicalAnd(MLX.logicalNot(isPrefixKey),
                                                    kFrame .== qFrame))
            row = MLX.logicalOr(row, MLX.logicalAnd(MLX.logicalNot(isPrefixKey),
                                                    kPos .== qPos))
            row = MLX.logicalOr(row, isLandmarkKey)
            allowed.append(row)
        }
        let stacked = MLX.stacked(allowed, axis: 0)
        // -1e30 rather than -infinity: the streaming path computes
        // `exp(m - m_new)` and two infinities there give NaN, so the two
        // implementations have to agree on a large finite sentinel or they
        // stop being comparable in exactly the degenerate cases that matter.
        return MLX.where(stacked, MLXArray(Float(0)), MLXArray(Float(-1e30)))
            .asType(dtype)
    }

    /// The definition: chunk over queries, one masked softmax each.
    ///
    /// `[H, S, D]` in and out, float32 out.
    package static func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                               scale: Float, topology: AxialTopology,
                               queryChunk: Int = 512) -> MLXArray {
        let heads = queries.dim(0), s = queries.dim(1)
        var out: [MLXArray] = []
        var lo = 0
        while lo < s {
            let hi = Swift.min(lo + queryChunk, s)
            let mask = maskRows(lo ..< hi, topology: topology)
                .reshaped([1, 1, hi - lo, s])
            let o = MLXFast.scaledDotProductAttention(
                queries: queries[0..., lo ..< hi, 0...]
                    .expandedDimensions(axis: 0).asType(.float32),
                keys: keys.expandedDimensions(axis: 0).asType(.float32),
                values: values.expandedDimensions(axis: 0).asType(.float32),
                // `[1, 1, rows, S]` and left to broadcast over heads. Expanding
                // it here would materialise `[1, 56, rows, S]` — 1.8 GB in
                // fp32 at a 512-row chunk, for a mask whose 56 copies are
                // identical.
                scale: scale, mask: mask)
            _ = heads
            out.append(o.squeezed(axis: 0))
            lo = hi
        }
        return MLX.concatenated(out, axis: 1)
    }

    /// The same answer, accumulated over key tiles with an online softmax.
    ///
    /// This is the recurrence a kernel has to implement:
    ///
    ///     m_new = max(m, rowmax(scores))
    ///     l     = l * exp(m - m_new) + sum(exp(scores - m_new))
    ///     acc   = acc * exp(m - m_new) + exp(scores - m_new) @ V
    ///
    /// `m` starts at a large finite negative rather than `-infinity` because
    /// the first rescale evaluates `exp(m - m_new)` with both terms infinite,
    /// which is `exp(NaN)`. A finite sentinel underflows to zero, which is the
    /// answer that was wanted. The same mistake in the Sol-Attn kernel is why
    /// this is spelled out here rather than left to the kernel author.
    package static func attendStreamingKeys(queries: MLXArray, keys: MLXArray,
                                            values: MLXArray, scale: Float,
                                            topology: AxialTopology,
                                            queryChunk: Int = 512,
                                            keyTile: Int = 1024) -> MLXArray {
        let s = queries.dim(1), d = queries.dim(2), heads = queries.dim(0)
        var out: [MLXArray] = []
        var lo = 0
        while lo < s {
            let hi = Swift.min(lo + queryChunk, s)
            let q = queries[0..., lo ..< hi, 0...].asType(.float32)
            let mask = maskRows(lo ..< hi, topology: topology)   // [rows, S]

            var m = MLXArray.full([heads, hi - lo, 1], values: MLXArray(Float(-3e38)))
            var l = MLXArray.zeros([heads, hi - lo, 1])
            var acc = MLXArray.zeros([heads, hi - lo, d])

            var kLo = 0
            while kLo < s {
                let kHi = Swift.min(kLo + keyTile, s)
                let kt = keys[0..., kLo ..< kHi, 0...].asType(.float32)
                let vt = values[0..., kLo ..< kHi, 0...].asType(.float32)
                var scores = MLX.matmul(q, kt.transposed(0, 2, 1)) * scale
                scores = scores + mask[0..., kLo ..< kHi].expandedDimensions(axis: 0)

                let tileMax = MLX.max(scores, axis: -1, keepDims: true)
                let mNew = MLX.maximum(m, tileMax)
                let rescale = MLX.exp(m - mNew)
                let p = MLX.exp(scores - mNew)
                l = l * rescale + MLX.sum(p, axis: -1, keepDims: true)
                acc = acc * rescale + MLX.matmul(p, vt)
                m = mNew
                kLo = kHi
            }
            // A row whose every key was masked has l == 0. It cannot happen
            // under this topology — the prefix is always allowed — but dividing
            // by it would produce NaN rather than an error, so it is guarded
            // where a kernel would also have to guard it.
            out.append(acc / MLX.maximum(l, MLXArray(Float(1e-30))))
            lo = hi
        }
        return MLX.concatenated(out, axis: 1)
    }
}
