// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Foundation

/// Packed-sequence modulation.
///
/// The reference walks a list of `(start, stop, row)` segments and mutates
/// slices in place. That is a memory optimisation, not semantics: every token
/// in `[start, stop)` uses AdaLN row `row`. We flatten that to a per-token row
/// index once and gather, which is the same arithmetic and vectorises.
///
/// Row layout is `timestepRow * 3 + modalityTag`, with the modality tags fixed
/// by the reference as **video 0, text 1, audio 2** (`seg_tag` in
/// `comfy/ldm/minimax/model.py`). Video and audio carry *different* timesteps,
/// which is why there is a timestep row at all — get this wrong and the model
/// modulates the audio branch with the video schedule.
package struct ModSegment: Sendable, Equatable {
    package let start: Int
    package let stop: Int
    package let row: Int
    package init(start: Int, stop: Int, row: Int) {
        self.start = start
        self.stop = stop
        self.row = row
    }
}

package struct ModulationIndex {
    /// `[S]` — AdaLN row for each token in the packed sequence.
    package let rows: MLXArray
    package let tokenCount: Int

    package init(segments: [ModSegment], tokenCount: Int) {
        var r = [Int32](repeating: -1, count: tokenCount)
        for s in segments {
            precondition(s.start >= 0 && s.stop <= tokenCount && s.start <= s.stop,
                         "segment \(s) outside 0..<\(tokenCount)")
            for i in s.start ..< s.stop { r[i] = Int32(s.row) }
        }
        precondition(!r.contains(-1), "mod segments must cover the packed sequence contiguously")
        self.rows = MLXArray(r)
        self.tokenCount = tokenCount
    }

    /// Rows given directly, one per token. Used by the op-level oracle, where
    /// the row assignment comes out of the reference fixture rather than being
    /// derived — the point there is to isolate one operator, not to re-test the
    /// layout.
    package init(rows: MLXArray) {
        self.rows = rows
        self.tokenCount = rows.dim(0)
    }

    /// Rows for a layout under one timestep plan.
    ///
    /// `textTags` is the reference's `minimax_token_tags`: a per-token modality
    /// for the text span, because a prompt carrying vision pads mixes tags
    /// inside a single contiguous segment. Both parity presets are pure text
    /// (all tags 1), so omitting it is correct for them and wrong the moment an
    /// image enters the prompt.
    package init(layout: PackedLayout, plan: TimestepPlan, textTags: [Int]? = nil) {
        var segs: [ModSegment] = []
        for s in layout.segments {
            let base = plan.row(for: s.kind) * 3
            if s.kind == .text, let tags = textTags {
                precondition(tags.count == s.count,
                             "textTags has \(tags.count) entries for a \(s.count)-token text span")
                var runStart = 0
                for i in 1 ... tags.count where i == tags.count || tags[i] != tags[runStart] {
                    segs.append(ModSegment(start: s.start + runStart, stop: s.start + i,
                                           row: base + tags[runStart]))
                    runStart = i
                }
            } else {
                segs.append(ModSegment(start: s.start, stop: s.stop,
                                       row: base + s.kind.modality.rawValue))
            }
        }
        self.init(segments: segs, tokenCount: layout.totalTokens)
    }

    /// `[rows, hidden]` -> `[S, hidden]`, one row per token.
    package func gather(_ table: MLXArray) -> MLXArray { table[rows] }
}

/// `h * (1 + scale) + shift`, per token.
package func modScaleShift(_ h: MLXArray, shift: MLXArray, scale: MLXArray,
                          index: ModulationIndex) -> MLXArray {
    h * (1.0 + index.gather(scale)) + index.gather(shift)
}

/// `x + other * gate`, per token. The reference fuses this as `addcmul_` in
/// place; functionally identical, and the oracle showed fusion does not move
/// the numbers.
package func modGate(_ x: MLXArray, gate: MLXArray, other: MLXArray,
                    index: ModulationIndex) -> MLXArray {
    x + other * index.gather(gate)
}

/// RMSNorm over the last axis: `x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Computed in fp32 and cast back, matching the reference. The accumulation
/// precision here is not optional — doing it in bf16 changes the result well
/// beyond the measured equivalence class.
package struct H3RMSNorm {
    package let weight: MLXArray
    package let eps: Float
    package init(weight: MLXArray, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let n = f * rsqrt(mean(f * f, axis: -1, keepDims: true) + eps)
        return (n * weight.asType(.float32)).asType(x.dtype)
    }
}
