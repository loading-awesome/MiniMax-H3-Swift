// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXFast
import H3Foundation

/// RMSNorm, the AdaLN gather and the modulation, in one pass over the hidden
/// state — and the gated residual likewise.
///
/// ## What this saves, and what it does not
///
/// **It does not touch the AdaLN projection.** Those 13 B parameters are 26 of
/// the checkpoint's 66 GB and they have to be read whatever happens; the
/// projection stays on MLX's `matmul`, which is better tuned than anything
/// written here would be. What is fused is everything downstream of it, which
/// is pure memory traffic:
///
///     h  = norm1(x)                      read x, write h
///     s  = index.gather(scale)           write [S, hidden]  <- from ~9 rows
///     sh = index.gather(shift)           write [S, hidden]  <- from ~9 rows
///     h1 = h * (1 + s) + sh              read 3, write 1
///
/// At S = 15,731 and hidden = 5,376 one `[S, hidden]` bf16 tensor is 169 MB, so
/// that sequence moves roughly 1.2 GB to produce 169 MB of answer. **The two
/// gathers are the indefensible part**: the modulation coefficients live in a
/// table of about nine distinct rows, and the gather's only job is to broadcast
/// them into a tensor the size of the hidden state so that an elementwise
/// kernel can read them back. Fusing the indexing into the arithmetic deletes
/// both materialisations outright.
///
/// Twice per block for the two norms, twice more for the two gated residuals,
/// fifty blocks, two CFG branches. It is not a small fraction of a step.
///
/// ## Exactness
///
/// **This is not an approximation and must never become one.** Every
/// intermediate is rounded to the input dtype at exactly the points MLX's
/// unfused sequence rounds — `(T)` casts below are load-bearing, not cosmetic.
/// The temptation is to keep the modulation in fp32 through to the store, which
/// would be *more* accurate and would still be wrong: this tree's conformance
/// taps are calibrated against measured per-op equivalence classes, and an op
/// that lands closer to fp32 than the measurement it is checked against is an
/// unexplained deviation like any other.
///
/// The fp32 reduction inside RMSNorm is likewise not optional. Accumulating a
/// 5,376-wide sum of squares in bf16 moves the result well outside the block's
/// equivalence class — that was measured before this file existed and is why
/// `H3RMSNorm` upcasts.
///
/// Both entry points return nil for any shape they were not written for, and
/// the callers fall back to the unfused path. A fused kernel that quietly
/// handled an unmeasured shape would be worse than no kernel.
package enum FusedModulation {

    /// Threads per row. One threadgroup handles one token, striding across the
    /// hidden dimension, so this is also the width of the reduction tree.
    ///
    /// 256 divides 5,376 exactly (21 elements per thread) at H3's hidden size,
    /// but nothing here depends on that — the strided loop handles any width and
    /// the ragged tail costs one predicated iteration.
    package static let threadsPerRow = 256

    /// `out[i] = (rms(x[i]) * w) * (1 + scale[rows[i]]) + shift[rows[i]]`
    ///
    /// `scale` and `shift` are the AdaLN table rows, `[R, hidden]`, indexed per
    /// token by `rows`. That indirection is the whole point: it is what removes
    /// the two `[S, hidden]` gathers.
    private static let normSource = """
        const uint tokenRow = threadgroup_position_in_grid.x;
        const uint tid = thread_index_in_threadgroup;

        threadgroup float partial[THREADS];

        const uint base = tokenRow * (uint)H;

        // Pass one: sum of squares, in fp32. Not optional — a bf16 accumulation
        // over 5,376 terms lands outside the block's measured equivalence class.
        float acc = 0.0f;
        for (uint i = tid; i < (uint)H; i += (uint)THREADS) {
            const float v = (float)x[base + i];
            acc += v * v;
        }
        partial[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = (uint)THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) { partial[tid] += partial[tid + stride]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv = metal::rsqrt(partial[0] / (float)H + eps[0]);

        // Pass two. The modulation row is read once per thread rather than
        // materialised into an [S, hidden] tensor, which is the traffic this
        // kernel exists to avoid.
        const uint modBase = (uint)rows[tokenRow] * (uint)H;

        for (uint i = tid; i < (uint)H; i += (uint)THREADS) {
            // Matching MLX's unfused sequence rounding for rounding. Each (T)
            // is a point where the readable path also rounds; removing one
            // makes this kernel more accurate than the thing it must agree
            // with.
            const float n = (float)x[base + i] * inv * (float)w[i];
            const T h = (T)n;
            const T one_plus = (T)(1.0f + (float)scale[modBase + i]);
            const T prod = (T)((float)h * (float)one_plus);
            out[base + i] = (T)((float)prod + (float)shift[modBase + i]);
        }
        """

    /// `out[i] = x[i] + other[i] * gate[rows[i]]`
    ///
    /// No reduction, so this is one flat elementwise pass — one thread per
    /// element, and the only saving is the gate gather. That is still 169 MB
    /// written and read back per call at production shape, twice per block.
    private static let gateSource = """
        const uint idx = thread_position_in_grid.x;
        if (idx >= (uint)TOTAL) return;
        const uint tokenRow = idx / (uint)H;
        const uint col = idx - tokenRow * (uint)H;
        const uint modBase = (uint)rows[tokenRow] * (uint)H;

        const T scaled = (T)((float)other[idx] * (float)gate[modBase + col]);
        out[idx] = (T)((float)x[idx] + (float)scaled);
        """

    private static let normKernel = MLXFast.metalKernel(
        name: "h3_fused_mod_rmsnorm",
        inputNames: ["x", "w", "scale", "shift", "rows", "eps"],
        outputNames: ["out"],
        source: normSource)

    private static let gateKernel = MLXFast.metalKernel(
        name: "h3_fused_gated_residual",
        inputNames: ["x", "other", "gate", "rows"],
        outputNames: ["out"],
        source: gateSource)

    /// **Off by default: measured, and it did not clear its gate.**
    ///
    /// Eight controlled renders at 864x480x124, 20 steps, one seed, one
    /// machine, fusion switched inside a single binary so the comparison
    /// isolated the kernel:
    ///
    /// | arm | mean s/step | median full step |
    /// |---|---|---|
    /// | `control-cached` (3 runs) | 33.23 | 58.84 |
    /// | `fused-cached` (3 runs)   | 32.64 | 58.29 |
    ///
    /// **1.81% on wall clock, 0.94% on the steps the kernel actually touches**,
    /// against a gate of 5%. The gain is real — the two sets of runs do not
    /// overlap, and the full-step figure repeats to 0.1% — it is simply small.
    ///
    /// The arithmetic said so before the kernel was written, and doing it
    /// afterwards is the lesson worth keeping: roughly fourteen `[S, hidden]`
    /// tensor passes saved per block, 169 MB each, fifty blocks, is about
    /// 118 GB — a few hundred milliseconds against a 58.8-second step. A step
    /// at this shape is bound by attention and the large GEMMs, not by
    /// modulation traffic. MLX also fuses elementwise chains of its own, so the
    /// path being replaced was never as naive as the source reads.
    ///
    /// It is also **not a free swap**: the norm's reduction reassociates, and a
    /// few ulps at block 0 propagate through fifty blocks and twenty steps into
    /// a different render. Same quality, different pixels — which means
    /// enabling it would need its own quality pass. For 1%, that is not a trade
    /// worth making.
    ///
    /// Kept rather than deleted because the measurement is machine-specific:
    /// the ratio of memory bandwidth to compute is what makes this small here,
    /// and that ratio is not the same on every Apple part. `H3_FUSED_MODULATION=1`
    /// turns it on; anyone who does should re-run the controls first.
    ///
    /// **Read by `DiTBlock`, not by the entry points below.** The kernels
    /// answer "is this a shape I can do", the caller answers "should I". Wiring
    /// the switch into the kernels made every differential test decline the
    /// moment the default flipped, which would have left a disabled kernel
    /// covered by a suite that silently tested nothing.
    package static let enabled: Bool =
        ProcessInfo.processInfo.environment["H3_FUSED_MODULATION"] == "1"

    /// Fused `modScaleShift(norm(x), shift:scale:index:)`.
    ///
    /// - Returns: nil when the shapes are not the two-dimensional packed case
    ///   this was written and measured for, leaving the caller on the readable
    ///   path.
    package static func modulatedRMSNorm(_ x: MLXArray, weight: MLXArray, eps: Float,
                                         shift: MLXArray, scale: MLXArray,
                                         index: ModulationIndex) -> MLXArray? {
        // The refiner's batched text path is three dimensional and a few hundred
        // rows; it is not worth a second kernel and it is not where the time is.
        guard x.ndim == 2, weight.ndim == 1 else { return nil }
        let s = x.dim(0), h = x.dim(1)
        guard weight.dim(0) == h, shift.ndim == 2, scale.ndim == 2,
              shift.dim(1) == h, scale.dim(1) == h,
              shift.dim(0) == scale.dim(0),
              index.rows.dim(0) == s else { return nil }
        // The kernel indexes the table with `rows` directly and Metal will not
        // trap on an out-of-bounds read; it returns whatever is in memory. The
        // unfused `gather` would fault or clamp. Checked here rather than
        // trusted, because a silently wrong modulation row produces output that
        // is merely subtly wrong.
        guard x.dtype == shift.dtype, x.dtype == scale.dtype,
              weight.dtype == x.dtype else { return nil }

        let out = normKernel(
            [x, weight, scale, shift, index.rows.asType(.int32), MLXArray([eps])],
            template: [("T", x.dtype), ("H", h), ("THREADS", threadsPerRow)],
            grid: (threadsPerRow * s, 1, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [[s, h]],
            outputDTypes: [x.dtype])
        return out[0]
    }

    /// Fused `modGate(x, gate:other:index:)`.
    package static func gatedResidual(_ x: MLXArray, gate: MLXArray, other: MLXArray,
                                      index: ModulationIndex) -> MLXArray? {
        guard x.ndim == 2, other.ndim == 2, gate.ndim == 2 else { return nil }
        let s = x.dim(0), h = x.dim(1)
        guard other.dim(0) == s, other.dim(1) == h, gate.dim(1) == h,
              index.rows.dim(0) == s,
              x.dtype == other.dtype, x.dtype == gate.dtype else { return nil }

        let total = s * h
        let out = gateKernel(
            [x, other, gate, index.rows.asType(.int32)],
            template: [("T", x.dtype), ("H", h), ("TOTAL", total)],
            grid: (total, 1, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [[s, h]],
            outputDTypes: [x.dtype])
        return out[0]
    }
}
