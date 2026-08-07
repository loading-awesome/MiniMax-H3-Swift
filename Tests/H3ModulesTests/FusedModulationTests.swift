// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import H3Foundation
@testable import H3Modules

/// The fused kernels against the readable path they replace.
///
/// ## Two different bars, for a reason
///
/// The **gated residual has no reduction**, so it is held to bit-identity.
/// Every operation is elementwise, both paths round at the same points, and
/// any difference at all would mean a real bug.
///
/// The **norm cannot be bit-identical and should not be asked to be.** It sums
/// 5,376 squares, and MLX's reduction tree is not this kernel's; floating-point
/// addition is not associative, so the two sums differ in the last bits by
/// construction. Measured on the production hidden size: fp32 relative RMS
/// 4.3e-08 with 90.4% of elements bit-identical, bf16 relative RMS 2.6e-05 with
/// 99.9994% bit-identical — reduction-order noise, at the ulp scale, and no
/// more.
///
/// A tolerance alone would not settle that, because a tolerance cannot tell
/// "the sum was reassociated" from "the kernel is slightly wrong". So the norm
/// is checked against a **double-precision oracle computed on the CPU**, and the
/// claim asserted is the one that actually matters: the fused kernel is no
/// further from the true answer than the path it replaces. Fusion has to move
/// the error, not add to it.
///
/// The failure mode this suite exists for is neither: it is a wrong
/// *modulation row*, which produces output that is entirely plausible and
/// subtly wrong everywhere. `rowIndexingIsPerToken` is the test for that.
@Suite("fused modulation", .serialized)
struct FusedModulationTests {

    /// A packed sequence with several modality segments, like a real render's.
    /// A single-row index would pass every test here while a row-indexing bug
    /// sat untouched.
    private func index(tokens: Int, rows: Int) -> ModulationIndex {
        var segs: [ModSegment] = []
        let per = tokens / rows
        for r in 0 ..< rows {
            segs.append(ModSegment(start: r * per,
                                   stop: r == rows - 1 ? tokens : (r + 1) * per, row: r))
        }
        return ModulationIndex(segments: segs, tokenCount: tokens)
    }

    private func fixture(tokens: Int, hidden: Int, rows: Int, dtype: DType)
        -> (x: MLXArray, w: MLXArray, shift: MLXArray, scale: MLXArray,
            gate: MLXArray, other: MLXArray, idx: ModulationIndex) {
        MLXRandom.seed(20)
        func r(_ shape: [Int]) -> MLXArray { MLXRandom.normal(shape).asType(dtype) }
        return (r([tokens, hidden]), r([hidden]), r([rows, hidden]), r([rows, hidden]),
                r([rows, hidden]), r([tokens, hidden]), index(tokens: tokens, rows: rows))
    }

    private func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
        MLX.eval(a, b)
        return MLX.all(a .== b).item(Bool.self)
    }

    @Test("fusing the norm moves the rounding error, it does not add any")
    func normIsNoWorseThanUnfused() throws {
        // The oracle: the same arithmetic in Double on the CPU, where the
        // 5,376-term sum is accurate enough that both GPU paths can be measured
        // against it rather than against each other. Comparing the two paths
        // alone can only say they differ; it cannot say which is closer, and
        // "they differ" is the expected outcome of reassociating a sum.
        let tokens = 24, hidden = 5_376, rows = 6
        let f = fixture(tokens: tokens, hidden: hidden, rows: rows, dtype: .float32)
        let eps: Float = 1e-6
        let norm = H3RMSNorm(weight: f.w, eps: eps)
        let unfused = modScaleShift(norm(f.x), shift: f.shift, scale: f.scale, index: f.idx)
        let fused = try #require(FusedModulation.modulatedRMSNorm(
            f.x, weight: f.w, eps: eps, shift: f.shift, scale: f.scale, index: f.idx))
        MLX.eval(fused, unfused)

        let x = f.x.asArray(Float.self), w = f.w.asArray(Float.self)
        let scale = f.scale.asArray(Float.self), shift = f.shift.asArray(Float.self)
        let rowOf = f.idx.rows.asType(.int32).asArray(Int32.self)
        var oracle = [Double](repeating: 0, count: tokens * hidden)
        for t in 0 ..< tokens {
            var sum = 0.0
            for i in 0 ..< hidden {
                let v = Double(x[t * hidden + i]); sum += v * v
            }
            let inv = 1.0 / (sum / Double(hidden) + Double(eps)).squareRoot()
            let m = Int(rowOf[t]) * hidden
            for i in 0 ..< hidden {
                oracle[t * hidden + i] = Double(x[t * hidden + i]) * inv * Double(w[i])
                    * (1.0 + Double(scale[m + i])) + Double(shift[m + i])
            }
        }

        func distance(_ a: MLXArray) -> Double {
            let v = a.asArray(Float.self)
            var num = 0.0, den = 0.0
            for i in 0 ..< v.count {
                let d = Double(v[i]) - oracle[i]
                num += d * d; den += oracle[i] * oracle[i]
            }
            return (num / den).squareRoot()
        }
        let fusedError = distance(fused), unfusedError = distance(unfused)
        print(String(format: "  fused %.3e vs unfused %.3e from the double-precision oracle",
                     fusedError, unfusedError))
        // A 20% allowance on the comparison, not on the error: both paths sit
        // at the fp32 noise floor and which one wins is down to the fixture.
        // What would fail here is a kernel that is *systematically* worse —
        // a bf16 accumulation, a dropped tail element, a missing eps.
        #expect(fusedError <= unfusedError * 1.2,
                "the fused kernel is further from the truth than the path it replaces")
        #expect(fusedError < 1e-6, "fp32 reduction should land at the fp32 noise floor")
    }

    @Test("the two paths differ only at the last bits", arguments: [DType.float32, .bfloat16])
    func normDiffersOnlyByRounding(dtype: DType) throws {
        // The companion to the oracle test: bounded, per element, so that a
        // difference confined to a handful of tokens — a ragged tail read
        // wrongly, one modulation row off — cannot hide behind a whole-tensor
        // RMS that stays small because everything else agrees.
        let f = fixture(tokens: 96, hidden: 5_376, rows: 6, dtype: dtype)
        let norm = H3RMSNorm(weight: f.w, eps: 1e-6)
        let want = modScaleShift(norm(f.x), shift: f.shift, scale: f.scale, index: f.idx)
        let got = try #require(FusedModulation.modulatedRMSNorm(
            f.x, weight: f.w, eps: 1e-6, shift: f.shift, scale: f.scale, index: f.idx))
        MLX.eval(got, want)
        // Mantissa bits: fp32 has 24, bf16 has 8. A few ulps of headroom covers
        // the rounding chain after the reduction; an actual error is orders of
        // magnitude larger than this, not a factor of two.
        let ulp: Float = dtype == .float32 ? 1.2e-7 : 7.9e-3
        let w32 = want.asType(.float32)
        let d = MLX.abs(got.asType(.float32) - w32)
        // The denominator carries the tensor's own scale as well as the
        // element's. `h * (1 + scale) + shift` cancels: a result near zero came
        // from operands of order one, and its error is an ulp of *those*, not
        // of the near-zero answer. Dividing by |want| alone would report a
        // cancelled element as thousands of ulps out and say nothing.
        let rms = MLX.sqrt(MLX.mean(w32 * w32)).item(Float.self)
        let worst = MLX.max(d / (MLX.abs(w32) + MLXArray(rms))).item(Float.self) / ulp
        print("  \(dtype): worst element deviation \(worst) ulp (rms \(rms))")
        // Measured at 3.0 ulp in fp32 and 0.6 in bf16 on this fixture. Eight
        // leaves room for the fixture without leaving room for a defect: a
        // dropped tail element or a misread modulation row lands thousands of
        // ulps out, not double.
        #expect(worst < 8, "\(dtype): \(worst) ulp is more than reduction order explains")
    }

    @Test("the fused gated residual is bit-identical", arguments: [DType.float32, .bfloat16])
    func gateMatches(dtype: DType) throws {
        let f = fixture(tokens: 96, hidden: 5_376, rows: 6, dtype: dtype)
        let want = modGate(f.x, gate: f.gate, other: f.other, index: f.idx)
        let got = try #require(FusedModulation.gatedResidual(
            f.x, gate: f.gate, other: f.other, index: f.idx))
        #expect(identical(got, want))
    }

    @Test("a hidden size the thread count does not divide still matches")
    func raggedHiddenWidth() throws {
        // 5,376 happens to be 256 x 21. Nothing in the kernel may depend on
        // that, or the first checkpoint with a different hidden size produces
        // a truncated row and no error.
        for hidden in [1, 255, 257, 1_000] {
            let f = fixture(tokens: 12, hidden: hidden, rows: 3, dtype: .float32)
            let norm = H3RMSNorm(weight: f.w, eps: 1e-6)
            let want = modScaleShift(norm(f.x), shift: f.shift, scale: f.scale, index: f.idx)
            let got = try #require(FusedModulation.modulatedRMSNorm(
                f.x, weight: f.w, eps: 1e-6, shift: f.shift, scale: f.scale, index: f.idx),
                "declined at hidden \(hidden)")
            MLX.eval(got, want)
            // Per element, not an RMS. A tail element skipped by a strided loop
            // is a large error on a few columns, which an RMS over a wide row
            // would average into nothing.
            let w32 = want.asType(.float32)
            let d = MLX.abs(got.asType(.float32) - w32)
            let rms = MLX.sqrt(MLX.mean(w32 * w32)).item(Float.self)
            let worst = MLX.max(d / (MLX.abs(w32) + MLXArray(rms))).item(Float.self) / 1.2e-7
            #expect(worst < 8, "hidden \(hidden): worst deviation \(worst) ulp")
        }
    }

    @Test("every token reads its own modulation row")
    func rowIndexingIsPerToken() throws {
        // The failure this suite exists for. With the tables made constant per
        // row and the norm neutralised, the output *is* the row index, so a
        // kernel that read row 0 for everything — or that indexed by
        // threadgroup instead of by token — is visible directly rather than as
        // a small numerical drift.
        let tokens = 64, hidden = 128, rows = 8
        let x = MLXArray.ones([tokens, hidden])
        let w = MLXArray.ones([hidden])
        let zero = MLXArray.zeros([rows, hidden])
        // shift[r] = r, scale = 0, so out[i] = rms(1)*1*(1+0) + rows[i].
        let shift = MLX.broadcast(MLXArray((0 ..< rows).map { Float($0) }).reshaped([rows, 1]),
                                  to: [rows, hidden])
        let idx = index(tokens: tokens, rows: rows)
        let got = try #require(FusedModulation.modulatedRMSNorm(
            x, weight: w, eps: 1e-6, shift: shift, scale: zero, index: idx))
        let want = modScaleShift(H3RMSNorm(weight: w, eps: 1e-6)(x),
                                 shift: shift, scale: zero, index: idx)
        #expect(identical(got, want))
        // And spelled out, so a change to both paths at once cannot pass.
        MLX.eval(got)
        for r in 0 ..< rows {
            let token = r * (tokens / rows)
            #expect(abs(got[token, 0].item(Float.self) - (1.0 + Float(r))) < 1e-5,
                    "token \(token) read the wrong modulation row")
        }
    }

    @Test("shapes the kernel was not written for are declined, not guessed at")
    func declinesUnsupportedShapes() {
        let f = fixture(tokens: 12, hidden: 64, rows: 3, dtype: .float32)
        // The refiner's batched text path.
        let batched = f.x.expandedDimensions(axis: 0)
        #expect(FusedModulation.modulatedRMSNorm(batched, weight: f.w, eps: 1e-6,
                                                 shift: f.shift, scale: f.scale,
                                                 index: f.idx) == nil)
        // A dtype mix. The kernel has one element type; silently reinterpreting
        // a bf16 table as float32 would read garbage at plausible magnitudes.
        #expect(FusedModulation.modulatedRMSNorm(f.x, weight: f.w, eps: 1e-6,
                                                 shift: f.shift.asType(.bfloat16),
                                                 scale: f.scale, index: f.idx) == nil)
        // An index that does not cover the tokens: the kernel would read past
        // the table, and Metal does not trap on that — it returns whatever was
        // in memory.
        #expect(FusedModulation.modulatedRMSNorm(f.x, weight: f.w, eps: 1e-6,
                                                 shift: f.shift, scale: f.scale,
                                                 index: index(tokens: 6, rows: 3)) == nil)
        #expect(FusedModulation.gatedResidual(f.x, gate: f.gate,
                                              other: f.other.expandedDimensions(axis: 0),
                                              index: f.idx) == nil)
    }

    @Test("production shape dispatches and agrees",
          .enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func productionShape() throws {
        // 15,731 tokens x 5,376 hidden — the real packed sequence at
        // 864x480x124. The unit tests above run at 96 tokens, which exercises
        // the arithmetic but not the dispatch: the norm launches 15,731
        // threadgroups here and the gate launches 84.6 M threads in one grid
        // dimension. Neither is exotic, and neither had been run before two
        // hours of renders were about to depend on it.
        //
        //     H3_BIG=1 swift test --filter productionShape
        let tokens = 15_731, hidden = 5_376
        MLXRandom.seed(1)
        let x = MLXRandom.normal([tokens, hidden]).asType(.bfloat16)
        let w = MLXRandom.normal([hidden]).asType(.bfloat16)
        let shift = MLXRandom.normal([9, hidden]).asType(.bfloat16)
        let scale = MLXRandom.normal([9, hidden]).asType(.bfloat16)
        let idx = index(tokens: tokens, rows: 9)

        let got = try #require(FusedModulation.modulatedRMSNorm(
            x, weight: w, eps: 1e-6, shift: shift, scale: scale, index: idx))
        let want = modScaleShift(H3RMSNorm(weight: w, eps: 1e-6)(x),
                                 shift: shift, scale: scale, index: idx)
        MLX.eval(got, want)
        let d = MLX.abs(got.asType(.float32) - want.asType(.float32))
        let w32 = want.asType(.float32)
        let rms = MLX.sqrt(MLX.mean(w32 * w32)).item(Float.self)
        let worst = MLX.max(d / (MLX.abs(w32) + MLXArray(rms))).item(Float.self) / 7.9e-3
        print("  norm at production shape: worst \(worst) ulp")
        #expect(worst < 8)

        let other = MLXRandom.normal([tokens, hidden]).asType(.bfloat16)
        let gate = MLXRandom.normal([9, hidden]).asType(.bfloat16)
        let gGot = try #require(FusedModulation.gatedResidual(
            x, gate: gate, other: other, index: idx))
        let gWant = modGate(x, gate: gate, other: other, index: idx)
        #expect(identical(gGot, gWant))
    }

    @Test("a whole block gives the same answer with the fusion as without")
    func blockLevelAgreement() throws {
        // The kernels are correct in isolation above. This checks they are
        // wired into `DiTBlock` in the right places and the right order —
        // norm1 before the attention, norm2 after the first residual — which
        // no amount of kernel testing can establish.
        MLXRandom.seed(3)
        let tokens = 96, hidden = 256, heads = 4, headDim = 64, ffn = 512, rowCount = 6
        func r(_ shape: [Int]) -> MLXArray { MLXRandom.normal(shape).asType(.float32) * 0.05 }

        // `fuseModulation: true` explicitly. The default is off — it did not
        // clear its gate — and a block built at the default would compare the
        // readable path against itself and pass having tested nothing.
        let block = DiTBlock(
            norm1: H3RMSNorm(weight: r([hidden]), eps: 1e-6),
            norm2: H3RMSNorm(weight: r([hidden]), eps: 1e-6),
            attn: AttentionLayer(qkvWeight: r([3 * heads * headDim, hidden]),
                                 outWeight: r([hidden, heads * headDim]),
                                 qNormWeight: r([headDim]), kNormWeight: r([headDim]),
                                 heads: heads, headDim: headDim, eps: 1e-6),
            mlp: H3MLP(fc1: r([2 * ffn, hidden]), fc2: r([hidden, ffn])),
            adaln: AdalnProj(weight: r([6 * hidden * 3, 32]), bias: nil,
                             expand: 6, modalities: 3, hidden: hidden),
            fuseModulation: true)

        let x = r([tokens, hidden])
        let tEmb = r([rowCount / 3, 32])
        let idx = index(tokens: tokens, rows: rowCount)
        let fused = block(x, tEmb: tEmb, index: idx, ropeTable: nil)

        // The unfused reference, spelled out here rather than reached through
        // an environment variable, because `FusedModulation.enabled` is read
        // once at static-initialisation time and cannot be toggled mid-process.
        let m = block.adaln(tEmb)
        let h1 = modScaleShift(block.norm1(x), shift: m[0], scale: m[1], index: idx)
        let x1 = modGate(x, gate: m[2], other: block.attn(h1, ropeTable: nil), index: idx)
        let h2 = modScaleShift(block.norm2(x1), shift: m[3], scale: m[4], index: idx)
        let want = modGate(x1, gate: m[5], other: block.mlp(h2), index: idx)

        MLX.eval(fused, want)
        // Not bit-identity: two RMSNorm reductions sit inside, and their
        // last-bit disagreement is then multiplied through an attention and an
        // MLP. The bar is that it stays at that scale — a wiring mistake
        // (norm2 fed the wrong residual, the gates swapped) lands at O(1).
        let d = fused.asType(.float32) - want.asType(.float32)
        let relRMS = MLX.sqrt(MLX.mean(d * d)).item(Float.self)
            / MLX.sqrt(MLX.mean(want.asType(.float32) * want.asType(.float32))).item(Float.self)
        #expect(relRMS < 1e-5, "block-level relative RMS \(relRMS)")
        // And the fused block must not be the unfused one wearing a label: if
        // these were bit-identical the test above would pass with the fusion
        // switched off, which is how it read for one commit.
        #expect(relRMS > 0, "the fused block produced the unfused result exactly, so the kernel is not being reached")
    }

    @Test("the shipping default is the readable path")
    func defaultIsUnfused() {
        // The gate was 5%; it measured 1.81% on wall clock and 0.94% on the
        // steps it touches. Asserted rather than left to a doc comment, because
        // a default that drifts back on would change every render's pixels for
        // one percent.
        #expect(FusedModulation.enabled == false
                || ProcessInfo.processInfo.environment["H3_FUSED_MODULATION"] == "1")
    }
}
