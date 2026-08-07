// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import H3Foundation
@testable import H3Modules

/// Ceiling test for fusing the per-head Q/K RMSNorm and split-half RoPE.
///
/// This measures the complete production chain before writing a Metal kernel.
/// A perfect fused implementation cannot save more than the unfused chain's
/// entire wall time, so a ceiling below the 5% gate closes the avenue without
/// acquiring a second reduction implementation and its equivalence burden.
///
///     H3_BIG=1 swift test --filter qkNormRoPECeiling
@Suite("QK RMSNorm RoPE fusion", .serialized)
struct QKFusionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func qkNormRoPECeiling() {
        let cfg = H3Config()
        let s = 15_731
        MLXRandom.seed(11)

        let qkv = (MLXRandom.normal([s, 3 * cfg.innerDim]) * 0.02).asType(.bfloat16)
        let qNorm = H3RMSNorm(
            weight: (MLXRandom.normal([cfg.headDim]) * 0.02).asType(.bfloat16),
            eps: cfg.qkNormEps)
        let kNorm = H3RMSNorm(
            weight: (MLXRandom.normal([cfg.headDim]) * 0.02).asType(.bfloat16),
            eps: cfg.qkNormEps)
        let rope = H3RoPE.rotationTable(
            angles: MLXRandom.normal([s, 96]).asType(.float32)).asType(.bfloat16)
        MLX.eval(qkv, qNorm.weight, kNorm.weight, rope)

        let parts = qkv.split(parts: 3, axis: -1)
        let shape = [s, cfg.numHeads, cfg.headDim]
        let q = parts[0].reshaped(shape)
        let k = parts[1].reshaped(shape)

        let normOnly = { [qNorm(q), kNorm(k)] }
        let ropeOnly = {
            [SplitHalfRoPE.apply(q, table: rope), SplitHalfRoPE.apply(k, table: rope)]
        }
        let complete = {
            [SplitHalfRoPE.apply(qNorm(q), table: rope),
             SplitHalfRoPE.apply(kNorm(k), table: rope)]
        }
        let fused = {
            let out = FusedQKRoPE.apply(qkv: qkv, qWeight: qNorm.weight,
                                        kWeight: kNorm.weight, rope: rope,
                                        heads: cfg.numHeads, headDim: cfg.headDim,
                                        eps: cfg.qkNormEps)!
            return [out.q, out.k]
        }

        let normMs = BenchmarkSupport.medianArrays(normOnly) * 1000
        let ropeMs = BenchmarkSupport.medianArrays(ropeOnly) * 1000
        let measured = BenchmarkSupport.interleavedArrays(first: complete, second: fused)
        let completeMs = measured.first * 1000
        let fusedMs = measured.second * 1000

        let expected = complete(), actual = fused()
        MLX.eval(expected)
        MLX.eval(actual)
        var worstRel: Float = 0
        var worstUlp: Float = 0
        var identical = true
        for (a, b) in zip(expected, actual) {
            let a32 = a.asType(.float32)
            let delta = a32 - b.asType(.float32)
            let rel = MLX.sqrt(MLX.mean(delta * delta)).item(Float.self)
                / MLX.sqrt(MLX.mean(a32 * a32)).item(Float.self)
            worstRel = max(worstRel, rel)
            let rms = MLX.sqrt(MLX.mean(a32 * a32)).item(Float.self)
            let scaled = MLX.abs(delta) / (MLX.abs(a32) + MLXArray(rms))
            worstUlp = max(worstUlp,
                           MLX.max(scaled).item(Float.self) / 0.0078125)
            identical = identical && MLX.all(a .== b).item(Bool.self)
        }

        // Cap 5 runs block 0 on all twenty steps and the remaining 49 blocks on
        // ten full steps. Treating a fused kernel as free is the strict upper
        // bound; real dispatch, loads, reduction and writes can only save less.
        let calls = 20 + (cfg.numLayers - 1) * 10
        let denseStepCeiling = completeMs * Double(cfg.numLayers) / 1000
        let cachedRenderCeiling = completeMs * Double(calls) / 1000
        let denseStepPercent = 100 * denseStepCeiling / 60.0
        let cachedRenderPercent = 100 * cachedRenderCeiling / 660.5
        let denseStepSaving = (completeMs - fusedMs) * Double(cfg.numLayers) / 1000
        let cachedRenderSaving = (completeMs - fusedMs) * Double(calls) / 1000

        print(String(format: "\n  Q/K production chain S=%d H=%d D=%d",
                     s, cfg.numHeads, cfg.headDim))
        print(String(format: "  RMSNorm only %.1f ms; RoPE only %.1f ms; complete %.1f ms",
                     normMs, ropeMs, completeMs))
        print(String(format: "  fused %.1f ms   %.2fx   rel RMS %.3e   worst %.2f ulp   identical %@",
                     fusedMs, completeMs / fusedMs, worstRel, worstUlp,
                     (identical ? "yes" : "no") as NSString))
        print(String(format: "  impossible-free-kernel ceiling: %.2f s/full step (%.2f%%)",
                     denseStepCeiling, denseStepPercent))
        print(String(format: "  cap-5 render ceiling: %.2f s of 660.5 s (%.2f%%)\n",
                     cachedRenderCeiling, cachedRenderPercent))
        print(String(format: "  measured saving: %.2f s/full step; %.2f s/cap-5 render\n",
                     denseStepSaving, cachedRenderSaving))

        #expect(completeMs.isFinite && completeMs > 0)
        #expect(worstRel < 1e-3, "fused Q/K path exceeded the block equivalence band")
        #expect(worstUlp < 8, "fused Q/K path has a localized error beyond reduction order")
    }

    @Test("unsupported geometry declines instead of indexing it as H3")
    func declinesUnsupportedGeometry() {
        let qkv = MLXArray.zeros([4, 3 * 2 * 64]).asType(.bfloat16)
        let weight = MLXArray.ones([64]).asType(.bfloat16)
        let rope = H3RoPE.rotationTable(
            angles: MLXArray.zeros([4, 64]).asType(.float32)).asType(.bfloat16)
        #expect(FusedQKRoPE.apply(qkv: qkv, qWeight: weight, kWeight: weight,
                                 rope: rope, heads: 2, headDim: 64,
                                 eps: 1e-5) == nil)
    }
}
