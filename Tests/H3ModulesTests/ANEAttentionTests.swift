// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import H3ANEBridge
@testable import H3Attention

@Suite("ANE fused attention", .serialized, .enabled(if: h3_ane_is_available()))
struct ANEAttentionTests {
    private func relRMS(_ reference: MLXArray, _ actual: MLXArray) -> Float {
        let r = reference.asType(.float32), a = actual.asType(.float32)
        let d = r - a
        return MLX.sqrt(MLX.sum(d * d)).item(Float.self)
            / max(MLX.sqrt(MLX.sum(r * r)).item(Float.self), 1e-30)
    }

    @Test
    func completeHeadSplitMatchesDenseSDPA() throws {
        let heads = 9, sequence = 64, dimension = 128
        var state: UInt64 = 0x9e3779b97f4a7c15
        func values(scale: Float) -> [Float] {
            (0 ..< heads * sequence * dimension).map { _ in
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let unit = Float(Int32(truncatingIfNeeded: state >> 32))
                    / Float(Int32.max)
                return unit * scale
            }
        }
        let q = MLXArray(values(scale: 0.3), [heads, sequence, dimension])
            .asType(.bfloat16)
        let k = MLXArray(values(scale: 0.3), [heads, sequence, dimension])
            .asType(.bfloat16)
        let v = MLXArray(values(scale: 0.2), [heads, sequence, dimension])
            .asType(.bfloat16)
        let scale = 1 / Float(dimension).squareRoot()
        MLX.eval(q, k, v)

        let reference = MLXFast.scaledDotProductAttention(
            queries: q.expandedDimensions(axis: 0),
            keys: k.expandedDimensions(axis: 0),
            values: v.expandedDimensions(axis: 0),
            scale: scale, mask: nil).squeezed(axis: 0)
        let context = AttentionContext(
            blockIndex: 0, blockCount: 1, scheduleProgress: 0,
            sequenceLength: sequence, videoSpan: nil)
        let actual = try #require(ANEAttentionBackend().attend(
            queries: q, keys: k, values: v, scale: scale,
            mask: nil, context: context))
        MLX.eval(reference, actual)

        let error = relRMS(reference, actual)
        #expect(error < ANEAttentionBackend.equivalenceClass,
                "complete-head route differs from dense SDPA by relRMS \(error)")
    }

    /// Real captured tensors are the production gate. Random q/k/v pin layout
    /// and operator arithmetic, but not H3's score distribution or magnitude.
    ///
    ///     H3_ANE_QKV=/path/to/qkv_block24_p0.25.safetensors \
    ///       swift test -c release --filter realCaptureConformsAndTimes
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_ANE_QKV"] != nil))
    func realCaptureConformsAndTimes() throws {
        let path = try #require(ProcessInfo.processInfo.environment["H3_ANE_QKV"])
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let q0 = try #require(arrays["q"]), k0 = try #require(arrays["k"])
        let v0 = try #require(arrays["v"])
        // Captures are [S,H,D]; the backend contract is [H,S,D].
        let q = MLX.contiguous(q0.transposed(1, 0, 2).asType(.bfloat16))
        let k = MLX.contiguous(k0.transposed(1, 0, 2).asType(.bfloat16))
        let v = MLX.contiguous(v0.transposed(1, 0, 2).asType(.bfloat16))
        let scale = 1 / Float(q.dim(2)).squareRoot()
        MLX.eval(q, k, v)
        let context = AttentionContext(
            blockIndex: 24, blockCount: 50, scheduleProgress: 0.25,
            sequenceLength: q.dim(1), videoSpan: nil)

        func route() throws -> MLXArray {
            let value = try #require(ANEAttentionBackend().attend(
                queries: q, keys: k, values: v, scale: scale,
                mask: nil, context: context))
            MLX.eval(value)
            return value
        }
        func dense() -> MLXArray {
            let value = MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0),
                keys: k.expandedDimensions(axis: 0),
                values: v.expandedDimensions(axis: 0),
                scale: scale, mask: nil).squeezed(axis: 0)
            MLX.eval(value)
            return value
        }

        // The first call compiles and loads both graphs. Production pays that
        // once per shape, so the steady state is what the route is worth.
        let coldBegan = Date()
        _ = try route()
        let cold = Date().timeIntervalSince(coldBegan)
        _ = dense()

        func time(_ body: () throws -> MLXArray) rethrows -> Double {
            var best = Double.infinity
            for _ in 0 ..< 3 {
                let began = Date()
                _ = try body()
                best = min(best, Date().timeIntervalSince(began))
            }
            return best
        }
        let routed = try time(route)
        let denseSeconds = time(dense)
        let actual = try route()
        let reference = dense()
        let error = relRMS(reference, actual)
        print(String(format: "\nANE attention real capture S=%d H=%d rel_rms=%.6g cold=%.0fms routed=%.1fms dense=%.1fms speedup=%.3fx",
                     q.dim(1), q.dim(0), error, cold * 1e3,
                     routed * 1e3, denseSeconds * 1e3, denseSeconds / routed))
        #expect(error < ANEAttentionBackend.equivalenceClass)
    }
}
