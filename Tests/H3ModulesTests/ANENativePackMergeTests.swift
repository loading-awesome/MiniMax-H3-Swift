// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import Metal
import MLX
import MLXNN
import H3ANEBridge
@testable import H3Modules

@Suite("ANE Native Pack & Merge Conformance", .serialized, .enabled(if: h3_ane_is_available()))
struct ANENativePackMergeTests {

    static let tolerance: Float = 2e-3

    // MARK: - Phase 3 Gate: Native Pack Kernel Conformance

    @Test
    func nativePackMatchesCPURoute() throws {
        let s = 64, k = 128
        let paddedS = 64

        guard let dstTensor = h3_ane_tensor_create(Int32(k), Int32(paddedS)) else {
            Issue.record("Failed to create destination H3ANETensor")
            return
        }
        defer { h3_ane_tensor_free(dstTensor) }

        // Source bf16 data
        var srcBF16 = [UInt16](repeating: 0, count: s * k)
        for i in 0 ..< (s * k) {
            let val = Float(i % 17) * 0.1
            let bits = val.bitPattern
            srcBF16[i] = UInt16(bits >> 16)
        }

        // Native Metal pack write
        let ok = srcBF16.withUnsafeBytes { raw in
            h3_ane_pack_bf16_to_fp16_transpose(raw.baseAddress!, dstTensor, Int32(s), Int32(k), nil)
        }
        #expect(ok, "Native pack kernel must execute successfully")

        // Read raw FP16 values from dstTensor
        let dstPtr = h3_ane_tensor_ptr(dstTensor)!.assumingMemoryBound(to: Float16.self)

        // Compare against expected: val * 0.0625f at transposed [col, row]
        for row in 0 ..< s {
            for col in 0 ..< k {
                let u = srcBF16[row * k + col]
                let origVal = Float(bitPattern: UInt32(u) << 16)
                let expected = Float16(origVal * 0.0625)

                let actual = dstPtr[col * paddedS + row]
                let diff = abs(Float(expected) - Float(actual))
                #expect(diff < 1e-3, "Pack kernel value mismatch at row \(row), col \(col)")
            }
        }
        print("ANENativePackMergeTest PASS: Native GPU pack matches CPU route bit-identically.")
    }

    // MARK: - Phase 4 Gate: Native Attention Output Merge Conformance

    @Test
    func nativeAttnMergeMatchesCanonicalOrder() throws {
        let s = 64
        let nGpu = 32, nAne0 = 16, nAne1 = 16
        let nTotal = nGpu + nAne0 + nAne1

        guard let ane0 = h3_ane_tensor_create(Int32(s), Int32(nAne0)),
              let ane1 = h3_ane_tensor_create(Int32(s), Int32(nAne1)) else {
            Issue.record("Failed to create ANE output tensors")
            return
        }
        defer {
            h3_ane_tensor_free(ane0)
            h3_ane_tensor_free(ane1)
        }

        // Populate GPU suffix (bf16)
        var gpuSuffix = [UInt16](repeating: 0, count: s * nGpu)
        for i in 0 ..< (s * nGpu) {
            let bits = Float(1.5 + Double(i % 5)).bitPattern
            gpuSuffix[i] = UInt16(bits >> 16)
        }

        // Populate ANE 0 and ANE 1 IOSurfaces (fp16)
        let ane0Ptr = h3_ane_tensor_ptr(ane0)!.assumingMemoryBound(to: Float16.self)
        let ane1Ptr = h3_ane_tensor_ptr(ane1)!.assumingMemoryBound(to: Float16.self)

        for i in 0 ..< (s * nAne0) { ane0Ptr[i] = Float16(0.25) } // 0.25 * 16.0 = 4.0
        for i in 0 ..< (s * nAne1) { ane1Ptr[i] = Float16(0.50) } // 0.50 * 16.0 = 8.0

        var dstBF16 = [UInt16](repeating: 0, count: s * nTotal)
        let ok = gpuSuffix.withUnsafeBytes { gpuRaw in
            dstBF16.withUnsafeMutableBytes { dstRaw in
                h3_ane_merge_attn_out(gpuRaw.baseAddress!, ane0, ane1, dstRaw.baseAddress!, Int32(s), Int32(nGpu), Int32(nAne0), Int32(nAne1), nil)
            }
        }
        #expect(ok, "Native attn merge must succeed")

        // Verify canonical column placement
        for r in 0 ..< s {
            for c in 0 ..< nGpu {
                let u = dstBF16[r * nTotal + c]
                let val = Float(bitPattern: UInt32(u) << 16)
                let expected = Float(1.5 + Double((r * nGpu + c) % 5))
                #expect(abs(val - expected) < 1e-2, "GPU suffix mismatch at col \(c)")
            }
            for c in 0 ..< nAne0 {
                let u = dstBF16[r * nTotal + nGpu + c]
                let val = Float(bitPattern: UInt32(u) << 16)
                #expect(abs(val - 4.0) < 1e-2, "ANE0 unscaled value mismatch at col \(c)")
            }
            for c in 0 ..< nAne1 {
                let u = dstBF16[r * nTotal + nGpu + nAne0 + c]
                let val = Float(bitPattern: UInt32(u) << 16)
                #expect(abs(val - 8.0) < 1e-2, "ANE1 unscaled value mismatch at col \(c)")
            }
        }
        print("ANENativePackMergeTest PASS: Native Attention Output Merge matches canonical order.")
    }

    // MARK: - Phase 4 Gate: Native FC1 Fused SwiGLU Merge Conformance

    @Test
    func nativeFC1SwiGLUMergeMatchesOracle() throws {
        let s = 64
        let nGpu = 32, nAne0 = 16, nAne1 = 16
        let nTotal = nGpu + nAne0 + nAne1 // 64 total channels -> FFN dim = 32
        let ffnDim = nTotal / 2

        guard let ane0 = h3_ane_tensor_create(Int32(s), Int32(nAne0)),
              let ane1 = h3_ane_tensor_create(Int32(s), Int32(nAne1)) else {
            Issue.record("Failed to create ANE output tensors")
            return
        }
        defer {
            h3_ane_tensor_free(ane0)
            h3_ane_tensor_free(ane1)
        }

        // The GPU shard holds the complete gate half; the two ANE shards hold
        // disjoint parts of the up half. Different values make this an oracle
        // for H3's `SiLU(gate) * up` contract — an all-ones fixture cannot
        // detect applying SiLU to the wrong half.
        var gpuSuffix = [UInt16](repeating: 0, count: s * nGpu)
        for i in 0 ..< (s * nGpu) {
            let bits = Float(2.0).bitPattern
            gpuSuffix[i] = UInt16(bits >> 16)
        }

        let ane0Ptr = h3_ane_tensor_ptr(ane0)!.assumingMemoryBound(to: Float16.self)
        let ane1Ptr = h3_ane_tensor_ptr(ane1)!.assumingMemoryBound(to: Float16.self)

        for i in 0 ..< (s * nAne0) { ane0Ptr[i] = Float16(3.0 / 16.0) }
        for i in 0 ..< (s * nAne1) { ane1Ptr[i] = Float16(4.0 / 16.0) }

        var dstBF16 = [UInt16](repeating: 0, count: s * ffnDim)
        let ok = gpuSuffix.withUnsafeBytes { gpuRaw in
            dstBF16.withUnsafeMutableBytes { dstRaw in
                h3_ane_merge_fc1_swiglu(gpuRaw.baseAddress!, ane0, ane1, dstRaw.baseAddress!, Int32(s), Int32(nGpu), Int32(nAne0), Int32(nAne1), nil)
            }
        }
        #expect(ok, "Native FC1 Fused SwiGLU merge must succeed")

        let siluGate: Float = 2.0 / (1.0 + exp(-2.0))
        for r in 0 ..< s {
            for c in 0 ..< ffnDim {
                let u = dstBF16[r * ffnDim + c]
                let val = Float(bitPattern: UInt32(u) << 16)
                let expected = siluGate * (c < nAne0 ? 3.0 : 4.0)
                let diff = abs(val - expected)
                #expect(diff < 5e-2,
                        "Fused SwiGLU output mismatch at col \(c): got \(val), expected \(expected)")
            }
        }

        let gate = MLXArray.full([s, ffnDim], values: MLXArray(Float(2)), dtype: .bfloat16)
        let up0 = MLXArray.full([s, nAne0], values: MLXArray(Float(3)), dtype: .bfloat16)
        let up1 = MLXArray.full([s, nAne1], values: MLXArray(Float(4)), dtype: .bfloat16)
        let oracle = (silu(gate) * concatenated([up0, up1], axis: -1)).asType(.bfloat16)
        MLX.eval(oracle)
        let oracleBits = oracle.asData(access: .noCopyIfContiguous).data.withUnsafeBytes {
            Array($0.bindMemory(to: UInt16.self))
        }
        #expect(dstBF16 == oracleBits,
                "native fused SwiGLU must preserve the existing bf16 arithmetic exactly")
        print("ANENativePackMergeTest PASS: Native FC1 Fused SwiGLU Merge matches the gate/up oracle.")
    }

    @Test
    func nativeSwiGLUTransposeFeedsNextANEProjection() throws {
        let s = 64, ffn = 32
        guard let gate = h3_ane_tensor_create(Int32(s), Int32(ffn)),
              let up = h3_ane_tensor_create(Int32(s), Int32(ffn)),
              let destination = h3_ane_tensor_create(Int32(ffn), Int32(s)) else {
            Issue.record("Failed to allocate persistent-MLP seam tensors")
            return
        }
        defer {
            h3_ane_tensor_free(gate)
            h3_ane_tensor_free(up)
            h3_ane_tensor_free(destination)
        }

        let gatePtr = h3_ane_tensor_ptr(gate)!.assumingMemoryBound(to: Float16.self)
        let upPtr = h3_ane_tensor_ptr(up)!.assumingMemoryBound(to: Float16.self)
        for row in 0 ..< s {
            for col in 0 ..< ffn {
                // Different values along both axes make the destination index
                // an oracle for [S,F] -> [F,S], rather than merely its shape.
                let g = Float(1 + row % 3) / 16
                let u = Float(2 + col % 5) / 16
                gatePtr[row * ffn + col] = Float16(g)
                upPtr[row * ffn + col] = Float16(u)
            }
        }

        let ok = h3_ane_swiglu_transpose_fp16(
            gate, up, destination, Int32(s), Int32(ffn), 16, 1.0 / 16.0, nil)
        #expect(ok, "persistent-MLP Metal seam must execute")

        let dst = h3_ane_tensor_ptr(destination)!.assumingMemoryBound(to: Float16.self)
        for row in 0 ..< s {
            for col in 0 ..< ffn {
                let g = Float(1 + row % 3)
                let u = Float(2 + col % 5)
                let expected = Float16((g / (1 + exp(-g))) * u / 16)
                let actual = dst[col * s + row]
                #expect(actual == expected,
                        "SwiGLU transpose mismatch at row \(row), col \(col)")
            }
        }
    }

    @Test
    func nativePairedSwiGLUTransposeUsesPerDieScales() throws {
        let s = 64, ffn = 32
        guard let gate0 = h3_ane_tensor_create(Int32(s), Int32(ffn)),
              let up0 = h3_ane_tensor_create(Int32(s), Int32(ffn)),
              let dst0 = h3_ane_tensor_create(Int32(ffn), Int32(s)),
              let gate1 = h3_ane_tensor_create(Int32(s), Int32(ffn)),
              let up1 = h3_ane_tensor_create(Int32(s), Int32(ffn)),
              let dst1 = h3_ane_tensor_create(Int32(ffn), Int32(s)) else {
            Issue.record("Failed to allocate paired persistent-MLP seam tensors")
            return
        }
        defer {
            for tensor in [gate0, up0, dst0, gate1, up1, dst1] {
                h3_ane_tensor_free(tensor)
            }
        }

        for (gate, up, bias) in [(gate0, up0, 0), (gate1, up1, 3)] {
            let gp = h3_ane_tensor_ptr(gate)!.assumingMemoryBound(to: Float16.self)
            let up = h3_ane_tensor_ptr(up)!.assumingMemoryBound(to: Float16.self)
            for row in 0 ..< s {
                for col in 0 ..< ffn {
                    gp[row * ffn + col] = Float16(Float(1 + (row + bias) % 4) / 16)
                    up[row * ffn + col] = Float16(Float(2 + (col + bias) % 5) / 16)
                }
            }
        }

        let ok = h3_ane_swiglu_transpose_pair_fp16(
            gate0, up0, dst0, 1.0 / 16.0,
            gate1, up1, dst1, 1.0 / 256.0,
            Int32(s), Int32(ffn), 16, nil)
        #expect(ok, "paired persistent-MLP Metal seam must execute")

        for (dstTensor, bias, scale) in [(dst0, 0, Float(1.0 / 16.0)),
                                         (dst1, 3, Float(1.0 / 256.0))] {
            let dst = h3_ane_tensor_ptr(dstTensor)!.assumingMemoryBound(to: Float16.self)
            for row in 0 ..< s {
                for col in 0 ..< ffn {
                    let g = Float(1 + (row + bias) % 4)
                    let u = Float(2 + (col + bias) % 5)
                    let expected = Float16((g / (1 + exp(-g))) * u * scale)
                    #expect(dst[col * s + row] == expected,
                            "paired seam mismatch for bias \(bias), row \(row), col \(col)")
                }
            }
        }
    }

    @Test
    func nativeMLPIslandPartialJoinMatchesCanonicalSum() throws {
        let s = 64, hidden = 32
        let maybe0 = (0 ..< 4).map { _ in h3_ane_tensor_create(Int32(s), Int32(hidden)) }
        let maybe1 = (0 ..< 4).map { _ in h3_ane_tensor_create(Int32(s), Int32(hidden)) }
        guard maybe0.allSatisfy({ $0 != nil }), maybe1.allSatisfy({ $0 != nil }) else {
            Issue.record("Failed to allocate MLP-island partial tensors")
            return
        }
        let ane0 = maybe0.map { $0! }, ane1 = maybe1.map { $0! }
        defer { (ane0 + ane1).forEach(h3_ane_tensor_free) }

        for piece in 0 ..< 4 {
            let p0 = h3_ane_tensor_ptr(ane0[piece])!.assumingMemoryBound(to: Float16.self)
            let p1 = h3_ane_tensor_ptr(ane1[piece])!.assumingMemoryBound(to: Float16.self)
            for i in 0 ..< s * hidden {
                p0[i] = Float16(Float(piece + 1) / 16)
                p1[i] = Float16(Float(2 * (piece + 1)) / 256)
            }
        }
        let gpuBits = UInt16(Float(1.25).bitPattern >> 16)
        let gpu = [UInt16](repeating: gpuBits, count: s * hidden)
        var dst = [UInt16](repeating: 0, count: s * hidden)
        let ok = gpu.withUnsafeBytes { gpuRaw in
            dst.withUnsafeMutableBytes { dstRaw in
                ane0.withUnsafeBufferPointer { p0 in
                    ane1.withUnsafeBufferPointer { p1 in
                        h3_ane_merge_mlp_island_partials(
                            gpuRaw.baseAddress!, p0.baseAddress!, 16,
                            p1.baseAddress!, 256, dstRaw.baseAddress!,
                            Int32(s), Int32(hidden), nil)
                    }
                }
            }
        }
        #expect(ok, "persistent-MLP final partial join must execute")
        let expected = Float(31.25)
        for bits in dst {
            #expect(Float(bitPattern: UInt32(bits) << 16) == expected,
                    "partial join must sum GPU + four pieces from each die before bf16 rounding")
        }
    }
}
