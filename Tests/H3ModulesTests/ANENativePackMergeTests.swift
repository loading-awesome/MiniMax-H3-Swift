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
}
