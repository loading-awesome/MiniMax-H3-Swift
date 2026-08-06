// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import MLXRandom
import H3Hardware
@testable import H3Attention

private func relRMS(_ a: MLXArray, _ b: MLXArray) -> Float {
    let d = (a.asType(.float32) - b.asType(.float32))
    let num = MLX.sqrt(MLX.mean(d * d)).item(Float.self)
    let den = MLX.sqrt(MLX.mean(b.asType(.float32) * b.asType(.float32))).item(Float.self)
    return num / den
}

private func dense(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
    MLXFast.scaledDotProductAttention(
        queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
        values: v.expandedDimensions(axis: 0), scale: scale, mask: nil).squeezed(axis: 0)
}

/// Block-contiguous structure, at head dim 128 — the shape H3 actually uses, so
/// the kernel is exercised at its real `DPT` of 32 rather than at a smaller one
/// that would hide an indexing mistake in the lane split.
private func structured(heads: Int, s: Int, d: Int, clusters: Int, seed: UInt64)
    -> (MLXArray, MLXArray, MLXArray) {
    MLXRandom.seed(seed)
    let centres = MLXRandom.normal([clusters, d]) * 0.8
    let ids = MLXArray((0 ..< s).map { Int32(($0 / 96) % clusters) })
    let k = centres[ids].reshaped([1, s, d]) + MLXRandom.normal([heads, s, d]) * 0.25
    let q = centres[ids].reshaped([1, s, d]) + MLXRandom.normal([heads, s, d]) * 0.25
    let v = MLXRandom.normal([heads, s, d])
    return (q, k, v)
}

@Suite("Sol-Attn Metal kernel")
struct SolAttnMetalTests {

    /// The kernel must reproduce the reference, which is the only definition of
    /// correct this backend has. Float32 throughout so the comparison measures
    /// the kernel rather than bf16 rounding.
    @Test func kernelMatchesTheReference() {
        let (q, k, v) = structured(heads: 4, s: 1024, d: 128, clusters: 8, seed: 21)
        let scale = 1.0 / Float(128).squareRoot()
        let config = SolAttnConfig(beta: 1.2, blockSize: 64, exactConditioningKV: false)

        let want = SolAttnReference.attend(queries: q, keys: k, values: v,
                                           scale: scale, config: config, videoSpan: nil)
        let got = try! #require(SolAttnMetalKernel.attend(
            queries: q, keys: k, values: v, scale: scale, config: config, videoSpan: nil))
        #expect(relRMS(got, want) < 2e-3)
    }

    /// With nothing rejected the kernel is doing dense attention the hard way,
    /// and must agree with the fused kernel that does it the easy way. This is
    /// the tile loop, the running max and the quad reduction all at once, with
    /// the correction switched out of the picture.
    @Test func allSelectedMatchesDense() {
        let (q, k, v) = structured(heads: 3, s: 512, d: 128, clusters: 4, seed: 22)
        let scale = 1.0 / Float(128).squareRoot()
        let config = SolAttnConfig(beta: -50, blockSize: 64, exactConditioningKV: false)

        let got = try! #require(SolAttnMetalKernel.attend(
            queries: q, keys: k, values: v, scale: scale, config: config, videoSpan: nil))
        #expect(relRMS(got, dense(q, k, v, scale: scale)) < 2e-3)
    }

    /// Every real H3 length leaves a short final block, and the kernel handles
    /// it in three separate places — the query rows that fall off the end, the
    /// key tile that is partly out of range, and the pooled block whose count is
    /// not `blockSize`. 1,061 leaves 37 rows in the last block and 5 in the last
    /// tile.
    @Test func shortTailMatchesTheReference() {
        let (q, k, v) = structured(heads: 2, s: 1061, d: 128, clusters: 5, seed: 23)
        let scale = 1.0 / Float(128).squareRoot()
        let config = SolAttnConfig(beta: 1.2, blockSize: 64, exactConditioningKV: false)

        let want = SolAttnReference.attend(queries: q, keys: k, values: v,
                                           scale: scale, config: config, videoSpan: nil)
        let got = try! #require(SolAttnMetalKernel.attend(
            queries: q, keys: k, values: v, scale: scale, config: config, videoSpan: nil))
        #expect(relRMS(got, want) < 2e-3)
    }

    /// The sink changes which blocks are exact, so it has to be checked through
    /// the kernel rather than only through the router.
    @Test func sinkMatchesTheReference() {
        let (q, k, v) = structured(heads: 2, s: 1024, d: 128, clusters: 6, seed: 24)
        let scale = 1.0 / Float(128).squareRoot()
        let config = SolAttnConfig(beta: 1.5, blockSize: 64, exactConditioningKV: true)
        let span = 192 ..< 1024

        let want = SolAttnReference.attend(queries: q, keys: k, values: v,
                                           scale: scale, config: config, videoSpan: span)
        let got = try! #require(SolAttnMetalKernel.attend(
            queries: q, keys: k, values: v, scale: scale, config: config, videoSpan: span))
        #expect(relRMS(got, want) < 2e-3)
    }

    /// A routing block larger than the kernel's threadgroup block.
    ///
    /// The two sizes are chosen for different reasons — 64 query rows to a
    /// threadgroup is a register-file decision, the routing block size is a
    /// selection-precision one — and they must not be welded together. At
    /// `blockSize` 128 each routing block spans two threadgroups, which is the
    /// case the `(qb * BQ) / BS` mapping exists for.
    ///
    /// The first version of this kernel asserted the two were independent in a
    /// comment and then required them to be equal in the guard immediately
    /// below it, so every `blockSize` but 64 fell through to dense attention
    /// while claiming to be sparse. Nothing failed; it just quietly stopped
    /// being Sol-Attn. Hence a test, rather than a fixed comment.
    @Test func routingBlockLargerThanThreadgroupBlock() {
        let (q, k, v) = structured(heads: 2, s: 1024, d: 128, clusters: 6, seed: 26)
        let scale = 1.0 / Float(128).squareRoot()
        let config = SolAttnConfig(beta: 1.2, blockSize: 128, exactConditioningKV: false)

        let want = SolAttnReference.attend(queries: q, keys: k, values: v,
                                           scale: scale, config: config, videoSpan: nil)
        let got = try! #require(SolAttnMetalKernel.attend(
            queries: q, keys: k, values: v, scale: scale, config: config, videoSpan: nil))
        #expect(relRMS(got, want) < 2e-3)
    }

    /// Same, at 256, and with a length that leaves a short tail in the larger
    /// block so the mapping is checked where it is least comfortable.
    @Test func routingBlockOfFourThreadgroups() {
        let (q, k, v) = structured(heads: 2, s: 1061, d: 128, clusters: 5, seed: 27)
        let scale = 1.0 / Float(128).squareRoot()
        let config = SolAttnConfig(beta: 1.0, blockSize: 256, exactConditioningKV: true)
        let span = 300 ..< 1061

        let want = SolAttnReference.attend(queries: q, keys: k, values: v,
                                           scale: scale, config: config, videoSpan: span)
        let got = try! #require(SolAttnMetalKernel.attend(
            queries: q, keys: k, values: v, scale: scale, config: config, videoSpan: span))
        #expect(relRMS(got, want) < 2e-3)
    }

    /// A routing block *smaller* than a threadgroup cannot work — the rows in
    /// one threadgroup would need two different selections — so it must decline
    /// and let the caller run dense, not produce a plausible wrong answer.
    @Test func routingBlockSmallerThanThreadgroupDeclines() {
        let (q, k, v) = structured(heads: 2, s: 512, d: 128, clusters: 4, seed: 28)
        let config = SolAttnConfig(beta: 1.2, blockSize: 32, exactConditioningKV: false)
        #expect(SolAttnMetalKernel.attend(queries: q, keys: k, values: v,
                                          scale: 0.088, config: config, videoSpan: nil) == nil)
    }

    /// bf16 input is what production passes. The kernel keeps its accumulators
    /// in float32 either way, so this checks the load path and — via the `T`
    /// template argument — that a bf16 call does not reuse the float32 kernel.
    @Test func bfloat16InputAgreesWithFloat32() {
        let (q, k, v) = structured(heads: 2, s: 512, d: 128, clusters: 4, seed: 25)
        let scale = 1.0 / Float(128).squareRoot()
        let config = SolAttnConfig(beta: 1.2, blockSize: 64, exactConditioningKV: false)

        let f32 = try! #require(SolAttnMetalKernel.attend(
            queries: q, keys: k, values: v, scale: scale, config: config, videoSpan: nil))
        let bf = try! #require(SolAttnMetalKernel.attend(
            queries: q.asType(.bfloat16), keys: k.asType(.bfloat16),
            values: v.asType(.bfloat16), scale: scale, config: config, videoSpan: nil))
        // bf16 carries seven mantissa bits; this is that, not a kernel disagreement.
        #expect(relRMS(bf, f32) < 3e-2)
    }
}

@Suite("Sol-Attn registration")
struct SolAttnRegistrationTests {

    private var machine: Machine {
        Machine(model: "Mac15,14", chip: "Apple M3 Ultra",
                memoryBytes: 275 * 1_000_000_000, cores: 28, isPortable: false)
    }

    /// `auto` must still resolve to dense. The backend is registered so it can
    /// be asked for, not so it can arrive by default — its equivalence class is
    /// carried over from a different implementation on different hardware and
    /// has not been measured here.
    @Test func autoStillResolvesToDense() throws {
        let s = try AttentionRegistry.resolve(requested: "auto", machine: machine)
        #expect(s.identifier == "sdpa")
    }

    /// And asking for it by name must hand back the real thing — the identifier
    /// and the instance, not a label copied onto an SDPA object.
    @Test func explicitRequestResolvesToSolAttn() throws {
        let s = try AttentionRegistry.resolve(requested: "sol", machine: machine)
        #expect(s.identifier == "sol")
        #expect(s.backend is SolAttnBackend)
        // The memory planner reads this and drops the quadratic activation term.
        #expect(s.materialisesScores == false)
    }

    /// Declining is how the schedule warm-up and the dense edge blocks are
    /// expressed, and the caller relies on `nil` meaning "run dense".
    @Test func declinesOutsideItsPolicy() {
        let b = SolAttnBackend()
        let q = MLXArray.zeros([2, 512, 128])
        func ctx(_ block: Int, _ progress: Double, _ len: Int) -> AttentionContext {
            AttentionContext(blockIndex: block, blockCount: 50, scheduleProgress: progress,
                             sequenceLength: len, videoSpan: 100 ..< len)
        }
        #expect(b.attend(queries: q, keys: q, values: q, scale: 0.088, mask: nil,
                         context: ctx(0, 0.5, 15_731)) == nil)      // first block
        #expect(b.attend(queries: q, keys: q, values: q, scale: 0.088, mask: nil,
                         context: ctx(25, 0.05, 15_731)) == nil)    // warm-up
        #expect(b.attend(queries: q, keys: q, values: q, scale: 0.088, mask: nil,
                         context: ctx(25, 0.5, 512)) == nil)        // too short
        // A mask is a contract this backend does not implement.
        #expect(b.attend(queries: q, keys: q, values: q, scale: 0.088,
                         mask: MLXArray.zeros([512, 512]),
                         context: ctx(25, 0.5, 15_731)) == nil)
    }
}
