// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import Metal
import MetalPerformanceShadersGraph
import MLX
import MLXNN
import MLXFast
import H3Foundation
@testable import H3Attention
@testable import H3Modules

/// Does the 1.13x survive contact with the block?
///
/// `mpsGraphBF16` measured MPSGraph 1.11–1.22x faster than MLX at every
/// production GEMM shape, bit-identical, in **isolation**. Isolation is doing a
/// lot of work in that sentence. A production path would cross between the two
/// runtimes four times a block and fifty blocks a step, and each crossing makes
/// MLX materialise a lazy graph it would otherwise have kept fusing and
/// scheduling. The isolated win is 98 ms a block; the crossing moves ~2.7 GB a
/// block if both directions copy. Those are the same order of magnitude, which
/// is the whole reason this test exists rather than a patch.
///
///     H3_BIG=1 swift test --filter mpsIntegration
///
/// **This models the block rather than editing it.** Both arms run the same
/// modelled forward — the real `H3RMSNorm`, `modScaleShift`, `modGate`,
/// `SplitHalfRoPE` and MLX's SDPA, in the real order — and differ only in who
/// multiplies. `blockModelIsFaithful` checks the model against the real
/// `DiTBlock` before any of it is believed: a model that is not the block
/// prices something nobody runs, which is how 6B happened.
@Suite("MPS integration", .serialized)
struct MPSIntegrationTests {

    static let s = 15_731                               // 864x480x124, the control shape

    // MARK: - The two multipliers

    /// `matmul(x, w.T)` — the shape every projection in the block has.
    protocol Projection {
        func callAsFunction(_ x: MLXArray) -> MLXArray
    }

    struct MLXProjection: Projection {
        let weight: MLXArray                            // [N, K], production layout
        func callAsFunction(_ x: MLXArray) -> MLXArray { matmul(x, weight.T) }
    }

    /// The same projection through `MPSGraph`, priced honestly.
    ///
    /// Two things here are deliberate and both flatter MPS, because the
    /// question is whether the win *can* survive rather than whether a naive
    /// port would lose it:
    ///
    ///  * **The weight is uploaded once**, transposed, as a graph constant. A
    ///    production integration would do exactly this — weights are fixed for a
    ///    render — so paying it per call would be pricing a strawman.
    ///  * **The activation crosses `noCopy`.** `asMTLBuffer(noCopy: true)`
    ///    wraps MLX's own backing store when it is contiguous, so the inbound
    ///    direction is free.
    ///
    /// What is *not* excused: `asMTLBuffer` calls `eval()`, so MLX drains
    /// before every GEMM, and the result is copied back into MLX because the
    /// public API offers no way to adopt an `MTLBuffer` without one. Those two
    /// are the integration cost, and they are what is being measured.
    final class MPSProjection: Projection {
        private let graph = MPSGraph()
        private let input: MPSGraphTensor
        private let output: MPSGraphTensor
        private let result: MTLBuffer
        private let device: any MTLDevice
        private let queue: any MTLCommandQueue
        private let m: Int, k: Int, n: Int

        init(weight: MLXArray, rows: Int, device: any MTLDevice,
             queue: any MTLCommandQueue) throws {
            self.device = device
            self.queue = queue
            self.m = rows
            self.n = weight.dim(0)                      // [N, K] -> matmul wants [K, N]
            self.k = weight.dim(1)

            // Transposed once, on the GPU, into a contiguous buffer. `.T` alone
            // is a strided view and its bytes are not the layout MPS reads.
            let transposed = weight.T.contiguous()
            MLX.eval(transposed)
            let bytes = transposed.asData(access: .copy).data

            input = graph.placeholder(shape: [m as NSNumber, k as NSNumber],
                                      dataType: .bFloat16, name: nil)
            let constant = graph.constant(bytes, shape: [k as NSNumber, n as NSNumber],
                                          dataType: .bFloat16)
            output = graph.matrixMultiplication(primary: input, secondary: constant,
                                                name: nil)
            guard let buffer = device.makeBuffer(length: m * n * MemoryLayout<UInt16>.size,
                                                 options: .storageModeShared)
            else { throw POSIXError(.ENOMEM) }
            result = buffer
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            run(x)
            let words = result.contents().bindMemory(to: UInt16.self, capacity: m * n)
            return MLXArray(UnsafeBufferPointer(start: words, count: m * n))
                .view(dtype: .bfloat16)
                .reshaped([m, n])
        }

        /// Everything except adopting the result back into MLX.
        ///
        /// The difference between this and `callAsFunction` is the one part of
        /// the crossing a future MLX API could plausibly remove — there is no
        /// public way to hand MLX an existing `MTLBuffer`. Measuring it
        /// separately is what distinguishes "dead" from "dead until MLX changes".
        @discardableResult
        func run(_ x: MLXArray) -> Int {
            // Drains MLX. This is the crossing, and it is the cost under test.
            guard let xb = x.asMTLBuffer(device: device, noCopy: true),
                  let raw = queue.makeCommandBuffer()
            else { preconditionFailure("could not reach Metal") }

            let cb = MPSCommandBuffer(commandBuffer: raw)
            graph.encode(
                to: cb,
                feeds: [input: MPSGraphTensorData(xb, shape: [m as NSNumber, k as NSNumber],
                                                  dataType: .bFloat16)],
                targetOperations: nil,
                resultsDictionary: [output: MPSGraphTensorData(
                    result, shape: [m as NSNumber, n as NSNumber], dataType: .bFloat16)],
                executionDescriptor: nil)
            cb.commit()
            cb.waitUntilCompleted()
            return m * n
        }
    }

    // MARK: - The block, with the multiplier swapped out

    struct Parts {
        let norm1: H3RMSNorm, norm2: H3RMSNorm
        let qNorm: H3RMSNorm, kNorm: H3RMSNorm
        let adaln: AdalnProj
        let heads: Int, headDim: Int
    }

    /// The block's forward, in the block's order, with the four GEMMs injected.
    ///
    /// Mirrors `AttentionLayer` and `H3MLP` exactly: qkv split three ways, per
    /// head q/k RMSNorm before RoPE, SDPA, out projection; then fc1 emitting
    /// `2 * ffn` with **gate as the first half**, silu-gated, fc2.
    static func forward(_ x: MLXArray, tEmb: MLXArray, index: ModulationIndex,
                        rope: MLXArray, parts: Parts,
                        qkv: any Projection, out: any Projection,
                        fc1: any Projection, fc2: any Projection) -> MLXArray {
        let m = parts.adaln(tEmb)

        func attention(_ h: MLXArray) -> MLXArray {
            let projected = qkv(h)
            let split = projected.split(parts: 3, axis: -1)
            let shape = split[0].shape.dropLast() + [parts.heads, parts.headDim]
            var q = split[0].reshaped(shape)
            var k = split[1].reshaped(shape)
            let v = split[2].reshaped(shape)
            q = parts.qNorm(q)
            k = parts.kNorm(k)
            q = SplitHalfRoPE.apply(q, table: rope)
            k = SplitHalfRoPE.apply(k, table: rope)
            let merged = AttentionLayer.sdpa(q: q, k: k, v: v, headDim: parts.headDim,
                                             fp32: false, backend: SDPABackend(),
                                             context: nil)
            return out(merged)
        }
        func mlp(_ h: MLXArray) -> MLXArray {
            let inner = fc1(h).split(parts: 2, axis: -1)
            return fc2(silu(inner[0]) * inner[1])
        }

        let h1 = modScaleShift(parts.norm1(x), shift: m[0], scale: m[1], index: index)
        let x1 = modGate(x, gate: m[2], other: attention(h1), index: index)
        let h2 = modScaleShift(parts.norm2(x1), shift: m[3], scale: m[4], index: index)
        return modGate(x1, gate: m[5], other: mlp(h2), index: index)
    }

    // MARK: - Fixture

    struct Fixture {
        let x: MLXArray, tEmb: MLXArray, rope: MLXArray
        let index: ModulationIndex
        let parts: Parts
        let qkvWeight: MLXArray, outWeight: MLXArray
        let fc1Weight: MLXArray, fc2Weight: MLXArray
        let block: DiTBlock
    }

    static func fixture(seed: UInt64 = 11) -> Fixture {
        let cfg = H3Config()
        let h = cfg.hiddenSize, inner = cfg.innerDim, ffn = cfg.ffnHidden
        MLXRandom.seed(seed)
        func w(_ shape: [Int]) -> MLXArray { (MLXRandom.normal(shape) * 0.02).asType(.bfloat16) }

        let qkvWeight = w([3 * inner, h]), outWeight = w([h, inner])
        let fc1Weight = w([2 * ffn, h]), fc2Weight = w([h, ffn])
        let norm1 = H3RMSNorm(weight: w([h]), eps: cfg.normEps)
        let norm2 = H3RMSNorm(weight: w([h]), eps: cfg.normEps)
        let qNorm = H3RMSNorm(weight: w([cfg.headDim]), eps: cfg.qkNormEps)
        let kNorm = H3RMSNorm(weight: w([cfg.headDim]), eps: cfg.qkNormEps)
        let adaln = AdalnProj(weight: w([cfg.adalnOutFeatures, cfg.timeEmbedDim]),
                              bias: nil, expand: 6, modalities: 3, hidden: h,
                              computeFP32: true)
        let segs = [ModSegment(start: 0, stop: 746, row: 0),
                    ModSegment(start: 746, stop: s, row: 6)]
        let rope = H3RoPE.rotationTable(
            angles: MLXRandom.normal([s, 96]).asType(.float32)).asType(.bfloat16)

        return Fixture(
            x: w([s, h]), tEmb: w([3, cfg.timeEmbedDim]), rope: rope,
            index: ModulationIndex(segments: segs, tokenCount: s),
            parts: Parts(norm1: norm1, norm2: norm2, qNorm: qNorm, kNorm: kNorm,
                         adaln: adaln, heads: cfg.numHeads, headDim: cfg.headDim),
            qkvWeight: qkvWeight, outWeight: outWeight,
            fc1Weight: fc1Weight, fc2Weight: fc2Weight,
            block: DiTBlock(norm1: norm1, norm2: norm2,
                            attn: AttentionLayer(qkvWeight: qkvWeight, outWeight: outWeight,
                                                 qNormWeight: qNorm.weight,
                                                 kNormWeight: kNorm.weight,
                                                 heads: cfg.numHeads, headDim: cfg.headDim,
                                                 eps: cfg.qkNormEps),
                            mlp: H3MLP(fc1: fc1Weight, fc2: fc2Weight), adaln: adaln))
    }

    static func time(_ n: Int, _ body: () -> MLXArray) -> Double {
        MLX.eval(body())
        let t0 = Date()
        for _ in 0 ..< n { MLX.eval(body()) }
        return Date().timeIntervalSince(t0) / Double(n)
    }

    // MARK: - 1. Is the model the block?

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func blockModelIsFaithful() {
        let f = Self.fixture()
        let modelled = {
            Self.forward(f.x, tEmb: f.tEmb, index: f.index, rope: f.rope, parts: f.parts,
                         qkv: MLXProjection(weight: f.qkvWeight),
                         out: MLXProjection(weight: f.outWeight),
                         fc1: MLXProjection(weight: f.fc1Weight),
                         fc2: MLXProjection(weight: f.fc2Weight))
        }
        let real = { f.block(f.x, tEmb: f.tEmb, index: f.index, ropeTable: f.rope) }

        let a = modelled(), b = real()
        MLX.eval(a, b)
        let identical = MLX.all(a .== b).item(Bool.self)
        let measured = BenchmarkSupport.interleaved(first: real, second: modelled)
        print(String(format: """

            real DiTBlock  %.0f ms
            modelled       %.0f ms   (%.1f%% of the real block)
            bit-identical  %@
          """, measured.first * 1000, measured.second * 1000,
             100 * measured.second / measured.first, identical ? "yes" : "NO"))

        #expect(identical, "the model is not the block; nothing measured with it counts")
    }

    // MARK: - 2. Does the win survive the crossing?

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func mpsIntegration() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw POSIXError(.ENODEV) }
        let f = Self.fixture()

        let mlxArm = {
            Self.forward(f.x, tEmb: f.tEmb, index: f.index, rope: f.rope, parts: f.parts,
                         qkv: MLXProjection(weight: f.qkvWeight),
                         out: MLXProjection(weight: f.outWeight),
                         fc1: MLXProjection(weight: f.fc1Weight),
                         fc2: MLXProjection(weight: f.fc2Weight))
        }

        let qkv = try MPSProjection(weight: f.qkvWeight, rows: Self.s,
                                    device: device, queue: queue)
        let out = try MPSProjection(weight: f.outWeight, rows: Self.s,
                                    device: device, queue: queue)
        let fc1 = try MPSProjection(weight: f.fc1Weight, rows: Self.s,
                                    device: device, queue: queue)
        let fc2 = try MPSProjection(weight: f.fc2Weight, rows: Self.s,
                                    device: device, queue: queue)
        let mpsArm = {
            Self.forward(f.x, tEmb: f.tEmb, index: f.index, rope: f.rope, parts: f.parts,
                         qkv: qkv, out: out, fc1: fc1, fc2: fc2)
        }

        // Math first, as always: isolated bit-identity does not survive
        // automatically once the results feed elementwise work.
        let a = mlxArm(), b = mpsArm()
        MLX.eval(a, b)
        let identical = MLX.all(a .== b).item(Bool.self)
        let d = (a.asType(.float32) - b.asType(.float32))
        let rel = MLX.sqrt(MLX.mean(d * d)).item(Float.self)
            / MLX.sqrt(MLX.mean(a.asType(.float32) * a.asType(.float32))).item(Float.self)

        let measured = BenchmarkSupport.interleaved(first: mlxArm, second: mpsArm)
        let mlxMs = measured.first * 1000, mpsMs = measured.second * 1000
        print(String(format: """

            block, all MLX          %.0f ms
            block, MPSGraph GEMMs   %.0f ms
            change                  %+.1f%%
            bit-identical           %@   (relative RMS %.1e)

            isolated kernels predicted -98 ms (-8.2%%); the crossing costs the rest
          """, mlxMs, mpsMs, 100 * (mpsMs - mlxMs) / mlxMs,
             identical ? "yes" : "no", rel))
        print(String(format: "  extrapolated to a 50-block step: %.1f s -> %.1f s",
                     mlxMs * 50 / 1000, mpsMs * 50 / 1000))
    }

    // MARK: - 4. What does a crossing cost, and what would fusing buy?

    /// Prices the barrier per crossing, so a fused design can be costed before
    /// it is written.
    ///
    /// `mpsIntegration` routed all four GEMMs and lost. That is the worst
    /// possible arrangement — four barriers a block, 200 a step, each paying a
    /// full drain to save 20-odd milliseconds. `StoryForge`'s `MPSGraphDenoiseStep`
    /// makes the opposite trade: one MPSGraph dispatch per *step*, fusing
    /// guidance, rescale and the Euler update, writing straight into the
    /// caller's `MTLBuffer`. Its own bottleneck report puts 66.9% of denoise
    /// time in the MLX materialisation bridge, which is the same barrier this
    /// tree just measured.
    ///
    /// So the question is not "MPS or MLX" but "how much work per crossing".
    /// Routing 1, 2 and 4 projections gives the slope, and the slope prices
    /// every fusion granularity — including ones nobody has built.
    ///
    ///     H3_BIG=1 swift test --filter crossingCost
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func crossingCost() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw POSIXError(.ENODEV) }
        let f = Self.fixture()

        let mpsQkv = try MPSProjection(weight: f.qkvWeight, rows: Self.s,
                                       device: device, queue: queue)
        let mpsOut = try MPSProjection(weight: f.outWeight, rows: Self.s,
                                       device: device, queue: queue)
        let mpsFc1 = try MPSProjection(weight: f.fc1Weight, rows: Self.s,
                                       device: device, queue: queue)
        let mpsFc2 = try MPSProjection(weight: f.fc2Weight, rows: Self.s,
                                       device: device, queue: queue)

        // Standalone GEMM advantages from `crossingBreakdown`, used only to
        // report what each arm was *trying* to win.
        let advantage = ["qkv": 14.9, "out": 6.8, "fc1": 23.9, "fc2": 37.6]

        func arm(_ names: Set<String>) -> () -> MLXArray {
            {
                Self.forward(
                    f.x, tEmb: f.tEmb, index: f.index, rope: f.rope, parts: f.parts,
                    qkv: names.contains("qkv") ? mpsQkv : MLXProjection(weight: f.qkvWeight),
                    out: names.contains("out") ? mpsOut : MLXProjection(weight: f.outWeight),
                    fc1: names.contains("fc1") ? mpsFc1 : MLXProjection(weight: f.fc1Weight),
                    fc2: names.contains("fc2") ? mpsFc2 : MLXProjection(weight: f.fc2Weight))
            }
        }

        let baseline = arm([])
        let arms: [(String, Set<String>)] = [
            ("1 crossing  (fc2)     ", ["fc2"]),
            ("2 crossings (fc2+fc1) ", ["fc2", "fc1"]),
            ("4 crossings (all)     ", ["fc2", "fc1", "qkv", "out"]),
        ]

        print("\n  crossings   block ms   vs MLX    GEMM win   implied barrier")
        var points: [(Int, Double)] = []
        var baseMs = 0.0
        for (label, names) in arms {
            let measured = BenchmarkSupport.interleaved(first: baseline, second: arm(names))
            let mlxMs = measured.first * 1000, armMs = measured.second * 1000
            baseMs = mlxMs
            let win = names.reduce(0.0) { $0 + (advantage[$1] ?? 0) }
            let delta = armMs - mlxMs
            points.append((names.count, delta + win))
            print(String(format: "  %@  %7.1f   %+6.1f   %+7.1f     %6.1f/crossing",
                         label, armMs, delta, -win, (delta + win) / Double(names.count)))
        }

        let perCrossing = points.reduce(0.0) { $0 + $1.1 / Double($1.0) } / Double(points.count)
        let totalWin = advantage.values.reduce(0, +)
        print(String(format: """

            barrier costs about %.0f ms per crossing
            all four GEMMs are worth %.0f ms if they could be won at once

          what that prices, on a %.0f ms block:
            4 crossings (today)          %+.0f ms   %+.1f%%
            1 crossing  per block        %+.0f ms   %+.1f%%
            1 crossing  per step         %+.0f ms   %+.1f%%   (barrier amortised over 50)
          """, perCrossing, totalWin, baseMs,
             4 * perCrossing - totalWin, 100 * (4 * perCrossing - totalWin) / baseMs,
             perCrossing - totalWin, 100 * (perCrossing - totalWin) / baseMs,
             perCrossing / 50 - totalWin, 100 * (perCrossing / 50 - totalWin) / baseMs))
    }

    // MARK: - 3. Where does the crossing go?

    /// A regression is only actionable once it is attributed.
    ///
    /// Three numbers per projection: MLX's matmul, the full MPS crossing, and
    /// the crossing without adopting the result back into MLX. The third is the
    /// hypothetical where MLX gains an API to accept an existing `MTLBuffer` —
    /// it has none today. If the crossing still loses without the copy, no MLX
    /// change rescues this and the idea is closed rather than parked.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func crossingBreakdown() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw POSIXError(.ENODEV) }
        let f = Self.fixture()
        let cfg = H3Config()

        // Each projection with an input of its real shape, taken from the block.
        let cases: [(String, MLXArray, MLXArray)] = [
            ("qkv     ", f.x, f.qkvWeight),
            ("attn out", MLXRandom.normal([Self.s, cfg.innerDim]).asType(.bfloat16),
             f.outWeight),
            ("mlp fc1 ", f.x, f.fc1Weight),
            ("mlp fc2 ", MLXRandom.normal([Self.s, cfg.ffnHidden]).asType(.bfloat16),
             f.fc2Weight),
        ]

        print("\n  projection    MLX      MPS+copy   MPS only   copy    out MB")
        var totalMLX = 0.0, totalFull = 0.0, totalBare = 0.0
        for (label, x, weight) in cases {
            let projection = try MPSProjection(weight: weight, rows: Self.s,
                                               device: device, queue: queue)
            let mlxMs = Self.time(5) { matmul(x, weight.T) } * 1000
            let fullMs = Self.time(5) { projection(x) } * 1000
            // `run` returns an Int, so it is timed directly rather than through
            // the MLXArray-returning helper.
            projection.run(x)
            let t0 = Date()
            for _ in 0 ..< 5 { projection.run(x) }
            let bareMs = Date().timeIntervalSince(t0) / 5 * 1000

            let outMB = Double(Self.s * weight.dim(0) * 2) / 1e6
            totalMLX += mlxMs; totalFull += fullMs; totalBare += bareMs
            print(String(format: "  %@   %6.1f    %6.1f     %6.1f   %6.1f   %6.0f",
                         label, mlxMs, fullMs, bareMs, fullMs - bareMs, outMB))
        }
        print(String(format: """

            four GEMMs   MLX %.0f ms   MPS+copy %.0f ms   MPS only %.0f ms
            the copy back is %.0f ms; the barrier and dispatch are %.0f ms
          """, totalMLX, totalFull, totalBare,
             totalFull - totalBare, totalBare - totalMLX))
        if totalBare >= totalMLX {
            print("  => even with a free result adoption, the crossing loses. Closed.")
        } else {
            print(String(format: "  => a zero-copy result adoption would save %.0f ms "
                         + "and the idea survives an MLX change.", totalFull - totalBare))
        }
    }
}
