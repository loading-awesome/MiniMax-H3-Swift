// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXNN
import H3Foundation
import H3ANEBridge
@testable import H3Modules

private final class SendableArray: @unchecked Sendable {
    let value: MLXArray
    init(_ value: MLXArray) { self.value = value }
}

private final class ConcurrentOutputs: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int: MLXArray] = [:]
    func set(_ value: MLXArray, at index: Int) {
        lock.withLock { values[index] = value }
    }
    func get(_ index: Int) -> MLXArray? { lock.withLock { values[index] } }
}

/// Does the Neural Engine path produce the same numbers as the GPU path?
///
/// The version of this file that these tests replace was green while the
/// engine returned noise, for three reasons worth keeping in mind because each
/// is easy to reintroduce:
///
///  1. It never set `H3_ANE`, so it exercised the plain-`matmul` fallback and
///     never reached the engine at all. This suite requires both the production
///     opt-in and its own expensive-test flag, then calls the public entry point.
///  2. It sampled `out[0, 0]` — column zero, which is in the *GPU* shard. The
///     engine's columns start two thirds of the way along. These tests compare
///     each shard's range separately and name it.
///  3. Its inputs were `ones` times a constant, which is invariant under
///     transposing either operand — so the layout bug it was hiding could not
///     have shown up. These tests use asymmetric random data.
/// These ran behind `H3_ANE_TESTS=1` while a single call took 30 seconds. That
/// cost was a defect, not an inherent one — both the activation and the weight
/// upload were handing a transposed view to `asData`, which falls out of its
/// no-copy path into an element-wise gather. With `MLX.contiguous` in front of
/// both, the whole suite runs in under half a second, so it runs by default on
/// any machine the bridge validates and needs no environment variable. A test
/// nobody runs is the failure this file exists to prevent.
///
/// They call `route` rather than the public entry point deliberately: `route`
/// has no `H3_ANE` check, so coverage does not depend on the runner
/// remembering a flag.
@Suite("ANE linear projection", .serialized, .enabled(if: h3_ane_is_available()))
struct ANERealPrimitiveTests {

    /// The engine's own fp16 accuracy, from `docs/ANE_PRECISION_RESULTS.md`:
    /// 7e-5 to 5e-4 per projection on real tensors. Anything at or below this
    /// is the hardware's arithmetic; anything near 1.0 is uncorrelated noise.
    static let tolerance: Float = 2e-3

    static func relRMS(_ reference: MLXArray, _ actual: MLXArray) -> Float {
        let r = reference.asType(.float32), a = actual.asType(.float32)
        let d = r - a
        let num = MLX.sqrt(MLX.sum(d * d)).item(Float.self)
        let den = MLX.sqrt(MLX.sum(r * r)).item(Float.self)
        return num / max(den, 1e-30)
    }

    // MARK: - The layout contract, at the C boundary

    /// Feeds the bridge by hand and checks it against a CPU reference.
    ///
    /// This is the contract the Swift side has to honour: `x` goes in as
    /// `[k, s]` and `w` as `[k, n]` — both with the contraction axis leading,
    /// which is the orientation the engine runs fastest on — and `y` comes back
    /// row-major `[s, n]`. Getting this wrong is invisible, because the byte
    /// counts match either way and the result is merely noise, so it is pinned
    /// here against a CPU reference independently of anything MLX does.
    @Test
    func bridgeMatchesCPUReference() {
        guard h3_ane_is_available() else { return }
        let s = 64, k = 128, n = 32

        guard let program = h3_ane_program_create(Int32(s), Int32(k), Int32(n)),
              let xt = h3_ane_tensor_create(Int32(k), Int32(s)),
              let wt = h3_ane_tensor_create(Int32(k), Int32(n)),
              let yt = h3_ane_tensor_create(Int32(s), Int32(n)) else {
            Issue.record("engine refused a \(s)x\(k)x\(n) matmul")
            return
        }
        defer {
            h3_ane_program_free(program)
            h3_ane_tensor_free(xt); h3_ane_tensor_free(wt); h3_ane_tensor_free(yt)
        }

        var seed: UInt64 = 0x9E3779B97F4A7C15
        func next() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }
        let X = (0 ..< s * k).map { _ in next() }
        let W = (0 ..< n * k).map { _ in next() }

        // Both operands go in with `k` leading, which is the contract in
        // `H3ANEBridge.h`: x is [k, s] and w is [k, n].
        let xT = (0 ..< k).flatMap { c in (0 ..< s).map { Float16(X[$0 * k + c]) } }
        let wT = (0 ..< k).flatMap { c in (0 ..< n).map { Float16(W[$0 * k + c]) } }
        #expect(xT.withUnsafeBytes { h3_ane_tensor_write(xt, $0.baseAddress!, Int32(k), Int32(s)) })
        #expect(wT.withUnsafeBytes { h3_ane_tensor_write(wt, $0.baseAddress!, Int32(k), Int32(n)) })
        #expect(h3_ane_run(program, xt, wt, yt, 1), "evaluate must succeed")

        let row = h3_ane_tensor_row_bytes(yt) / MemoryLayout<Float16>.size
        let y = h3_ane_tensor_ptr(yt)!.assumingMemoryBound(to: Float16.self)
        var num = 0.0, den = 0.0
        for i in 0 ..< s {
            for j in 0 ..< n {
                var acc: Float = 0
                for c in 0 ..< k { acc += X[i * k + c] * W[j * k + c] }
                let got = Float(y[i * row + j])
                num += Double((got - acc) * (got - acc))
                den += Double(acc * acc)
            }
        }
        let rel = Float((num / max(den, 1e-30)).squareRoot())
        #expect(rel < Self.tolerance,
                "engine disagrees with a CPU matmul by relRMS \(rel) — the surface layout does not match what the MIL declares")
    }

    /// A shape that does not match what a tensor was allocated for must be
    /// refused, not truncated.
    ///
    /// The unchecked version of this copy overran a 22 MB surface by 147 MB at
    /// production sequence length and took the process down with SIGBUS.
    @Test
    func tensorWriteRefusesAShapeMismatch() {
        guard h3_ane_is_available() else { return }
        guard let t = h3_ane_tensor_create(64, 128) else {
            Issue.record("tensor create failed"); return
        }
        defer { h3_ane_tensor_free(t) }

        let scratch = [Float16](repeating: 1, count: 64 * 128)
        scratch.withUnsafeBytes { raw in
            #expect(h3_ane_tensor_write(t, raw.baseAddress!, 64, 128), "the declared shape is accepted")
            #expect(!h3_ane_tensor_write(t, raw.baseAddress!, 65, 128), "too many rows is refused")
            #expect(!h3_ane_tensor_write(t, raw.baseAddress!, 64, 129), "too wide is refused")
            #expect(!h3_ane_tensor_write(t, raw.baseAddress!, 4096, 5376), "a production-sized write into a small surface is refused")
        }
    }

    // MARK: - The routed projection

    /// Every shard of the joined output, against a plain MLX matmul.
    ///
    /// The assertion that matters is the one on the engine's own columns. A
    /// transposed operand scores about 1.0 here — the same as comparing two
    /// unrelated tensors — so this cannot pass on a layout error.
    @Test
    func routedProjectionMatchesMLXPerShard() {
        guard h3_ane_is_available() else { return }
        let s = 256, k = 5376, n = 21504

        let x = MLXRandom.normal([s, k]).asType(.bfloat16)
        let w = MLXRandom.normal([n, k]).asType(.bfloat16)
        MLX.eval(x, w)

        guard let plan = ANELinearBackend.plan(n: n) else {
            Issue.record("no shard plan for n=\(n)"); return
        }
        // `route`, not `project`. The public entry point checks `H3_ANE`, and
        // this suite deliberately does not require it — so calling `project`
        // here would take the MLX fallback and compare a matmul against itself.
        guard let actual = ANELinearBackend.route(x: x, qkvWeight: w) else {
            Issue.record("route declined a production-shaped projection"); return
        }
        let reference = MLX.matmul(x, w.transposed())
        MLX.eval(reference, actual)

        #expect(actual.shape == [s, n], "joined output keeps the canonical shape")

        for (name, range) in [("GPU", plan.gpu), ("ANE die 0", plan.ane0), ("ANE die 1", plan.ane1)] {
            let rel = Self.relRMS(reference[0..., range], actual[0..., range])
            #expect(rel < Self.tolerance,
                    "\(name) shard, columns \(range.lowerBound)..<\(range.upperBound): relRMS \(rel)")
        }
    }

    /// The join must preserve channel order, or `split(parts: 3)` downstream
    /// hands q, k and v to each other.
    ///
    /// Distinct per-channel weights make any reordering visible; the previous
    /// test's uniform weights would have passed under an arbitrary permutation.
    @Test
    func joinPreservesChannelOrder() {
        guard h3_ane_is_available() else { return }
        let s = 64, k = 5376, n = 21504

        // Row j selects input channel j % k and scales it by a value unique to
        // j, so output column j is a signature of exactly one weight row.
        let x = MLXRandom.normal([s, k]).asType(.bfloat16)
        var weightRows = [Float](repeating: 0, count: n * k)
        for j in 0 ..< n { weightRows[j * k + (j % k)] = Float(1 + j % 97) * 0.01 }
        let w = MLXArray(weightRows, [n, k]).asType(.bfloat16)
        MLX.eval(x, w)

        guard let actual = ANELinearBackend.route(x: x, qkvWeight: w) else {
            Issue.record("route declined"); return
        }
        let reference = MLX.matmul(x, w.transposed())
        MLX.eval(reference, actual)

        let rel = Self.relRMS(reference, actual)
        #expect(rel < Self.tolerance, "channel order differs between the joined and reference outputs: relRMS \(rel)")
    }

    /// `fc1` is the other routed projection, and it is a different shape:
    /// N=28672 rather than 21504, and its output is split into SwiGLU's gate
    /// and up halves rather than into q, k and v. A join that reordered
    /// channels would hand `silu` the wrong half — which stays plausible in the
    /// output, per `H3MLP`'s own warning — so the boundary is checked here.
    @Test
    func fc1ProjectionMatchesMLXPerShard() {
        let s = 256, k = 5376, n = 28672        // [2 * ffn, hidden]
        let x = MLXRandom.normal([s, k]).asType(.bfloat16)
        let w = MLXRandom.normal([n, k]).asType(.bfloat16)
        MLX.eval(x, w)

        guard let plan = ANELinearBackend.plan(n: n) else {
            Issue.record("no shard plan for fc1, n=\(n)"); return
        }
        guard let actual = ANELinearBackend.route(x: x, qkvWeight: w) else {
            Issue.record("route declined fc1"); return
        }
        let reference = MLX.matmul(x, w.transposed())
        MLX.eval(reference, actual)

        for (name, range) in [("GPU", plan.gpu), ("ANE die 0", plan.ane0), ("ANE die 1", plan.ane1)] {
            let rel = Self.relRMS(reference[0..., range], actual[0..., range])
            #expect(rel < Self.tolerance,
                    "fc1 \(name) shard, columns \(range.lowerBound)..<\(range.upperBound): relRMS \(rel)")
        }

        // The halves SwiGLU will take, compared separately: an off-by-one-shard
        // join shows up as one half being right and the other wrong.
        let gate = Self.relRMS(reference[0..., 0 ..< n / 2], actual[0..., 0 ..< n / 2])
        let up   = Self.relRMS(reference[0..., n / 2 ..< n], actual[0..., n / 2 ..< n])
        #expect(gate < Self.tolerance, "SwiGLU gate half: relRMS \(gate)")
        #expect(up < Self.tolerance, "SwiGLU up half: relRMS \(up)")
    }

    /// The native route does not materialise fc1's complete gate/up tensor: it
    /// merges the three column shards and applies `SiLU(gate) * up` in one Metal
    /// pass. Pin that optimization against the ordinary MLX expression with
    /// different gate and up weights so reversing the halves cannot pass.
    @Test
    func fusedFC1MatchesMLX() {
        let s = 64, hidden = 128, ffn = 512
        let x = MLXRandom.normal([s, hidden]).asType(.bfloat16)
        let gate = (MLXRandom.normal([ffn, hidden]) + 0.75).asType(.bfloat16)
        let up = (MLXRandom.normal([ffn, hidden]) - 0.25).asType(.bfloat16)
        let fc1 = concatenated([gate, up], axis: 0)
        let fc2 = MLXRandom.normal([hidden, ffn]).asType(.bfloat16)
        MLX.eval(x, fc1, fc2)

        let mlp = H3MLP(fc1: fc1, fc2: fc2)
        let fusedActivation = mlp.begin(x).swiGLUValue()
        guard let joined = ANELinearBackend.route(x: x, qkvWeight: fc1) else {
            Issue.record("route declined the fused fc1 oracle"); return
        }
        let joinedHalves = joined.split(parts: 2, axis: -1)
        let joinedActivation = silu(joinedHalves[0]) * joinedHalves[1]
        MLX.eval(fusedActivation, joinedActivation)
        let fusedDiff = MLX.abs(fusedActivation - joinedActivation).max().item(Float.self)
        let fusedRel = Self.relRMS(joinedActivation, fusedActivation)
        let fusedHead = fusedActivation[0, 0 ..< 8].asType(.float32).asArray(Float.self)
        let joinedHead = joinedActivation[0, 0 ..< 8].asType(.float32).asArray(Float.self)
        // These are two independent ANE evaluations. The private reduction can
        // differ by a bf16 ULP on the small fixture, so use the engine's pinned
        // numerical tolerance here; the production tiled-block test separately
        // requires bit identity between schedules at one fixed route.
        #expect(fusedRel < Self.tolerance,
                "native fc1 collection changed SwiGLU; relRMS \(fusedRel), max \(fusedDiff), fused \(fusedHead), joined \(joinedHead)")

        let actual = MLX.matmul(fusedActivation, fc2.transposed())
        let projected = MLX.matmul(x, fc1.transposed())
        let halves = projected.split(parts: 2, axis: -1)
        let reference = MLX.matmul(silu(halves[0]) * halves[1], fc2.transposed())
        MLX.eval(actual, reference)

        let rel = Self.relRMS(reference, actual)
        #expect(rel < 4e-3, "native fused fc1/SwiGLU changed the MLP: relRMS \(rel)")
    }

    @Test
    func persistentMLPIslandMatchesCompleteGPUMLP() {
        let s = 64
        // From the model, never from the backend's own constant. Deriving the
        // fixture from the value under test is what hid a wrong `expectedHidden`
        // through a passing conformance suite.
        let hidden = H3Config().hiddenSize
        let ffn = H3Config().ffnHidden
        MLXRandom.seed(39)
        let x = (MLXRandom.normal([s, hidden]) * 0.5).asType(.bfloat16)
        let fc1 = (MLXRandom.normal([2 * ffn, hidden]) * 0.02).asType(.bfloat16)
        let fc2 = (MLXRandom.normal([hidden, ffn]) * 0.02).asType(.bfloat16)
        MLX.eval(x, fc1, fc2)

        guard let actual = ANEMLPIslandBackend.route(
            x: x, fc1: fc1, fc2: fc2, blockIndex: 0
        ) else {
            Issue.record("persistent MLP island declined its production widths")
            return
        }
        let full = MLX.matmul(x, fc1.transposed())
        let halves = full.split(parts: 2, axis: -1)
        let reference = MLX.matmul(silu(halves[0]) * halves[1], fc2.transposed())
        let full32 = MLX.matmul(x.asType(.float32), fc1.asType(.float32).transposed())
        let halves32 = full32.split(parts: 2, axis: -1)
        let oracle = MLX.matmul(silu(halves32[0]) * halves32[1],
                                fc2.asType(.float32).transposed())
        guard let plan = ANEMLPIslandBackend.partition(ffn: ffn) else {
            Issue.record("production MLP partition was refused")
            return
        }
        func partial(_ range: Range<Int>) -> MLXArray {
            let gate = MLX.matmul(x, fc1[range].transposed())
            let upRange = (ffn + range.lowerBound) ..< (ffn + range.upperBound)
            let up = MLX.matmul(x, fc1[upRange].transposed())
            return MLX.matmul(silu(gate) * up, fc2[0..., range].transposed())
        }
        var gpuOwned = partial(plan.gpu).asType(.float32)
        var aneExpected = MLXArray.zeros([s, hidden], dtype: .float32)
        var aneOracle = MLXArray.zeros([s, hidden], dtype: .float32)
        for die in [plan.ane0, plan.ane1] {
            for piece in 0 ..< ANEMLPIslandBackend.pieces {
                let lower = die.lowerBound + piece * ANEMLPIslandBackend.neuronsPerPiece
                let range = lower ..< (lower + ANEMLPIslandBackend.neuronsPerPiece)
                if piece < ANEMLPIslandBackend.activePieces {
                    aneExpected = aneExpected + partial(range).asType(.float32)
                } else {
                    gpuOwned = gpuOwned + partial(range).asType(.float32)
                }
                let gate32 = MLX.matmul(x.asType(.float32),
                    fc1[range].asType(.float32).transposed())
                let upRange = (ffn + range.lowerBound) ..< (ffn + range.upperBound)
                let up32 = MLX.matmul(x.asType(.float32),
                    fc1[upRange].asType(.float32).transposed())
                if piece < ANEMLPIslandBackend.activePieces {
                    aneOracle = aneOracle + MLX.matmul(
                        silu(gate32) * up32,
                        fc2[0..., range].asType(.float32).transposed())
                }
            }
        }
        let gpuOwnedBF16 = gpuOwned.asType(.bfloat16)
        let partitioned = (gpuOwned + aneExpected).asType(.bfloat16)
        let aneActual = actual.asType(.float32) - gpuOwnedBF16.asType(.float32)
        MLX.eval(actual, reference, oracle, partitioned, aneExpected, aneActual, aneOracle)
        let rel = Self.relRMS(reference, actual)
        let splitVsReference = Self.relRMS(reference, partitioned)
        let islandVsSplit = Self.relRMS(partitioned, actual)
        let gpuVsFP32 = Self.relRMS(oracle, reference.asType(.float32))
        let splitVsFP32 = Self.relRMS(oracle, partitioned.asType(.float32))
        let islandVsFP32 = Self.relRMS(oracle, actual.asType(.float32))
        let aneRel = Self.relRMS(aneExpected, aneActual)
        let aneGPUVsFP32 = Self.relRMS(aneOracle, aneExpected)
        let aneActualVsFP32 = Self.relRMS(aneOracle, aneActual)
        let expectedNorm = MLX.sqrt(MLX.sum(aneExpected * aneExpected))
        let actualNorm = MLX.sqrt(MLX.sum(aneActual * aneActual))
        let normRatio = (actualNorm / expectedNorm).item(Float.self)
        let correlation = (MLX.sum(aneExpected * aneActual) /
                           (expectedNorm * actualNorm)).item(Float.self)
        print("MLP island: vs bf16=\(rel), split-vs-bf16=\(splitVsReference), " +
              "island-vs-split=\(islandVsSplit), GPU-vs-fp32=\(gpuVsFP32), " +
              "split-vs-fp32=\(splitVsFP32), island-vs-fp32=\(islandVsFP32), " +
              "ANE-rel=\(aneRel), ANE-GPU-vs-fp32=\(aneGPUVsFP32), " +
              "ANE-actual-vs-fp32=\(aneActualVsFP32), ANE-norm-ratio=\(normRatio), " +
              "ANE-correlation=\(correlation)")
        #expect(rel < 5e-3, "persistent MLP island changed the complete MLP: relRMS \(rel)")
    }

    @Test
    func persistentMLPPartitionMatchesCalibratedRanges() {
        guard let plan = ANEMLPIslandBackend.partition(ffn: 14_336) else {
            Issue.record("production MLP partition was refused"); return
        }
        #expect(plan.gpu == 0 ..< 10_240)
        #expect(plan.ane0 == 10_240 ..< 12_288)
        #expect(plan.ane1 == 12_288 ..< 14_336)
        #expect(ANEMLPIslandBackend.fc2Scales.count == 50)
        for (a, b) in ANEMLPIslandBackend.fc2Scales {
            for scale in [a, b] {
                let reciprocal = 1 / scale
                #expect(scale > 0 && scale <= 1)
                #expect(reciprocal.rounded() == reciprocal)
                let integer = Int(reciprocal.rounded())
                #expect(integer.nonzeroBitCount == 1, "fc2 scale must be a power of two")
            }
        }
        #expect(ANEMLPIslandBackend.fc2Scales[39].1 == 1.0 / 256.0)
    }

    @Test
    func persistentMLPIslandSinglePieceConformance() {
        let s = 64
        // From the model, never from the backend's own constant. Deriving the
        // fixture from the value under test is what hid a wrong `expectedHidden`
        // through a passing conformance suite.
        let hidden = H3Config().hiddenSize
        let ffn = H3Config().ffnHidden
        let lower = 10_240
        let width = ANEMLPIslandBackend.neuronsPerPiece
        MLXRandom.seed(40)
        let x = (MLXRandom.normal([s, hidden]) * 0.5).asType(.bfloat16)
        let gatePiece = (MLXRandom.normal([width, hidden]) * 0.02).asType(.bfloat16)
        let upPiece = (MLXRandom.normal([width, hidden]) * 0.02).asType(.bfloat16)
        var selector = [Float](repeating: 0, count: hidden * width)
        for i in 0 ..< width { selector[i * width + i] = 1 }
        let downPiece = MLXArray(selector, [hidden, width]).asType(.bfloat16)
        let zeroBefore = MLXArray.zeros([lower, hidden], dtype: .bfloat16)
        let zeroAfter = MLXArray.zeros([ffn - lower - width, hidden], dtype: .bfloat16)
        let gate = concatenated([zeroBefore, gatePiece, zeroAfter], axis: 0)
        let up = concatenated([zeroBefore, upPiece, zeroAfter], axis: 0)
        let fc1 = concatenated([gate, up], axis: 0)
        let fc2 = concatenated([
            MLXArray.zeros([hidden, lower], dtype: .bfloat16), downPiece,
            MLXArray.zeros([hidden, ffn - lower - width], dtype: .bfloat16),
        ], axis: 1)
        MLX.eval(x, fc1, fc2)

        guard let actual = ANEMLPIslandBackend.route(
            x: x, fc1: fc1, fc2: fc2, blockIndex: 0
        ) else {
            Issue.record("single-piece MLP island declined")
            return
        }
        let gateOut = MLX.matmul(x, gatePiece.transposed())
        let upOut = MLX.matmul(x, upPiece.transposed())
        let reference = MLX.matmul(silu(gateOut) * upOut, downPiece.transposed())
        let gate32 = MLX.matmul(x.asType(.float32), gatePiece.asType(.float32).transposed())
        let up32 = MLX.matmul(x.asType(.float32), upPiece.asType(.float32).transposed())
        let oracle = MLX.matmul(silu(gate32) * up32,
                                downPiece.asType(.float32).transposed())
        MLX.eval(actual, reference, oracle)
        let rel = Self.relRMS(reference, actual)
        let gpuVsFP32 = Self.relRMS(oracle, reference.asType(.float32))
        let aneVsFP32 = Self.relRMS(oracle, actual.asType(.float32))
        print("MLP island single piece: vs-bf16=\(rel), GPU-vs-fp32=\(gpuVsFP32), " +
              "ANE-vs-fp32=\(aneVsFP32)")
        #expect(rel < 5e-3, "one persistent ANE chain changed its MLP partial: relRMS \(rel)")
    }

    @Test
    func persistentMLPDownProjectionConformance() {
        let s = 64, k = ANEMLPIslandBackend.neuronsPerPiece
        let n = ANEMLPIslandBackend.expectedHidden
        MLXRandom.seed(41)
        let x = (MLXRandom.normal([s, k]) * 0.5).asType(.bfloat16)
        let w = (MLXRandom.normal([n, k]) * 0.02).asType(.bfloat16)
        let xPacked = MLX.contiguous(x.transposed().asType(.float16))
        let wPacked = MLX.contiguous(w.transposed().asType(.float16))
        MLX.eval(x, w, xPacked, wPacked)
        guard let program = h3_ane_program_create(Int32(s), Int32(k), Int32(n)),
              let xt = h3_ane_tensor_create(Int32(k), Int32(s)),
              let wt = h3_ane_tensor_create(Int32(k), Int32(n)),
              let yt = h3_ane_tensor_create(Int32(s), Int32(n)) else {
            Issue.record("down projection allocation failed")
            return
        }
        defer {
            h3_ane_tensor_free(xt); h3_ane_tensor_free(wt); h3_ane_tensor_free(yt)
            h3_ane_program_free(program)
        }
        let xOK = xPacked.asData(access: .noCopyIfContiguous).data.withUnsafeBytes {
            h3_ane_tensor_write(xt, $0.baseAddress!, Int32(k), Int32(s))
        }
        let wOK = wPacked.asData(access: .noCopyIfContiguous).data.withUnsafeBytes {
            h3_ane_tensor_write(wt, $0.baseAddress!, Int32(k), Int32(n))
        }
        #expect(xOK && wOK)
        #expect(h3_ane_run(program, xt, wt, yt, 1))
        let actual = MLXArray(rawPointer: h3_ane_tensor_ptr(yt)!, [s, n],
                              dtype: .float16) { }.asType(.bfloat16)
        let reference = MLX.matmul(x, w.transposed())
        let oracle = MLX.matmul(x.asType(.float32), w.asType(.float32).transposed())
        MLX.eval(actual, reference, oracle)
        let rel = Self.relRMS(reference, actual)
        let aneVsFP32 = Self.relRMS(oracle, actual.asType(.float32))
        print("MLP island down: vs-bf16=\(rel), ANE-vs-fp32=\(aneVsFP32)")
        #expect(rel < Self.tolerance)
    }

    /// The shard plan keeps heads and 64-byte rows intact, and leaves the
    /// engine roughly its measured share of the work.
    @Test
    func shardPlanIsWellFormed() {
        guard let plan = ANELinearBackend.plan(n: 21504) else {
            Issue.record("no plan for the production shape"); return
        }
        #expect(plan.gpu.lowerBound == 0)
        #expect(plan.gpu.upperBound == plan.ane0.lowerBound)
        #expect(plan.ane0.upperBound == plan.ane1.lowerBound)
        #expect(plan.ane1.upperBound == 21504, "the three shards cover every channel exactly once")

        for range in [plan.gpu, plan.ane0, plan.ane1] {
            #expect(range.count % 128 == 0, "a shard must not split a 128-wide head")
            #expect(range.lowerBound % 32 == 0, "a shard must start on a 64-byte row boundary")
        }

        // **Against the configured share, not against a literal.** These were
        // `3072` and `15360`, which is the plan for a share of 0.286 — the
        // balance point for an engine contracting 5,376 deep in one call. Split,
        // the engine is at parity and the balance moves to 0.45, and three
        // hardcoded plans failed for a tuning change that was the entire point
        // of the work. What the plan owes is that it matches the share it was
        // asked for, to within the head-boundary rounding.
        func expectMatchesShare(_ p: ANELinearBackend.ShardPlan, n: Int,
                                _ what: String, headDim: Int = 128) {
            let asked = ANELinearBackend.share
            let got = Double(2 * p.perDie) / Double(n)
            #expect(abs(got - asked) <= Double(2 * headDim) / Double(n),
                    "\(what) gives the engine \(got) of its columns against a share of \(asked)")
            #expect(p.gpu.count == n - 2 * p.perDie, "\(what) shards must cover n exactly")
        }
        expectMatchesShare(plan, n: 21504, "qkv")

        // Shapes the engine has no useful split for must say so rather than
        // returning a degenerate plan.
        // fc1's width has its own balance point.
        guard let fc1 = ANELinearBackend.plan(n: 28672) else {
            Issue.record("no plan for fc1"); return
        }
        expectMatchesShare(fc1, n: 28672, "fc1")
        #expect(fc1.ane1.upperBound == 28672, "fc1 shards cover every channel")

        // `attn out` is the third routed width, and the one whose contraction
        // axis is larger than its output — 7168 in, 5376 out — which is why it
        // returns the least of the three.
        guard let out = ANELinearBackend.plan(n: 5376) else {
            Issue.record("no plan for attn out"); return
        }
        expectMatchesShare(out, n: 5376, "attn out")

        #expect(ANELinearBackend.plan(n: 128) == nil, "too small to split")
        #expect(ANELinearBackend.plan(n: 21503) == nil, "not a whole number of heads")
    }

    /// Sessions are cached by shape and reuse mutable x/y surfaces. Two callers
    /// of the public entry point must not observe each other's activation or
    /// output when they meet the same cached session concurrently.
    @Test
    func concurrentCallsKeepSessionSurfacesIsolated() {
        let s = 64, k = 128, n = 1024
        let inputs = [
            SendableArray(MLXRandom.normal([s, k]).asType(.bfloat16)),
            SendableArray((MLXRandom.normal([s, k]) + 3).asType(.bfloat16)),
        ]
        let weight = SendableArray(MLXRandom.normal([n, k]).asType(.bfloat16))
        MLX.eval(inputs.map(\.value) + [weight.value])

        let outputs = ConcurrentOutputs()
        DispatchQueue.concurrentPerform(iterations: inputs.count) { index in
            guard let output = ANELinearBackend.route(
                x: inputs[index].value, qkvWeight: weight.value) else { return }
            output.eval()
            outputs.set(output, at: index)
        }

        for index in inputs.indices {
            guard let actual = outputs.get(index) else {
                Issue.record("concurrent projection \(index) produced no output"); continue
            }
            let reference = MLX.matmul(inputs[index].value, weight.value.transposed())
            let rel = Self.relRMS(reference, actual)
            #expect(rel < Self.tolerance,
                    "concurrent projection \(index) crossed session data: relRMS \(rel)")
        }
    }

    /// The receipt records what the engine *did*, not what it was configured to
    /// do, so `project` has to observe rather than declare.
    ///
    /// This matters because `route` can decline — an unsupported shape, a failed
    /// compile — and a decline is deliberately silent to the caller: same
    /// numbers, no speedup. It must not be silent to the receipt, because a
    /// render that fell back is not the render that never offered.
    @Test
    func routingIsObservedNotDeclared() {
        let k = 5376
        let x = MLXRandom.normal([64, k]).asType(.bfloat16)
        let good = MLXRandom.normal([21504, k]).asType(.bfloat16)
        // 640 is a whole number of heads but too narrow to split three ways, so
        // `plan` refuses it and `project` must fall back and say so.
        let tooNarrow = MLXRandom.normal([640, k]).asType(.bfloat16)
        MLX.eval(x, good, tooNarrow)

        _ = ANELinearBackend.project(x: x, weight: good, label: "observed-routed")
        _ = ANELinearBackend.project(x: x, weight: tooNarrow, label: "observed-declined")

        let routed = ANELinearBackend.routedProjections
        let declined = ANELinearBackend.declinedProjections

        if ANELinearBackend.isEnabled {
            #expect(routed.contains("observed-routed"),
                    "a projection the engine computed must appear as routed")
            #expect(declined.contains("observed-declined"),
                    "a projection that fell back must appear as declined")
        } else {
            // Without the opt-in nothing is offered, so nothing is recorded —
            // which is itself the right receipt for a default render.
            #expect(routed.isEmpty && declined.isEmpty)
        }
        #expect(Set(routed).isDisjoint(with: declined),
                "a projection cannot be both routed and declined")
    }

    // MARK: - Does it actually pay?

    /// Measures the complete production-width MLP, including the final join,
    /// with the GPU and both ANE dies doing their assigned neuron ranges at the
    /// same time. Compilation and persistent weight upload are warmed outside
    /// the samples. Opt-in because the fixture holds several GB of surfaces.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func persistentMLPIslandBeatsCompleteGPUMatmul() {
        let s = 15_461
        // From the model, never from the backend's own constant. Deriving the
        // fixture from the value under test is what hid a wrong `expectedHidden`
        // through a passing conformance suite.
        let hidden = H3Config().hiddenSize
        let ffn = H3Config().ffnHidden
        MLXRandom.seed(42)
        let x = (MLXRandom.normal([s, hidden]) * 0.5).asType(.bfloat16)
        let fc1 = (MLXRandom.normal([2 * ffn, hidden]) * 0.02).asType(.bfloat16)
        let fc2 = (MLXRandom.normal([hidden, ffn]) * 0.02).asType(.bfloat16)
        MLX.eval(x, fc1, fc2)

        func plain() -> MLXArray {
            let projected = MLX.matmul(x, fc1.transposed())
            let halves = projected.split(parts: 2, axis: -1)
            return MLX.matmul(silu(halves[0]) * halves[1], fc2.transposed())
        }
        func island() -> MLXArray {
            guard let value = ANEMLPIslandBackend.route(
                x: x, fc1: fc1, fc2: fc2, blockIndex: 0
            ) else { preconditionFailure("production MLP island declined") }
            return value
        }

        MLX.eval(island())
        let measured = BenchmarkSupport.interleaved(rounds: 2, first: plain, second: island)
        let speedup = measured.first / measured.second
        print(String(format: "MLP island S=%d: GPU %.1f ms, hybrid %.1f ms, %.3fx",
                     s, measured.first * 1_000, measured.second * 1_000, speedup))
        #expect(speedup > 1.05,
                "persistent MLP island does not pay: GPU \(measured.first * 1_000) ms, hybrid \(measured.second * 1_000) ms, \(speedup)x")
    }

    /// The whole point, asserted.
    ///
    /// Every other test here checks numbers, and numbers stayed correct through
    /// a 400x slowdown: a strided `asData` turned a 22 MB memcpy into an
    /// element-wise gather and the routed path fell to 0.003x while this suite
    /// stayed green. Correctness assertions cannot see that. This one can.
    @Test
    func routedProjectionBeatsPlainMatmul() {
        let s = 15_731, k = 5_376, n = 21_504      // the production control shape
        let x = MLXRandom.normal([s, k]).asType(.bfloat16)
        let w = MLXRandom.normal([n, k]).asType(.bfloat16)
        MLX.eval(x, w)

        // Compile and upload weights outside the measurement; both are once per
        // render, not once per call.
        guard let warm = ANELinearBackend.route(x: x, qkvWeight: w) else {
            Issue.record("route declined the production shape"); return
        }
        MLX.eval(warm)

        func median(_ rounds: Int, _ body: () -> Void) -> Double {
            body()
            var samples: [Double] = []
            for _ in 0 ..< rounds {
                let t0 = Date(); body(); samples.append(Date().timeIntervalSince(t0))
            }
            return samples.sorted()[rounds / 2] * 1000
        }

        let plain  = median(5) { MLX.eval(MLX.matmul(x, w.transposed())) }
        let routed = median(5) { MLX.eval(ANELinearBackend.route(x: x, qkvWeight: w)!) }
        let speedup = plain / routed

        // The bar is deliberately just above parity rather than at the 1.15x
        // gate: this guards against a regression making the engine a liability,
        // and the gate is a question for the roadmap, not for a unit test on a
        // machine that may be busy.
        #expect(speedup > 1.05,
                "routing qkv to the engine is not paying: plain \(plain) ms, routed \(routed) ms, \(speedup)x")
        print(String(format: "qkv S=%d: plain %.1f ms, routed %.1f ms, %.3fx", s, plain, routed, speedup))
    }
}
