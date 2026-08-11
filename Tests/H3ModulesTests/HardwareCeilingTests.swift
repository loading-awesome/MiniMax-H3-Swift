// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MLX
import H3Foundation
@testable import H3Modules

/// Is 17 TFLOP/s this machine, or is it MLX?
///
/// `gemmCeiling` established that the model runs at MLX's isolated rate — 16.0
/// against 17.3 for a perfectly shaped 8192³ square, so 92%, and no layout or
/// shape work can reclaim anything. That left exactly one speed project, and
/// `docs/PERF_ROADMAP.md` gates it on this measurement:
///
/// > A hand-written Metal GEMM that beats MLX's ... should not be started
/// > without first establishing what the hardware can actually do.
///
/// This is that measurement, and it is deliberately not a kernel. Comparing
/// MLX against a second *vendor* implementation of the same arithmetic answers
/// the question without writing anything: if Apple's own MPS reaches materially
/// more than MLX on identical shapes, the headroom is real, already reachable,
/// and possibly reachable by routing rather than by writing a kernel. If MPS
/// lands on the same number, two independent implementations agree and the
/// hand-written GEMM is dead before it costs anyone a week.
///
///     H3_BIG=1 swift test --filter hardwareCeiling
///
/// Two tests, and the second is the one that matters. `hardwareCeiling` asks
/// the question through `MPSMatrixMultiplication`, which exposes **no bf16** —
/// so it measures fp16, on the argument that both run on the same units and
/// MLX's own fp16 and bf16 rates are identical. `mpsGraphBF16` then removes the
/// argument by measuring the model's actual precision through `MPSGraph`.
@Suite("hardware ceiling", .serialized)
struct HardwareCeilingTests {

    static func tflops(_ m: Int, _ k: Int, _ n: Int, seconds: Double) -> Double {
        2.0 * Double(m) * Double(k) * Double(n) / seconds / 1e12
    }

    // MARK: - MLX

    static func mlxRate(_ m: Int, _ k: Int, _ n: Int, dtype: DType, rounds: Int = 20)
        -> Double {
        let a = MLXRandom.normal([m, k]).asType(dtype)
        let b = MLXRandom.normal([k, n]).asType(dtype)
        MLX.eval(MLX.matmul(a, b))
        let t0 = Date()
        for _ in 0 ..< rounds { MLX.eval(MLX.matmul(a, b)) }
        return tflops(m, k, n, seconds: Date().timeIntervalSince(t0) / Double(rounds))
    }

    // MARK: - MPS

    /// Apple's own GEMM on the same shape, through `MPSMatrixMultiplication`.
    ///
    /// Buffers are `.storageModePrivate`-equivalent shared allocations left
    /// uninitialised: this measures the multiply, and denormals would be the
    /// only content-dependent risk on a GPU that flushes them.
    static func mpsRate(_ m: Int, _ k: Int, _ n: Int, rounds: Int = 20) throws -> Double {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw POSIXError(.ENODEV) }

        let width = MemoryLayout<UInt16>.size          // fp16
        func buffer(_ rows: Int, _ columns: Int) throws -> (MTLBuffer, MPSMatrix) {
            let rowBytes = columns * width
            guard let raw = device.makeBuffer(length: rows * rowBytes,
                                              options: .storageModeShared)
            else { throw POSIXError(.ENOMEM) }
            let descriptor = MPSMatrixDescriptor(rows: rows, columns: columns,
                                                 rowBytes: rowBytes,
                                                 dataType: .float16)
            return (raw, MPSMatrix(buffer: raw, descriptor: descriptor))
        }
        let (_, a) = try buffer(m, k)
        let (_, b) = try buffer(k, n)
        let (_, c) = try buffer(m, n)

        let gemm = MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: false,
            resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0)

        func once() {
            guard let cb = queue.makeCommandBuffer() else { return }
            gemm.encode(commandBuffer: cb, leftMatrix: a, rightMatrix: b, resultMatrix: c)
            cb.commit()
            cb.waitUntilCompleted()
        }
        once()                                          // warm: compile, allocate
        let t0 = Date()
        for _ in 0 ..< rounds { once() }
        return tflops(m, k, n, seconds: Date().timeIntervalSince(t0) / Double(rounds))
    }

    // MARK: -

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func hardwareCeiling() throws {
        let cfg = H3Config()
        let s = 15_731                                  // the control shape
        let h = cfg.hiddenSize, inner = cfg.innerDim, ffn = cfg.ffnHidden

        // The four GEMMs a block runs, plus the square as a clean upper bound.
        let shapes: [(String, Int, Int, Int)] = [
            ("qkv      [S,H]x[H,3I]", s, h, 3 * inner),
            ("attn out [S,I]x[I,H]", s, inner, h),
            ("mlp fc1  [S,H]x[H,2F]", s, h, 2 * ffn),
            ("mlp fc2  [S,F]x[F,H]", s, ffn, h),
            ("square   8192^3", 8192, 8192, 8192),
        ]

        print("\n  shape                     MLX bf16   MLX fp16   MPS fp16    MPS/MLX")
        var ratios: [Double] = []
        for (label, m, k, n) in shapes {
            let bf16 = Self.mlxRate(m, k, n, dtype: .bfloat16)
            let fp16 = Self.mlxRate(m, k, n, dtype: .float16)
            let mps = try Self.mpsRate(m, k, n)
            ratios.append(mps / fp16)
            print(String(format: "  %@   %7.1f    %7.1f    %7.1f     %5.2fx",
                         label, bf16, fp16, mps, mps / fp16))
        }

        let best = ratios.max() ?? 1
        print(String(format: """

            best MPS advantage on any production shape: %.2fx
            the model runs at 16.0 TFLOP/s; MLX's own square ceiling is 17.3
          """, best))
        if best < 1.10 {
            print("  => two independent vendor GEMMs agree. The hand-written kernel is dead.")
        } else {
            print("  => MPS reaches materially more. The headroom is real and already "
                  + "reachable without writing a kernel.")
        }
    }

    // MARK: - Large-K behaviour on non-NAX hardware

    /// `MPSGraph` bf16 matmul rate at an arbitrary shape.
    ///
    /// Buffers are filled with bf16 `1.0` rather than left uninitialised: GEMM
    /// timing does not depend on operand values on this hardware, but a buffer
    /// full of arbitrary bit patterns can hold NaNs, and "probably harmless" is
    /// not worth defending in a benchmark.
    @available(macOS 15.0, *)
    static func mpsGraphRate(_ m: Int, _ k: Int, _ n: Int, rounds: Int = 10) throws
        -> Double {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw POSIXError(.ENODEV) }

        func buffer(_ count: Int) throws -> MTLBuffer {
            let bytes = count * MemoryLayout<UInt16>.size
            guard let raw = device.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw POSIXError(.ENOMEM) }
            var pattern: UInt32 = 0x3F80_3F80              // two bf16 1.0 values
            memset_pattern4(raw.contents(), &pattern, bytes)
            return raw
        }
        let ba = try buffer(m * k), bb = try buffer(k * n), bc = try buffer(m * n)

        let graph = MPSGraph()
        let sa = [m as NSNumber, k as NSNumber]
        let sb = [k as NSNumber, n as NSNumber]
        let sc = [m as NSNumber, n as NSNumber]
        let pa = graph.placeholder(shape: sa, dataType: .bFloat16, name: nil)
        let pb = graph.placeholder(shape: sb, dataType: .bFloat16, name: nil)
        let pc = graph.matrixMultiplication(primary: pa, secondary: pb, name: nil)

        let da = MPSGraphTensorData(ba, shape: sa, dataType: .bFloat16)
        let db = MPSGraphTensorData(bb, shape: sb, dataType: .bFloat16)
        let dc = MPSGraphTensorData(bc, shape: sc, dataType: .bFloat16)

        func once() {
            guard let raw = queue.makeCommandBuffer() else { return }
            let cb = MPSCommandBuffer(commandBuffer: raw)
            graph.encode(to: cb, feeds: [pa: da, pb: db], targetOperations: nil,
                         resultsDictionary: [pc: dc], executionDescriptor: nil)
            cb.commit()
            cb.waitUntilCompleted()
        }
        once()
        let t0 = Date()
        for _ in 0 ..< rounds { once() }
        return tflops(m, k, n, seconds: Date().timeIntervalSince(t0) / Double(rounds))
    }

    /// Per-iteration timings, because the mean was the wrong instrument.
    struct Timing {
        let samples: [Double]
        var p50: Double { percentile(0.50) }
        var p95: Double { percentile(0.95) }
        var worst: Double { samples.max() ?? 0 }
        /// How far the slow tail runs past the median. `mlx#3017` is a report
        /// about *this number*, not about the average.
        var tail: Double { p95 / p50 }
        func percentile(_ q: Double) -> Double {
            let sorted = samples.sorted()
            let i = min(sorted.count - 1, max(0, Int((Double(sorted.count) * q).rounded(.down))))
            return sorted[i]
        }
    }

    /// Does MLX's GEMM lose rate as K grows, on hardware with no NAX?
    ///
    /// [`mlx#3017`](https://github.com/ml-explore/mlx/issues/3017) reports large-K
    /// GEMMs on M5 showing **high run-to-run variance and slow tail iterations**,
    /// measured as per-iteration p50/p95/max, and recovers up to 1.62x by
    /// partitioning K. This machine is an M3 Ultra and has no NAX, so whether the
    /// effect is a NAX-path property or general has no public data either way.
    ///
    ///     H3_BIG=1 swift test --filter gemmKSweep
    ///
    /// Three things this harness has to get right, each of which an earlier
    /// version of it got wrong:
    ///
    ///  * **Report the tail, not the mean.** Averaging iterations hides exactly
    ///    the slow-tail behaviour #3017 is about, so a mean-based "does not
    ///    reproduce" would have been unfounded. Every iteration is timed.
    ///  * **Vary one thing.** Comparing `M=4096,N=4096` against
    ///    `M=15731,N=5376` moves M and N together and cannot attribute anything
    ///    to either. The grid below crosses both independently.
    ///  * **Interleave the arms**, so thermal drift and ordering cannot land on
    ///    one implementation.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func gemmKSweep() throws {
        guard #available(macOS 15.0, *) else { return }

        /// MLX's Case-1 split-K gate, from `mlx/backend/metal/matmul.cpp`.
        /// The 2048 threshold is the one Max and Ultra take.
        func path(_ m: Int, _ k: Int, _ n: Int) -> String {
            let tm = (m + 15) / 16, tn = (n + 15) / 16, tk = k / 16
            return (tm * tn) <= 2048 && tk >= 8 && k >= max(m, n) ? "split-K" : "steel"
        }

        func measure(_ m: Int, _ k: Int, _ n: Int, rounds: Int = 15) throws
            -> (mlx: Timing, mps: Timing) {
            guard let device = MTLCreateSystemDefaultDevice(),
                  let queue = device.makeCommandQueue()
            else { throw POSIXError(.ENODEV) }

            let a = MLXRandom.normal([m, k]).asType(.bfloat16)
            let b = MLXRandom.normal([k, n]).asType(.bfloat16)

            func buffer(_ count: Int) throws -> MTLBuffer {
                let bytes = count * MemoryLayout<UInt16>.size
                guard let raw = device.makeBuffer(length: bytes, options: .storageModeShared)
                else { throw POSIXError(.ENOMEM) }
                var pattern: UInt32 = 0x3F80_3F80          // two bf16 1.0 values
                memset_pattern4(raw.contents(), &pattern, bytes)
                return raw
            }
            let ba = try buffer(m * k), bb = try buffer(k * n), bc = try buffer(m * n)
            let graph = MPSGraph()
            let sa = [m as NSNumber, k as NSNumber]
            let sb = [k as NSNumber, n as NSNumber]
            let sc = [m as NSNumber, n as NSNumber]
            let pa = graph.placeholder(shape: sa, dataType: .bFloat16, name: nil)
            let pb = graph.placeholder(shape: sb, dataType: .bFloat16, name: nil)
            let pc = graph.matrixMultiplication(primary: pa, secondary: pb, name: nil)
            let da = MPSGraphTensorData(ba, shape: sa, dataType: .bFloat16)
            let db = MPSGraphTensorData(bb, shape: sb, dataType: .bFloat16)
            let dc = MPSGraphTensorData(bc, shape: sc, dataType: .bFloat16)

            func mlxOnce() -> Double {
                let t = Date(); MLX.eval(MLX.matmul(a, b))
                return Date().timeIntervalSince(t)
            }
            func mpsOnce() -> Double {
                guard let raw = queue.makeCommandBuffer() else { return 0 }
                let cb = MPSCommandBuffer(commandBuffer: raw)
                let t = Date()
                graph.encode(to: cb, feeds: [pa: da, pb: db], targetOperations: nil,
                             resultsDictionary: [pc: dc], executionDescriptor: nil)
                cb.commit()
                cb.waitUntilCompleted()
                return Date().timeIntervalSince(t)
            }
            _ = mlxOnce(); _ = mpsOnce()                   // warm: compile, allocate

            var mlxSamples: [Double] = [], mpsSamples: [Double] = []
            for i in 0 ..< rounds {
                if i.isMultiple(of: 2) {
                    mlxSamples.append(mlxOnce()); mpsSamples.append(mpsOnce())
                } else {
                    mpsSamples.append(mpsOnce()); mlxSamples.append(mlxOnce())
                }
            }
            return (Timing(samples: mlxSamples), Timing(samples: mpsSamples))
        }

        func report(_ title: String, _ shapes: [(Int, Int, Int)]) throws {
            print("\n  \(title)")
            print("      M      K      N  path       MLX p50   p95/p50  max/p50"
                  + "    MPS p50   p95/p50   MPS/MLX")
            for (m, k, n) in shapes {
                let (mlx, mps) = try measure(m, k, n)
                let mlxRate = Self.tflops(m, k, n, seconds: mlx.p50)
                let mpsRate = Self.tflops(m, k, n, seconds: mps.p50)
                print(String(format:
                    "  %6d %6d %6d  %-8@   %6.1f    %6.2fx  %6.2fx     %6.1f    %6.2fx    %5.2fx",
                    m, k, n, path(m, k, n), mlxRate, mlx.tail, mlx.worst / mlx.p50,
                    mpsRate, mps.tail, mpsRate / mlxRate))
            }
        }

        // M and N crossed independently, so neither can be blamed for the other.
        try report("M and N crossed at K = 2688 / 5376 / 14336", [
            (4096, 2_688, 4096), (4096, 5_376, 4096), (4096, 14_336, 4096),
            (4096, 2_688, 5376), (4096, 5_376, 5376), (4096, 14_336, 5376),
            (15_731, 2_688, 4096), (15_731, 5_376, 4096), (15_731, 14_336, 4096),
            (15_731, 2_688, 5376), (15_731, 5_376, 5376), (15_731, 14_336, 5376),
        ])

        // #3017's own regime, now with the tail statistics it is actually about.
        try report("mlx#3017's shapes, M = N = 4096", [
            (4096, 4096, 4096), (4096, 12_288, 4096), (4096, 24_576, 4096),
        ])

        // Matched pairs straddling the Case-1 gate. N = 1024 gives
        // _tm*_tn = 32*64 = 2048 (inside); N = 1088 gives 32*68 = 2176 (outside).
        // A 6% change in N flips the kernel and nothing else, which is the only
        // way to attribute anything to split-K rather than to skinny shapes.
        try report("matched pairs across MLX's Case-1 split-K boundary", [
            (512, 32_768, 1024), (512, 32_768, 1088),
            (512, 8_192, 1024), (512, 8_192, 1088),
        ])

        print("""

            Rates are TFLOP/s at the median. p95/p50 and max/p50 are the slow-tail
            factors mlx#3017 reports; a value near 1.00 means no tail.
          """)
    }

    // MARK: - Attention

    /// The 37.8% nobody has priced.
    ///
    /// Attention is 446.2 ms of a 1224 ms block and 7.095 TFLOP of a forward's
    /// 961 — a larger prize than all four GEMMs together, and every measurement
    /// in this tree has compared MLX against MLX. §10 closed the custom-GEMM
    /// path; if attention carries a similar gap it is the more valuable target,
    /// and if it does not, §8's 11% was the whole story and the file is closed.
    ///
    ///     H3_BIG=1 swift test --filter mpsAttention
    ///
    /// **Watch for a materialised score matrix.** `[B,H,N,N]` at this shape is
    /// 27.7 GB. MLX's SDPA is a flash-style kernel that never forms it; if
    /// MPSGraph's does, that shows up as an allocation and a time far worse
    /// than MLX rather than as a wrong answer, and the comparison is then
    /// between two different algorithms rather than two implementations.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func mpsAttention() throws {
        guard #available(macOS 15.0, *) else {
            print("\n  MPSGraph SDPA needs macOS 15; skipped.")
            return
        }
        let cfg = H3Config()
        let s = 15_731, h = cfg.numHeads, d = cfg.headDim      // 56 heads, 128 dim
        let scale = 1.0 / Float(d).squareRoot()
        let shape = [1, h, s, d]
        let count = h * s * d

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw POSIXError(.ENODEV) }

        // Same bytes to both, as in `mpsGraphBF16`.
        let qw = Self.bf16Words(count, seed: 0x2545_F491)
        let kw = Self.bf16Words(count, seed: 0x1440_2A29)
        let vw = Self.bf16Words(count, seed: 0x6C07_8965)

        func buffer(_ words: [UInt16]) throws -> MTLBuffer {
            let bytes = words.count * MemoryLayout<UInt16>.size
            guard let raw = device.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw POSIXError(.ENOMEM) }
            words.withUnsafeBytes { _ = memcpy(raw.contents(), $0.baseAddress!, bytes) }
            return raw
        }
        let qb = try buffer(qw), kb = try buffer(kw), vb = try buffer(vw)
        guard let ob = device.makeBuffer(length: count * MemoryLayout<UInt16>.size,
                                         options: .storageModeShared)
        else { throw POSIXError(.ENOMEM) }

        let graph = MPSGraph()
        let ns = shape.map { $0 as NSNumber }
        let pq = graph.placeholder(shape: ns, dataType: .bFloat16, name: nil)
        let pk = graph.placeholder(shape: ns, dataType: .bFloat16, name: nil)
        let pv = graph.placeholder(shape: ns, dataType: .bFloat16, name: nil)
        let po = graph.scaledDotProductAttention(query: pq, key: pk, value: pv,
                                                 scale: scale, name: nil)

        let dq = MPSGraphTensorData(qb, shape: ns, dataType: .bFloat16)
        let dk = MPSGraphTensorData(kb, shape: ns, dataType: .bFloat16)
        let dv = MPSGraphTensorData(vb, shape: ns, dataType: .bFloat16)
        let doo = MPSGraphTensorData(ob, shape: ns, dataType: .bFloat16)

        func once() {
            guard let raw = queue.makeCommandBuffer() else { return }
            let cb = MPSCommandBuffer(commandBuffer: raw)
            graph.encode(to: cb, feeds: [pq: dq, pk: dk, pv: dv], targetOperations: nil,
                         resultsDictionary: [po: doo], executionDescriptor: nil)
            cb.commit()
            cb.waitUntilCompleted()
        }
        once()                                              // warm: compile the graph
        let rounds = 5
        let t0 = Date()
        for _ in 0 ..< rounds { once() }
        let mpsSeconds = Date().timeIntervalSince(t0) / Double(rounds)

        // MLX on identical bytes, through the exact call production makes.
        let q = Self.mlxArray(qw, shape), k = Self.mlxArray(kw, shape)
        let v = Self.mlxArray(vw, shape)
        let mlxCall = {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                              scale: scale, mask: nil)
        }
        MLX.eval(mlxCall())
        let t1 = Date()
        for _ in 0 ..< rounds { MLX.eval(mlxCall()) }
        let mlxSeconds = Date().timeIntervalSince(t1) / Double(rounds)

        // 2 * B*H*N*N*D for QK^T, the same again for (softmax)V.
        let flop = 4.0 * Double(h) * Double(s) * Double(s) * Double(d)
        let mlxRate = flop / mlxSeconds / 1e12
        let mpsRate = flop / mpsSeconds / 1e12

        let mine = mlxCall().asType(.float32)
        let theirs = MLXArray(UnsafeBufferPointer(
            start: ob.contents().bindMemory(to: UInt16.self, capacity: count),
            count: count)).view(dtype: .bfloat16).reshaped(shape).asType(.float32)
        let diff = mine - theirs
        let rel = MLX.sqrt(MLX.mean(diff * diff)).item(Float.self)
            / MLX.sqrt(MLX.mean(mine * mine)).item(Float.self)

        print(String(format: """

            attention  B=1 H=%d N=%d D=%d  bf16   (%.3f TFLOP)

              MLX SDPA        %7.1f ms   %5.1f TFLOP/s
              MPSGraph SDPA   %7.1f ms   %5.1f TFLOP/s   %.2fx

              relative RMS %.2e
              attention is 446.2 ms of a 1224 ms block; the four GEMMs were 1.13-1.22x
            """, h, s, d, flop / 1e12,
             mlxSeconds * 1000, mlxRate, mpsSeconds * 1000, mpsRate,
             mpsRate / mlxRate, rel))

        // Softmax in a different order and possibly a different algorithm, so
        // this is a tolerance. A materialised-score implementation would still
        // agree numerically — the time is what would give it away.
        #expect(rel < 5e-2, "MPSGraph SDPA is not computing the same attention")
    }

    // MARK: - The bf16 gate

    /// bf16 test data as raw bit patterns, so MLX and MPSGraph see **the same
    /// bytes**.
    ///
    /// Generating separately from a seed would leave any disagreement
    /// ambiguous between the inputs and the arithmetic. Truncating fp32 to its
    /// top 16 bits is exactly a bf16 value, and keeps every pattern finite.
    static func bf16Words(_ count: Int, seed: UInt64) -> [UInt16] {
        var state = seed | 1
        return (0 ..< count).map { _ in
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            let unit = Float(state >> 40) / Float(1 << 24) - 0.5     // [-0.5, 0.5)
            return UInt16(truncatingIfNeeded: unit.bitPattern >> 16)
        }
    }

    static func mlxArray(_ words: [UInt16], _ shape: [Int]) -> MLXArray {
        words.withUnsafeBufferPointer { MLXArray($0) }
            .view(dtype: .bfloat16)
            .reshaped(shape)
    }

    /// Does `MPSGraph` do bf16 at production shapes, and does it get the right
    /// answer?
    ///
    /// **This is the gate on section 8.** `MPSMatrixMultiplication` exposes no
    /// bf16, and fp16's exponent range is not a free substitution in a
    /// transformer, so the 8.2% only exists if `MPSGraph` will do the model's
    /// own precision. Speed is measured second and only matters if the numbers
    /// are right: a graph that silently computes in fp32 would look fine on a
    /// tolerance and be a different kernel than the one being priced.
    ///
    ///     H3_BIG=1 swift test --filter mpsGraphBF16
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func mpsGraphBF16() throws {
        let cfg = H3Config()
        let s = 15_731
        let h = cfg.hiddenSize, inner = cfg.innerDim, ffn = cfg.ffnHidden
        let shapes: [(String, Int, Int, Int)] = [
            ("qkv      [S,H]x[H,3I]", s, h, 3 * inner),
            ("attn out [S,I]x[I,H]", s, inner, h),
            ("mlp fc1  [S,H]x[H,2F]", s, h, 2 * ffn),
            ("mlp fc2  [S,F]x[F,H]", s, ffn, h),
            ("square   8192^3", 8192, 8192, 8192),
        ]

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw POSIXError(.ENODEV) }

        print("\n  shape                     MLX bf16   MPSGraph bf16   ratio    rel RMS"
              + "    control   exact")
        var ratios: [Double] = []
        var worst = 0.0
        var allIdentical = true

        for (label, m, k, n) in shapes {
            let aWords = Self.bf16Words(m * k, seed: 0x9E37_79B9)
            let bWords = Self.bf16Words(k * n, seed: 0xBF58_476D)

            // --- MPSGraph
            let graph = MPSGraph()
            let pa = graph.placeholder(shape: [m as NSNumber, k as NSNumber],
                                       dataType: .bFloat16, name: nil)
            let pb = graph.placeholder(shape: [k as NSNumber, n as NSNumber],
                                       dataType: .bFloat16, name: nil)
            let pc = graph.matrixMultiplication(primary: pa, secondary: pb, name: nil)

            func buffer(_ words: [UInt16]) throws -> MTLBuffer {
                let bytes = words.count * MemoryLayout<UInt16>.size
                guard let raw = device.makeBuffer(length: bytes, options: .storageModeShared)
                else { throw POSIXError(.ENOMEM) }
                words.withUnsafeBytes { _ = memcpy(raw.contents(), $0.baseAddress!, bytes) }
                return raw
            }
            let ba = try buffer(aWords), bb = try buffer(bWords)
            // Poisoned with bf16 NaN, not zeroed and not a finite sentinel.
            // A zero buffer is indistinguishable from a correctly computed zero,
            // and a finite sentinel is worse than useless here: bf16 carries 8
            // mantissa bits, so a "distinctive" value like -2.0 is one the
            // product lands on legitimately thousands of times. NaN cannot be
            // produced by this product, so surviving NaN means unwritten.
            guard let bc = device.makeBuffer(length: m * n * MemoryLayout<UInt16>.size,
                                             options: .storageModeShared)
            else { throw POSIXError(.ENOMEM) }
            var poison: UInt32 = 0x7FC0_7FC0
            memset_pattern4(bc.contents(), &poison, m * n * MemoryLayout<UInt16>.size)

            let da = MPSGraphTensorData(ba, shape: [m as NSNumber, k as NSNumber],
                                        dataType: .bFloat16)
            let db = MPSGraphTensorData(bb, shape: [k as NSNumber, n as NSNumber],
                                        dataType: .bFloat16)
            let dc = MPSGraphTensorData(bc, shape: [m as NSNumber, n as NSNumber],
                                        dataType: .bFloat16)

            func once() {
                guard let raw = queue.makeCommandBuffer() else { return }
                let cb = MPSCommandBuffer(commandBuffer: raw)
                graph.encode(to: cb, feeds: [pa: da, pb: db], targetOperations: nil,
                             resultsDictionary: [pc: dc], executionDescriptor: nil)
                cb.commit()
                cb.waitUntilCompleted()
            }
            once()                                       // warm: compile the graph
            let rounds = 20
            let t0 = Date()
            for _ in 0 ..< rounds { once() }
            let mpsRate = Self.tflops(m, k, n,
                                      seconds: Date().timeIntervalSince(t0) / Double(rounds))

            // --- MLX on identical bytes
            let a = Self.mlxArray(aWords, [m, k])
            let b = Self.mlxArray(bWords, [k, n])
            MLX.eval(MLX.matmul(a, b))
            let t1 = Date()
            for _ in 0 ..< rounds { MLX.eval(MLX.matmul(a, b)) }
            let mlxRate = Self.tflops(m, k, n,
                                      seconds: Date().timeIntervalSince(t1) / Double(rounds))

            // --- Do they agree? Accumulation order differs, so this is a
            // tolerance, not an equality. bf16 carries 8 mantissa bits and
            // these reductions are 5,376 to 14,336 long; anything at 1e-2 is
            // the same arithmetic and anything at 1e-1 is not.
            let mine = MLX.matmul(a, b).asType(.float32)
            let theirs = MLXArray(UnsafeBufferPointer(
                start: bc.contents().bindMemory(to: UInt16.self, capacity: m * n),
                count: m * n)).view(dtype: .bfloat16).reshaped([m, n]).asType(.float32)
            func relativeRMS(_ x: MLXArray, _ y: MLXArray) -> Float {
                let d = x - y
                return MLX.sqrt(MLX.mean(d * d)).item(Float.self)
                    / MLX.sqrt(MLX.mean(x * x)).item(Float.self)
            }
            let rel = relativeRMS(mine, theirs)

            // **A zero needs two controls, and the obvious one is not enough.**
            //
            // Perturbing an input by one bf16 ulp shows the *comparison* can
            // detect a change — but it compares MLX against MLX, so it would
            // pass identically if `theirs` had accidentally been bound to
            // `mine` and MPSGraph had never run. That is precisely the bug a
            // reported zero should be suspected of.
            //
            // The second control closes it: the output buffer was poisoned with
            // NaN before the run, which this product cannot produce, so any
            // surviving NaN means the buffer was never written and the
            // agreement is fictional.
            var probed = aWords
            probed[0] = probed[0] ^ 1
            let control = relativeRMS(
                mine, MLX.matmul(Self.mlxArray(probed, [m, k]), b).asType(.float32))
            let poisoned = MLX.any(theirs .!= theirs).item(Bool.self)   // NaN != NaN
            #expect(!poisoned, "MPSGraph did not write \(label); the comparison is vacuous")

            worst = max(worst, Double(rel))
            ratios.append(mpsRate / mlxRate)
            let identical = MLX.all(mine .== theirs).item(Bool.self)
            allIdentical = allIdentical && identical

            print(String(format: "  %@   %7.1f         %7.1f   %5.2fx   %8.2e   %8.2e   %@",
                         label, mlxRate, mpsRate, mpsRate / mlxRate, rel, control,
                         identical ? "yes" : "no"))
            #expect(control > 0, "the comparison cannot detect a one-ulp change; it proves nothing")
        }

        let best = ratios.max() ?? 1
        print(String(format: """

            MPSGraph does bf16 at every production shape.
            best advantage %.2fx, worst relative RMS %.1e
          """, best, worst))

        // **Assert the documented claim, not a weaker one.** §8 reports these
        // as bit-identical, so that is what is checked; a tolerance would let
        // the headline drift away from what the test actually defends. It is a
        // measurement rather than a guarantee — a future MLX or macOS could
        // reorder an accumulation — and a failure here is then the informative
        // event, not a flaky one.
        #expect(worst < 1e-2, "MPSGraph bf16 disagrees with MLX beyond bf16 rounding")
        #expect(allIdentical, "no longer bit-identical; PERF_ROADMAP.md section 8 needs updating")
    }
}
