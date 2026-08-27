// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXRandom
import H3Foundation
import H3ANEBridge
@testable import H3Modules

/// Does expressing the projection as a 1x1 convolution beat expressing it as a
/// matmul, and does it compute the same thing?
///
/// The matmul lowering transposes the activation **inside the engine graph** —
/// 169 MB moved at production shard before any arithmetic — because the engine
/// wants the sequence as the minor axis and `matmul` wants it as the major one.
/// A 1x1 convolution over `[1,k,1,s]` wants exactly what the caller already
/// has: channels in, channels out, sequence on the spatial axis.
///
/// The gap this is chasing is not speculative. `ANECeilingTests` measures Core
/// ML's own convolution lowering at 5.46 TFLOP/s a die on the qkv shape while
/// `docs/ANE_REVERSE_ENGINEERING.md` measures this bridge's matmul lowering at
/// 3.87 — same arithmetic, 41% apart, and 41% is the difference between a
/// 1.37x CFG pair and a 1.53x one.
///
///     H3_BIG=1 swift test --filter aneForm
@Suite("ANE lowering form", .serialized, .enabled(if: h3_ane_is_available()))
struct ANEFormTests {

    /// The production shard: one die's share of `qkv` at production length.
    static let s = ANELinearBackend.paddedSequence(15_731)
    static let k = 5376
    static let n = 3072

    /// Deterministic fp16 in the range the saturation bound was measured on.
    static func halves(_ count: Int, seed: UInt64) -> [Float16] {
        var state = seed
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float16(0.02 + 0.06 * (Float((state >> 40) & 0xFFFF) / 65_535.0))
        }
    }

    static func write(_ t: OpaquePointer, _ values: [Float16], rows: Int, width: Int) -> Bool {
        values.withUnsafeBytes { raw in
            h3_ane_tensor_write(t, raw.baseAddress!, Int32(rows), Int32(width))
        }
    }

    static func read(_ t: OpaquePointer, rows: Int, width: Int) -> MLXArray {
        let ptr = h3_ane_tensor_ptr(t)!
        return MLXArray(rawPointer: ptr, [rows, width], dtype: .float16) { }
    }

    private static func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }

    /// Both forms, same numbers, and what each one costs — across the whole
    /// decomposition, not just the one the shard plan happens to use.
    ///
    /// The shard plan splits **output columns** and runs the full sequence:
    /// `n = 3072` a die at `s = 15744`. Core ML's own harness measured 5.46
    /// TFLOP/s on a different decomposition entirely — the full `n = 21504`
    /// over a **sequence tile** of 2048 — and that is a shape this bridge has
    /// never compiled. Which axis the work is cut along is a free choice, so
    /// it should be measured rather than inherited.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func aneForm() throws {
        let k = Self.k
        print("\n  k = \(k) throughout; rate is both dies under h3_ane_run_pair\n")
        print("  " + "s".padding(toLength: 8, withPad: " ", startingAt: 0)
              + "n".padding(toLength: 8, withPad: " ", startingAt: 0)
              + "TFLOP".padding(toLength: 9, withPad: " ", startingAt: 0)
              + "matmul ms   TF/s     conv ms   TF/s")

        // Cut by output column at full length (what the shard plan does today),
        // by sequence tile at full width (what Core ML measured), and the
        // rectangles between them.
        let shapes: [(s: Int, n: Int)] = [
            (15_744, 3072), (15_744, 10_752), (15_744, 21_504),
            (8192, 3072), (8192, 10_752), (8192, 21_504),
            (4096, 21_504), (2048, 21_504), (2048, 3072),
        ]

        var best = (rate: 0.0, s: 0, n: 0, form: "")
        for shape in shapes {
            let s = shape.s, n = shape.n
            let flops = 2.0 * Double(s) * Double(k) * Double(n)
            var line = String(format: "  %-8d%-8d%-9.3f", s, n, flops / 1e12)

            for form in Self.forms(k: k, n: n, s: s) {
                guard let m = Self.measure(form: form, s: s, k: k, n: n) else {
                    line += "     refused        "
                    continue
                }
                let rate = 2 * flops / m / 1e12
                line += String(format: "  %8.1f%7.2f", m * 1000, rate)
                if rate > best.rate { best = (rate, s, n, form.name) }
            }
            print(line)
        }

        print(String(format: "\n  best: %.2f TF/s at s=%d n=%d (%@)",
                     best.rate, best.s, best.n, best.form as NSString))
        // 7.69 TF/s is what the shipping shard plan gets. What the rest of the
        // schedule can be built on is whatever this sweep finds.
        print(String(format: "  against the shipping shard's 7.69 TF/s: %.3fx\n", best.rate / 7.69))
    }

    struct Form {
        let name: String
        let raw: H3ANEForm
        let wRows: Int, wWidth: Int
        let yRows: Int, yWidth: Int
    }

    /// Rate against the **contraction** dimension, which no sweep has touched.
    ///
    /// Every measurement of this engine so far has cut the work by output
    /// column or by sequence tile at the model's own `k`. Tensor parallelism on
    /// NVIDIA splits either axis — columns *or* the contraction — and splitting
    /// `k` is the one this bridge has never tried. It is not free: two half-`k`
    /// products must be summed, which costs a pass over the output and changes
    /// the accumulation. It is only worth paying if the engine is materially
    /// faster at a smaller `k`.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func contractionSweep() {
        let s = Self.s, n = Self.n
        print("\n  s=\(s) n=\(n), rate is both dies under h3_ane_run_pair\n")
        print("  " + "k".padding(toLength: 10, withPad: " ", startingAt: 0)
              + "TFLOP".padding(toLength: 10, withPad: " ", startingAt: 0)
              + "ms".padding(toLength: 10, withPad: " ", startingAt: 0) + "TF/s")
        for k in [1344, 2688, 5376, 7168, 10_752, 14_336] {
            let flops = 2.0 * Double(s) * Double(k) * Double(n)
            let form = Form(name: "matmul", raw: H3ANEFormMatmul,
                            wRows: k, wWidth: n, yRows: s, yWidth: n)
            guard let ms = Self.measure(form: form, s: s, k: k, n: n) else {
                print(String(format: "  %-10d refused", k)); continue
            }
            print(String(format: "  %-10d%-10.3f%-10.1f%.2f",
                         k, flops / 1e12, ms * 1000, 2 * flops / ms / 1e12))
        }
        print("")
    }

    /// Split the contraction, sum the partials, and check both the numbers and
    /// the clock.
    ///
    /// `contractionSweep` says the engine is 2.65x faster doing `k = 5376` as
    /// four chunks of 1344 than as one contraction. That is only a lever if the
    /// chunks actually compute the projection: a compiler that quietly skipped
    /// work would look exactly this fast. So the split is checked against MLX,
    /// against the single full-`k` call, and timed in the same test.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func splitContraction() {
        for k in [5376, 7168, 14_336] { splitAt(k: k) }
    }

    func splitAt(k: Int) {
        let s = Self.s, n = Self.n
        let xv = Self.halves(k * s, seed: 0x2545_F491_4F6C_DD1D)   // [k, s]
        let wv = Self.halves(k * n, seed: 0x9E37_79B9_7F4A_7C15)   // [k, n]

        // The engine's own layout makes this natural: `x` is [k, s] and `w` is
        // [k, n], both row-major, so a slice of the contraction is a slice of
        // whole rows in each — contiguous, no gather.
        func run(chunks: Int) -> (MLXArray, Double)? {
            let kc = k / chunks
            var programs: [OpaquePointer] = [], tensors: [OpaquePointer] = []
            defer {
                for p in programs { h3_ane_program_free(p) }
                for t in tensors { h3_ane_tensor_free(t) }
            }
            var parts: [(OpaquePointer, OpaquePointer, OpaquePointer, OpaquePointer,
                         OpaquePointer, OpaquePointer)] = []
            for c in 0 ..< chunks {
                guard let p0 = h3_ane_program_create(Int32(s), Int32(kc), Int32(n)),
                      let p1 = h3_ane_program_create(Int32(s), Int32(kc), Int32(n)),
                      let x0 = h3_ane_tensor_create(Int32(kc), Int32(s)),
                      let w0 = h3_ane_tensor_create(Int32(kc), Int32(n)),
                      let y0 = h3_ane_tensor_create(Int32(s), Int32(n)),
                      let y1 = h3_ane_tensor_create(Int32(s), Int32(n))
                else { return nil }
                programs += [p0, p1]; tensors += [x0, w0, y0, y1]
                let xs = Array(xv[(c * kc * s) ..< ((c + 1) * kc * s)])
                let ws = Array(wv[(c * kc * n) ..< ((c + 1) * kc * n)])
                guard Self.write(x0, xs, rows: kc, width: s),
                      Self.write(w0, ws, rows: kc, width: n) else { return nil }
                parts.append((p0, p1, x0, w0, y0, y1))
            }
            // Both dies get half the chunk's output columns in the real plan;
            // here each die runs the whole chunk so the rate is comparable with
            // `contractionSweep`, which does the same.
            func once() -> Bool {
                for p in parts where !h3_ane_run_pair(p.0, p.2, p.3, p.4, p.1, p.2, p.3, p.5) {
                    return false
                }
                return true
            }
            guard once() else { return nil }
            var v: [Double] = []
            for _ in 0 ..< 5 { let t = Date(); _ = once(); v.append(Date().timeIntervalSince(t)) }

            var acc = Self.read(parts[0].4, rows: s, width: n).asType(.float32)
            for p in parts.dropFirst() {
                acc = acc + Self.read(p.4, rows: s, width: n).asType(.float32)
            }
            MLX.eval(acc)
            return (acc, Self.median(v) * 1000)
        }

        let reference = MLX.matmul(
            MLXArray(xv.map { Float($0) }, [k, s]).transposed(),
            MLXArray(wv.map { Float($0) }, [k, n]))
        MLX.eval(reference)

        print("\n  k=\(k) n=\(n) s=\(s), split into chunks of k\n")
        print("  chunks   k each        ms     TF/s    rel RMS vs MLX")
        let flops = 2.0 * Double(s) * Double(k) * Double(n)
        for chunks in [1, 2, 4, 8] where k % chunks == 0 && (k / chunks) % 64 == 0 {
            guard let (out, ms) = run(chunks: chunks) else {
                print("  \(chunks): refused"); continue
            }
            let rel = Self.relRMS(reference, out)
            print(String(format: "  %6d   %6d   %7.1f   %6.2f    %.3e",
                         chunks, k / chunks, ms, 2 * flops / (ms / 1000) / 1e12, rel))
            #expect(rel < 5e-3, "splitting the contraction into \(chunks) changed the result")
        }
        print("")
    }

    static func relRMS(_ reference: MLXArray, _ actual: MLXArray) -> Float {
        let r = reference.asType(.float32), a = actual.asType(.float32)
        let d = r - a
        return MLX.sqrt(MLX.mean(d * d)).item(Float.self)
            / MLX.sqrt(MLX.mean(r * r)).item(Float.self)
    }

    /// What does the whole routed projection cost, engine plus join?
    ///
    /// `splitContraction` timed the engine alone. The split also adds work the
    /// engine never sees: `splits` partials per die must be summed in fp32
    /// before the operand scale comes off, which is `splits - 1` extra passes
    /// over a `[s, perDie]` array. If that summation is eating the engine's
    /// gain, the accumulate belongs in one merge kernel rather than a chain of
    /// MLX adds — and this is the measurement that says whether it is worth
    /// writing.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func splitJoinCost() {
        let s = 15_731, k = 5376, n = 21_504
        ANELinearBackend.share = 0.5
        let mag = Float(ProcessInfo.processInfo.environment["H3_ACC_MAG"] ?? "") ?? 1.0
        let x = MLX.contiguous((MLXRandom.normal([s, k]) * mag).asType(.bfloat16))
        let w = MLX.contiguous((MLXRandom.normal([n, k]) * mag).asType(.bfloat16))
        MLX.eval(x, w)

        print("\n  qkv [\(s),\(k)]x[\(k),\(n)] at engine share 0.5\n")
        print("  split   project ms   engine ms   join + overhead")
        for (splits, engineMs) in [(1, 133.8), (4, 50.3)] {
            ANELinearBackend.splitOverride = splits
            defer { ANELinearBackend.splitOverride = nil }
            _ = ANELinearBackend.project(x: x, weight: w, label: "qkv")
            var v: [Double] = []
            for _ in 0 ..< 5 {
                let t = Date()
                MLX.eval(ANELinearBackend.project(x: x, weight: w, label: "qkv"))
                v.append(Date().timeIntervalSince(t))
            }
            let ms = v.sorted()[2] * 1000
            print(String(format: "  %5d   %10.1f   %9.1f   %10.1f",
                         splits, ms, engineMs, ms - engineMs))
        }
        ANELinearBackend.share = 0.286
        print("")
    }

    /// Splitting must not cost accuracy, and the fused reducer must not either.
    ///
    /// The split's justification is that it is faster *and* closer to fp32 —
    /// each piece is a shorter fp16 reduction, and the pieces are summed in
    /// fp32. Fusing that sum into one compiled kernel is a performance change
    /// that touches the arithmetic path, so it is checked rather than assumed.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func splitKeepsAccuracy() {
        let s = 2048, k = 5376, n = 21_504
        ANELinearBackend.share = 0.45
        defer { ANELinearBackend.share = 0.45 }
        let (gpuError, whole, split) = Self.routedError(s: s, k: k, n: n, sigma: 1.0)
        print(String(format: """

          rel RMS against fp32, unit-scale operands
            bf16 on the GPU     %.3e
            routed, whole k     %.3e
            routed, split 4     %.3e

        """, gpuError, whole, split))
        #expect(split <= whole,
                "splitting the contraction must not lose accuracy against the whole one")
        #expect(split < gpuError * 1.05,
                "the routed path must stay within reach of the bf16 path it replaces")
    }

    /// **The operand scale has a floor, and nothing in the code says so.**
    ///
    /// Every routed activation is multiplied by 1/16 before the engine sees it,
    /// to move the partial-sum envelope away from the 2^15 saturation cliff.
    /// That trade has a second edge nobody had measured: fp16's smallest normal
    /// is 6.1e-05, so scaling down far enough pushes the *products* into
    /// denormals and the engine loses them.
    ///
    /// This is a property of the shipping path, not of the contraction split —
    /// whole and split degrade together, and identically. `ANE_PRECISION_RESULTS`
    /// measured 7e-05 to 5e-04 on real captured activations, which sit well
    /// above the cliff; what was missing is any statement of where the cliff is.
    ///
    ///     sigma 1.00   engine sees 0.0625, product 6.3e-02   1.68e-03  fine
    ///     sigma 0.20   engine sees 0.0125, product 2.5e-03   2.13e-03  fine
    ///     sigma 0.05   engine sees 0.0031, product 1.6e-04   3.10e-02  20x worse
    ///
    /// A projection whose activations run near sigma 0.05 would be quietly
    /// wrong by 3%, which no downstream gate in this tree would catch.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func operandScaleHasAnUnderflowFloor() {
        let s = 2048, k = 5376, n = 21_504
        ANELinearBackend.share = 0.45
        let safe = Self.routedError(s: s, k: k, n: n, sigma: 1.0)
        let small = Self.routedError(s: s, k: k, n: n, sigma: 0.05)
        print(String(format: "\n  sigma 1.00 routed %.3e   sigma 0.05 routed %.3e   (%.0fx)\n",
                     safe.whole, small.whole, small.whole / safe.whole))
        #expect(safe.whole < safe.gpu * 1.05,
                "at production magnitudes the engine must match the GPU path")
        // Pinned so the day someone changes `operandScale` this says what it
        // costs at the bottom of the range rather than the top.
        #expect(small.whole > safe.whole * 5,
                "the underflow floor has moved; re-measure where it now sits")
    }

    /// Routed error against an fp32 reference, whole and split, plus the plain
    /// bf16 GPU path for scale.
    static func routedError(s: Int, k: Int, n: Int, sigma: Float)
        -> (gpu: Float, whole: Float, split: Float) {
        let x = MLX.contiguous((MLXRandom.normal([s, k]) * sigma).asType(.bfloat16))
        let w = MLX.contiguous((MLXRandom.normal([n, k]) * sigma).asType(.bfloat16))
        MLX.eval(x, w)
        let reference = MLX.matmul(x.asType(.float32), w.asType(.float32).transposed())
        let gpu = MLX.matmul(x, w.transposed())
        MLX.eval(reference, gpu)

        var routed: [Int: Float] = [:]
        for splits in [1, 4] {
            ANELinearBackend.splitOverride = splits
            defer { ANELinearBackend.splitOverride = nil }
            let out = ANELinearBackend.project(x: x, weight: w, label: "qkv")
            MLX.eval(out)
            routed[splits] = relRMS(reference, out)
        }
        return (relRMS(reference, gpu), routed[1]!, routed[4]!)
    }

    static func forms(k: Int, n: Int, s: Int) -> [Form] {
        [Form(name: "matmul", raw: H3ANEFormMatmul, wRows: k, wWidth: n, yRows: s, yWidth: n),
         Form(name: "conv", raw: H3ANEFormConv, wRows: n, wWidth: k, yRows: n, yWidth: s)]
    }

    /// Median wall time of one `h3_ane_run_pair` at this shape, or nil if the
    /// engine refuses the lowering.
    static func measure(form: Form, s: Int, k: Int, n: Int) -> Double? {
        guard let p0 = h3_ane_program_create_form(Int32(s), Int32(k), Int32(n), form.raw),
              let p1 = h3_ane_program_create_form(Int32(s), Int32(k), Int32(n), form.raw)
        else { return nil }
        defer { h3_ane_program_free(p0); h3_ane_program_free(p1) }

        guard let x0 = h3_ane_tensor_create(Int32(k), Int32(s)),
              let x1 = h3_ane_tensor_create(Int32(k), Int32(s)),
              let w0 = h3_ane_tensor_create(Int32(form.wRows), Int32(form.wWidth)),
              let w1 = h3_ane_tensor_create(Int32(form.wRows), Int32(form.wWidth)),
              let y0 = h3_ane_tensor_create(Int32(form.yRows), Int32(form.yWidth)),
              let y1 = h3_ane_tensor_create(Int32(form.yRows), Int32(form.yWidth))
        else { return nil }
        defer { for t in [x0, x1, w0, w1, y0, y1] { h3_ane_tensor_free(t) } }

        let xv = halves(k * s, seed: 0x2545_F491_4F6C_DD1D)
        let wv = halves(form.wRows * form.wWidth, seed: 0x9E37_79B9_7F4A_7C15)
        guard write(x0, xv, rows: k, width: s), write(x1, xv, rows: k, width: s),
              write(w0, wv, rows: form.wRows, width: form.wWidth),
              write(w1, wv, rows: form.wRows, width: form.wWidth),
              h3_ane_run_pair(p0, x0, w0, y0, p1, x1, w1, y1)
        else { return nil }

        var samples: [Double] = []
        for _ in 0 ..< 5 {
            let t = Date()
            guard h3_ane_run_pair(p0, x0, w0, y0, p1, x1, w1, y1) else { return nil }
            samples.append(Date().timeIntervalSince(t))
        }
        return median(samples)
    }

    /// Does an engine evaluation actually run beside GPU work, or do they take
    /// turns?
    ///
    /// Every schedule in this tree rests on the answer, and it has only ever
    /// been measured for the GPU *shard* of a routed projection — 20.2 ms
    /// overlapped against 39.9 serial — which is a 20 ms kernel, not the 441 ms
    /// attention the CFG schedule is trying to hide the engine behind. If the
    /// two units share something that serialises at this scale, no amount of
    /// reordering will help and the ceiling is not the ceiling.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func engineBesideGPU() throws {
        let s = Self.s, k = Self.k, n = Self.n
        guard let p0 = h3_ane_program_create(Int32(s), Int32(k), Int32(n)),
              let p1 = h3_ane_program_create(Int32(s), Int32(k), Int32(n)),
              let x0 = h3_ane_tensor_create(Int32(k), Int32(s)),
              let w0 = h3_ane_tensor_create(Int32(k), Int32(n)),
              let w1 = h3_ane_tensor_create(Int32(k), Int32(n)),
              let y0 = h3_ane_tensor_create(Int32(s), Int32(n)),
              let y1 = h3_ane_tensor_create(Int32(s), Int32(n))
        else { Issue.record("setup failed"); return }
        defer {
            h3_ane_program_free(p0); h3_ane_program_free(p1)
            for t in [x0, w0, w1, y0, y1] { h3_ane_tensor_free(t) }
        }
        _ = Self.write(x0, Self.halves(k * s, seed: 1), rows: k, width: s)
        _ = Self.write(w0, Self.halves(k * n, seed: 2), rows: k, width: n)
        _ = Self.write(w1, Self.halves(k * n, seed: 3), rows: k, width: n)

        // GPU work of roughly one block's attention, in FLOPs.
        let a = (MLXRandom.normal([8192, 8192]) * 0.02).asType(.bfloat16)
        let b = (MLXRandom.normal([8192, 8192]) * 0.02).asType(.bfloat16)
        func gpuBurst() -> MLXArray {
            var acc = MLX.matmul(a, b)
            for _ in 0 ..< 5 { acc = MLX.matmul(acc.asType(.bfloat16), b) }
            return acc
        }

        func time(_ body: () -> Void) -> Double {
            var v: [Double] = []
            for _ in 0 ..< 5 { let t = Date(); body(); v.append(Date().timeIntervalSince(t)) }
            return Self.median(v) * 1000
        }

        MLX.eval(gpuBurst())
        let aneOnly = time { _ = h3_ane_run_pair(p0, x0, w0, y0, p1, x0, w1, y1) }
        let gpuOnly = time { MLX.eval(gpuBurst()) }
        let both = time {
            let out = Stream.withNewDefaultStream(device: .gpu) { () -> MLXArray in
                let v = gpuBurst(); MLX.asyncEval(v); return v
            }
            _ = h3_ane_run_pair(p0, x0, w0, y0, p1, x0, w1, y1)
            MLX.eval(out)
        }
        print(String(format: """

          engine alone      %7.1f ms
          GPU alone         %7.1f ms
          both together     %7.1f ms
          serial would be   %7.1f ms   ideal would be %7.1f ms
          overlap recovered %6.1f%% of the smaller job

        """, aneOnly, gpuOnly, both, aneOnly + gpuOnly, max(aneOnly, gpuOnly),
             100 * (aneOnly + gpuOnly - both) / min(aneOnly, gpuOnly)))
    }

    /// The two lowerings must agree, or a rate is a number about nothing.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func loweringsAgree() throws {
        let s = Self.s, k = Self.k, n = Self.n

        let flops = 2.0 * Double(s) * Double(k) * Double(n)
        print(String(format: "\n  production shard s=%d k=%d n=%d — %.3f TFLOP a die\n",
                     s, k, n, flops / 1e12))

        // x is [k, s] for both forms: the engine wants sequence minor, and that
        // is the one thing the two lowerings agree on.
        let xValues = Self.halves(k * s, seed: 0x2545_F491_4F6C_DD1D)
        // The weight differs only in orientation. Building both from one
        // logical [k, n] matrix is what makes the outputs comparable at all.
        let wKN = Self.halves(k * n, seed: 0x9E37_79B9_7F4A_7C15)
        var wNK = [Float16](repeating: 0, count: k * n)
        for row in 0 ..< k {
            for col in 0 ..< n { wNK[col * k + row] = wKN[row * n + col] }
        }

    struct Form {
            let name: String
            let raw: H3ANEForm
            let wRows: Int, wWidth: Int
            let yRows: Int, yWidth: Int
        }
        let forms = [
            Form(name: "matmul", raw: H3ANEFormMatmul,
                 wRows: k, wWidth: n, yRows: s, yWidth: n),
            Form(name: "conv 1x1", raw: H3ANEFormConv,
                 wRows: n, wWidth: k, yRows: n, yWidth: s),
        ]

        var results: [String: MLXArray] = [:]
        var rates: [String: Double] = [:]

        for form in forms {
            let t0 = Date()
            guard let p0 = h3_ane_program_create_form(Int32(s), Int32(k), Int32(n), form.raw),
                  let p1 = h3_ane_program_create_form(Int32(s), Int32(k), Int32(n), form.raw)
            else {
                print("  \(form.name): the engine refused this lowering")
                continue
            }
            defer { h3_ane_program_free(p0); h3_ane_program_free(p1) }
            let compileMs = Date().timeIntervalSince(t0) * 1000

            guard let x0 = h3_ane_tensor_create(Int32(k), Int32(s)),
                  let x1 = h3_ane_tensor_create(Int32(k), Int32(s)),
                  let w0 = h3_ane_tensor_create(Int32(form.wRows), Int32(form.wWidth)),
                  let w1 = h3_ane_tensor_create(Int32(form.wRows), Int32(form.wWidth)),
                  let y0 = h3_ane_tensor_create(Int32(form.yRows), Int32(form.yWidth)),
                  let y1 = h3_ane_tensor_create(Int32(form.yRows), Int32(form.yWidth))
            else { Issue.record("surface allocation failed"); return }
            defer {
                for t in [x0, x1, w0, w1, y0, y1] { h3_ane_tensor_free(t) }
            }

            let wValues = form.raw == H3ANEFormConv ? wNK : wKN
            for (t, rows, width) in [(x0, k, s), (x1, k, s)] {
                #expect(Self.write(t, xValues, rows: rows, width: width))
            }
            for t in [w0, w1] {
                #expect(Self.write(t, wValues, rows: form.wRows, width: form.wWidth))
            }

            guard h3_ane_run_pair(p0, x0, w0, y0, p1, x1, w1, y1) else {
                print("  \(form.name): evaluation failed")
                continue
            }

            // [n, s] from the conv form is the transpose of what matmul gives,
            // so both are compared in [s, n].
            let out = Self.read(y0, rows: form.yRows, width: form.yWidth)
            results[form.name] = form.raw == H3ANEFormConv
                ? MLX.contiguous(out.transposed()).asType(.float32)
                : out.asType(.float32)
            MLX.eval(results[form.name]!)

            var one: [Double] = [], pair: [Double] = []
            for _ in 0 ..< 5 {
                var t = Date()
                _ = h3_ane_run(p0, x0, w0, y0, 1)
                one.append(Date().timeIntervalSince(t))
                t = Date()
                _ = h3_ane_run_pair(p0, x0, w0, y0, p1, x1, w1, y1)
                pair.append(Date().timeIntervalSince(t))
            }
            let oneMs = Self.median(one), pairMs = Self.median(pair)
            rates[form.name] = 2 * flops / pairMs / 1e12
            print(String(format: "  %-9@  compile %6.0f ms   one die %6.1f ms (%.2f TF/s)"
                         + "   both %6.1f ms (%.2f TF/s)",
                         form.name as NSString, compileMs,
                         oneMs * 1000, flops / oneMs / 1e12,
                         pairMs * 1000, 2 * flops / pairMs / 1e12))
        }

        // Same arithmetic or the rate is irrelevant.
        if let a = results["matmul"], let b = results["conv 1x1"] {
            let diff = MLX.abs(a - b).max().item(Float.self)
            let scale = MLX.abs(a).max().item(Float.self)
            print(String(format: "\n  agreement: max |matmul - conv| = %.3e against a max |y| of %.1f",
                         diff, scale))
            #expect(diff / scale < 2e-3, "the two lowerings must compute the same projection")
        }

        if let m = rates["matmul"], let c = rates["conv 1x1"] {
            print(String(format: "  conv is %.3fx the matmul lowering\n", c / m))
        }
    }
}
