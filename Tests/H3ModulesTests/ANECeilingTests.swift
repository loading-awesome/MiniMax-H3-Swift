// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import CoreML
import H3Foundation

/// What does the Neural Engine reach at *this model's* GEMM shapes?
///
/// `gemmCeiling` established the GPU rate (16.0 TFLOP/s, and that is MLX's
/// ceiling rather than the model's overhead). `hardwareCeiling` established
/// that a second vendor implementation agrees. Both measured one processor.
/// This measures the other one, because the ANE is not a faster GPU — it is an
/// *additional* compute unit, and the arithmetic that matters is additive:
///
///     forward = 961 TFLOP     attention = 355 (GPU only)   linears = 606
///     T = 355/16 + 606/(16 + R)
///
/// Attention is a serial GPU phase, not something the engine can be hidden
/// behind: a block is a chain and block i+1 consumes block i, so both dies sit
/// idle for the 22.2 s attention costs no matter how fast they are. Only the
/// linears are splittable, which caps the whole exercise at 2.7x even with an
/// infinitely fast engine.
///
/// So `R`, the ANE's achieved rate at production `K`/`N`, is the single number
/// the whole question turns on, and nothing published answers it. Apple does
/// not quote ANE throughput for M3 Ultra at all. The one reverse-engineered
/// account (Bryngelson 2026, arXiv:2606.22283) measures M1 and M5 and names
/// the A15/M3 generation "the one rail that remains unmeasured". oMLX
/// (`jundot/omlx`, Apache 2.0) drives both dies in production for Qwen prefill
/// and gets 1.356x, but at Qwen's MLP shapes, not these.
///
/// **The outcomes lead to different work:**
///
///  * **R at or below ~6 TFLOP/s.** A 1.1–1.35x ceiling on the whole forward,
///    bought with private APIs that break on macOS updates and an INT8
///    requantisation that contract 8 forbids. Dead, and cheaply.
///  * **R near 20 TFLOP/s.** A 2.2x ceiling — larger than every remaining item
///    in `docs/PERF_ROADMAP.md` combined. Then the residency problem is worth
///    solving, and only then.
///
/// The specific thing to catch is Bryngelson §11.3: the engine's fused matmul
/// "stalls on weight streaming once a square operand passes its on-chip working
/// set near N of 2048". Production `N` is 21,504 (qkv) and 28,672 (fc1) — an
/// order of magnitude past that. If the stall is real at these widths, `R`
/// lands at the bottom of the range and this is over. `stall control N=2048`
/// in the table below is the same `K` at a width the engine is said to like, so
/// the two rows bracket the claim rather than assuming it.
///
///     H3_BIG=1 swift test --filter aneCeiling
///
/// Expect 5–15 minutes and about 800 MB of temporary files: the probe writes a
/// real `[N,K]` fp16 weight blob per shape, because a GEMM whose weights fit in
/// cache is not the GEMM this model runs. Set `H3_ANE_TILES=512,2048,4096` to
/// sweep the sequence tile; the default is 2,048, which is the fixed shape oMLX
/// found best and is safely under the 16,384 spatial-extent cap that the M3
/// generation still has (it rises to 65,536 only at M4).
///
/// Never part of the normal suite, for the same reason as the other two: it is
/// a benchmark, and a timing assertion that fails under contention teaches
/// people to ignore failures.
@Suite("ANE ceiling", .serialized)
struct ANECeilingTests {

    // MARK: - Timing
    //
    // Deliberately not `BenchmarkSupport`: those helpers force an MLX graph
    // with `MLX.eval`, and a Core ML prediction has no graph to force. Same
    // ABBA protocol and same median, against a different execution model.

    private static func median(_ values: [Double]) -> Double {
        precondition(!values.isEmpty)
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }

    private static func interleaved(
        rounds: Int = 3, first: () throws -> Void, second: () throws -> Void
    ) rethrows -> (first: Double, second: Double) {
        try first()                                     // warm up: compile, page in
        try second()
        var firstSamples: [Double] = [], secondSamples: [Double] = []

        func sample(_ body: () throws -> Void, into samples: inout [Double]) rethrows {
            let t0 = Date()
            try body()
            samples.append(Date().timeIntervalSince(t0))
        }
        for round in 0 ..< rounds {
            if round.isMultiple(of: 2) {
                try sample(first, into: &firstSamples)
                try sample(second, into: &secondSamples)
                try sample(second, into: &secondSamples)
                try sample(first, into: &firstSamples)
            } else {
                try sample(second, into: &secondSamples)
                try sample(first, into: &firstSamples)
                try sample(first, into: &firstSamples)
                try sample(second, into: &secondSamples)
            }
        }
        return (median(firstSamples), median(secondSamples))
    }

    // MARK: - The measurement

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func aneCeiling() async throws {
        let cfg = H3Config()
        let h = cfg.hiddenSize                          // 5376
        let inner = cfg.innerDim                        // 7168
        let ffn = cfg.ffnHidden                         // 14336

        // The four weight matmuls a block runs, at production K and N. Only S
        // is tiled, and only because a linear is row-independent: tiling S is
        // exact, and it is what any real implementation would do against the
        // engine's fixed-shape programs.
        //
        // `production` is what feeds the projection below. The two controls
        // exist to bracket the weight-streaming claim, and letting a friendly
        // 2048-wide shape set the headline rate would be the same mistake §8
        // already made once by quoting a square GEMM as the model's reference.
        let shapes: [(name: String, k: Int, n: Int, production: Bool)] = [
            ("qkv        [H,3I]", h, 3 * inner, true),
            ("attn out   [I,H]", inner, h, true),
            ("mlp fc1    [H,2F]", h, 2 * ffn, true),
            ("mlp fc2    [F,H]", ffn, h, true),
            ("stall ctl  [H,2048]", h, 2_048, false),
            ("square ctl [2048,2048]", 2_048, 2_048, false)
        ]

        let tiles = (ProcessInfo.processInfo.environment["H3_ANE_TILES"]
            .map { $0.split(separator: ",").compactMap { Int($0) } })
            .flatMap { $0.isEmpty ? nil : $0 } ?? [2_048]

        let scratch = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("h3-ane-ceiling-\(ProcessInfo.processInfo.processIdentifier)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        print("""

          the model achieves 16.0 TFLOP/s on the GPU (961 TFLOP in 60.0 s)
          attention is 355 TFLOP of that and stays on the GPU in any split

        """)
        if let engine = Self.neuralEngineDescription() {
            print("  \(engine)\n")
        }

        var best = 0.0
        for tile in tiles {
            print("  sequence tile S = \(tile)")
            print("  " + "shape".padded(to: 24)
                  + "      ANE ms   ANE TF/s      CPU ms    CPU TF/s   placement")

            for shape in shapes {
                let flops = 2.0 * Double(tile) * Double(shape.k) * Double(shape.n)
                let built: BuiltModel
                do {
                    built = try await Self.build(k: shape.k, n: shape.n, s: tile, in: scratch)
                } catch {
                    print("  " + shape.name.padded(to: 24)
                          + "      compile failed: \(Self.reason(error))")
                    continue
                }
                defer { try? FileManager.default.removeItem(at: built.compiled) }

                let aneConfig = MLModelConfiguration()
                aneConfig.computeUnits = .cpuAndNeuralEngine
                let cpuConfig = MLModelConfiguration()
                cpuConfig.computeUnits = .cpuOnly

                let placement = await Self.placement(of: built.compiled, configuration: aneConfig)

                let aneModel = try MLModel(contentsOf: built.compiled, configuration: aneConfig)
                let cpuModel = try MLModel(contentsOf: built.compiled, configuration: cpuConfig)
                let input = try Self.input(k: shape.k, s: tile, dataType: built.ioDataType)

                let measured = try Self.interleaved(
                    first: { _ = try aneModel.prediction(from: input) },
                    second: { _ = try cpuModel.prediction(from: input) })

                // A rate only counts toward the projection when the layer
                // actually ran on the engine. A production shape Core ML
                // declined to place has measured the CPU, and folding that
                // into an ANE ceiling would report the wrong processor.
                let aneRate = flops / measured.first / 1e12
                if shape.production, placement.contains("ANE") {
                    best = max(best, aneRate)
                }
                print(String(format: "  %@ %11.1f %10.2f %11.1f %11.2f   %@",
                             shape.name.padded(to: 24),
                             measured.first * 1_000, aneRate,
                             measured.second * 1_000,
                             flops / measured.second / 1e12,
                             placement))
            }
            print("")
        }

        Self.report(best: best)
    }

    // MARK: - What the number means

    /// Turns the measured rate into the only thing anyone wants from it: the
    /// ceiling on a full forward, under the placement every design implies —
    /// attention on the GPU, linears split so both units finish together.
    private static func report(best: Double) {
        guard best > 0 else {
            print("""
              No production shape ran on the engine, so there is no rate to
              project from. That is an answer and not a failure: Core ML will
              not place this model's GEMMs on the ANE, and the direct route
              below Core ML would have to clear the same shape limits before
              anything else about it matters.

            """)
            return
        }
        let attention = 355.0, linears = 606.0, gpu = 16.0
        print("  best measured ANE rate: \(String(format: "%.2f", best)) TFLOP/s\n")
        print("  implied ceiling on a 60.0 s forward, attention pinned to the GPU:")
        print("  " + "ANE rate".padded(to: 16) + "linears on ANE    forward    speedup"
              + "    int8 weights")

        for (label, rate) in [("as measured (1 die)", best), ("both dies", best * 2)] {
            guard rate > 0 else { continue }
            // Attention cannot overlap the linears. A block is a chain —
            // qkv, attention, attn out, fc1, fc2 — and block i+1 consumes
            // block i, so there is no independent work to hide attention
            // behind. It is a serial GPU-only phase with both dies idle, and
            // a model that lets the engine eat into it reports a forward the
            // dependency graph cannot produce.
            //
            //   forward = A/G  +  L/(G + R)
            //
            // The linears balance at f·L/R == (1-f)·L/G, i.e. f = R/(G + R),
            // which is a *smaller* ANE share than the overlapped model asks
            // for, and correspondingly less weight that has to be resident.
            let fraction = rate / (gpu + rate)
            let forward = attention / gpu + linears / (gpu + rate)
            print(String(format: "  %@%13.0f%%%10.1f s%9.2fx%10.1f GB",
                         label.padded(to: 16), fraction * 100, forward,
                         60.0 / forward, fraction * 19.3))
        }
        print("""

          The last column is what must be resident on the engine to reach that
          row. oMLX measured the per-instance address window at about 4 GiB, so
          8.6 GB is what two dies hold at once. A row that fits needs no bank
          cycling and could be built against the window as it stands; a row that
          does not is the case where cycling has to be measured before anything
          is built on top of it.

          Neither is a reason to start until the precision question is answered
          separately: contract 8 pins the DiT at bf16, the engine is fp16 with a
          reduction error that grows with K, and fc2 contracts over 14,336 —
          1000 times per render.

        """)
    }

    // MARK: - Placement

    private static func neuralEngineDescription() -> String? {
        guard #available(macOS 14.0, *) else { return nil }
        for device in MLComputeDevice.allComputeDevices {
            if case .neuralEngine(let engine) = device {
                return "Core ML reports a Neural Engine with \(engine.totalCoreCount) cores"
                    + " (on an Ultra the driver steers whole submissions to one die;"
                    + " it never splits a tensor across the bridge)"
            }
        }
        return "Core ML reports no Neural Engine on this machine"
    }

    /// Whether the layer actually landed on the engine, asked of the compute
    /// plan rather than inferred from a stopwatch. A shape Core ML refuses to
    /// place is itself an answer, and it is not the same answer as a shape that
    /// lands on the engine and runs slowly.
    private static func placement(
        of compiled: URL, configuration: MLModelConfiguration
    ) async -> String {
        guard #available(macOS 14.4, *) else { return "unknown (needs macOS 14.4)" }
        do {
            let plan = try await MLComputePlan.load(contentsOf: compiled,
                                                    configuration: configuration)
            guard case .neuralNetwork(let network) = plan.modelStructure else {
                return "not a neural network"
            }
            var names: [String] = []
            for layer in network.layers {
                guard let usage = plan.deviceUsage(for: layer) else { continue }
                switch usage.preferred {
                case .neuralEngine: names.append("ANE")
                case .gpu:          names.append("GPU")
                case .cpu:          names.append("CPU")
                @unknown default:   names.append("?")
                }
            }
            return names.isEmpty ? "no layers" : names.joined(separator: "+")
        } catch {
            return "plan failed: \(reason(error))"
        }
    }

    private static func reason(_ error: Error) -> String {
        let text = (error as NSError).localizedDescription
        return text.count > 60 ? String(text.prefix(60)) + "…" : text
    }

    // MARK: - Building a Core ML model without coremltools

    private struct BuiltModel {
        let compiled: URL
        let ioDataType: MLMultiArrayDataType
    }

    /// Emits the `.mlmodel` protobuf directly and compiles it.
    ///
    /// Hand-writing the specification keeps this runnable with `swift test` and
    /// nothing else. The alternative is a coremltools dependency, i.e. a Python
    /// environment, for a tree whose entire premise is not having one — and a
    /// measurement nobody can run is a measurement nobody runs.
    ///
    /// fp16 input and output are tried first and fp32 is the fallback, because
    /// an fp32 boundary makes Core ML convert a `[N,1,S]` tensor on every call:
    /// at fc1 that is 235 MB of conversion wrapped around the arithmetic being
    /// timed. Which one was used is reported, since it changes what the number
    /// means.
    private static func build(k: Int, n: Int, s: Int, in scratch: URL) async throws -> BuiltModel {
        let weights = weightBlob(count: k * n)
        var lastError: Error?
        for dataType in [ArrayDataType.float16, ArrayDataType.float32] {
            let url = scratch.appendingPathComponent("gemm-\(k)x\(n)-s\(s)-\(dataType.rawValue).mlmodel")
            do {
                let spec = specification(k: k, n: n, s: s,
                                         ioDataType: dataType, weights: weights)
                try Data(spec).write(to: url)
                let compiled = try await MLModel.compileModel(at: url)
                try? FileManager.default.removeItem(at: url)
                return BuiltModel(compiled: compiled,
                                  ioDataType: dataType == .float16 ? .float16 : .float32)
            } catch {
                lastError = error
                try? FileManager.default.removeItem(at: url)
            }
        }
        throw lastError ?? ProbeFailure(
            detail: "neither fp16 nor fp32 I/O compiled for [\(k),\(n)] at S=\(s)")
    }

    private enum ArrayDataType: UInt64 {
        case float32 = 65_568
        case float16 = 65_552
    }

    /// A GEMM spelled the way this engine wants it: a 1x1 convolution over a
    /// `[K, 1, S]` input, contracting K into N.
    ///
    /// This is not stylistic. Convolution input- and output-channel extents are
    /// the one pair of axes that escape the tensor-extent cap the M3 generation
    /// still has — `Cin` and `Cout` of 16,385 compile where a spatial axis of
    /// 16,385 is rejected. Production `N` is 21,504 and 28,672, so an
    /// `innerProduct` spelling of the same arithmetic would be refused outright
    /// at two of the four shapes and the table would silently be measuring
    /// something else.
    private static func specification(
        k: Int, n: Int, s: Int, ioDataType: ArrayDataType, weights: [UInt8]
    ) -> [UInt8] {
        var model = ProtoWriter()
        model.reserve(weights.count + 4_096)
        model.varintField(1, 7)                              // specificationVersion
        model.messageField(2) { description in               // ModelDescription
            description.messageField(1) { feature in         // input
                feature.stringField(1, "x")
                feature.messageField(3) { type in
                    type.messageField(5) { array in          // multiArrayType
                        array.packedVarints(1, [UInt64(k), 1, UInt64(s)])
                        array.varintField(2, ioDataType.rawValue)
                    }
                }
            }
            description.messageField(10) { feature in        // output
                feature.stringField(1, "y")
                feature.messageField(3) { type in
                    type.messageField(5) { array in
                        array.packedVarints(1, [UInt64(n), 1, UInt64(s)])
                        array.varintField(2, ioDataType.rawValue)
                    }
                }
            }
        }
        model.messageField(500) { network in                 // NeuralNetwork
            network.messageField(1) { layer in               // layers[0]
                layer.stringField(1, "gemm")
                layer.stringField(2, "x")                    // input
                layer.stringField(3, "y")                    // output
                layer.messageField(100) { conv in            // ConvolutionLayerParams
                    conv.varintField(1, UInt64(n))           // outputChannels
                    conv.varintField(2, UInt64(k))           // kernelChannels
                    conv.varintField(10, 1)                  // nGroups
                    conv.packedVarints(20, [1, 1])           // kernelSize
                    conv.packedVarints(30, [1, 1])           // stride
                    conv.packedVarints(40, [1, 1])           // dilationFactor
                    conv.messageField(50) { _ in }           // valid padding
                    conv.messageField(90) { params in        // weights
                        params.bytesField(2, weights)        // float16Value
                    }
                }
            }
        }
        return model.bytes
    }

    /// `[N, K, 1, 1]` fp16 weights. Deterministic, and deliberately neither
    /// zero nor denormal: a weight blob of zeros measures whatever fast path
    /// the hardware keeps for zeros, which is not the question.
    private static func weightBlob(count: Int) -> [UInt8] {
        var blob = [UInt8](repeating: 0, count: count * 2)
        blob.withUnsafeMutableBytes { raw in
            let halves = raw.bindMemory(to: UInt16.self)
            var state: UInt64 = 0x2545_F491_4F6C_DD1D
            for index in 0 ..< count {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                let unit = Float((state >> 40) & 0xFFFF) / 65_535.0
                halves[index] = Float16(0.02 + 0.06 * unit).bitPattern
            }
        }
        return blob
    }

    private static func input(
        k: Int, s: Int, dataType: MLMultiArrayDataType
    ) throws -> MLDictionaryFeatureProvider {
        let array = try MLMultiArray(
            shape: [NSNumber(value: k), 1, NSNumber(value: s)], dataType: dataType)
        array.withUnsafeMutableBytes { raw, _ in
            switch dataType {
            case .float16:
                let halves = raw.bindMemory(to: UInt16.self)
                let value = Float16(0.05).bitPattern
                for index in 0 ..< halves.count { halves[index] = value }
            default:
                let floats = raw.bindMemory(to: Float.self)
                for index in 0 ..< floats.count { floats[index] = 0.05 }
            }
        }
        return try MLDictionaryFeatureProvider(
            dictionary: ["x": MLFeatureValue(multiArray: array)])
    }
}

private struct ProbeFailure: Error, CustomStringConvertible {
    let detail: String
    var description: String { "ANE probe could not build a model: \(detail)" }
}

// MARK: - Protobuf

/// Just enough of the protobuf wire format to emit one Core ML specification.
///
/// proto3 omits zero-valued scalars, and matching that matters rather than
/// being pedantic: a `hasBias: false` written explicitly is a different message
/// from one omitted, and the on-device compiler reads the difference.
private struct ProtoWriter {
    private(set) var bytes: [UInt8] = []

    mutating func reserve(_ count: Int) { bytes.reserveCapacity(count) }

    mutating func varint(_ value: UInt64) {
        var remaining = value
        repeat {
            var byte = UInt8(remaining & 0x7F)
            remaining >>= 7
            if remaining != 0 { byte |= 0x80 }
            bytes.append(byte)
        } while remaining != 0
    }

    mutating func tag(_ field: Int, _ wireType: UInt8) {
        varint(UInt64(field) << 3 | UInt64(wireType))
    }

    mutating func varintField(_ field: Int, _ value: UInt64) {
        guard value != 0 else { return }
        tag(field, 0)
        varint(value)
    }

    mutating func bytesField(_ field: Int, _ payload: [UInt8]) {
        tag(field, 2)
        varint(UInt64(payload.count))
        bytes.append(contentsOf: payload)
    }

    mutating func stringField(_ field: Int, _ value: String) {
        bytesField(field, Array(value.utf8))
    }

    mutating func messageField(_ field: Int, _ build: (inout ProtoWriter) -> Void) {
        var inner = ProtoWriter()
        build(&inner)
        bytesField(field, inner.bytes)
    }

    /// proto3 packs repeated scalars by default, and Core ML's `.proto` files
    /// are proto3 — an unpacked `kernelSize` is read as a malformed message.
    mutating func packedVarints(_ field: Int, _ values: [UInt64]) {
        var inner = ProtoWriter()
        for value in values { inner.varint(value) }
        bytesField(field, inner.bytes)
    }
}

private extension String {
    func padded(to length: Int) -> String {
        (self as NSString).padding(toLength: length, withPad: " ", startingAt: 0)
    }
}
