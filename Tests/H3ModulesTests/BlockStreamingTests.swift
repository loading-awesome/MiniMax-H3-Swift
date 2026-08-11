// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import Darwin
import MLX
import H3Foundation
@testable import H3Modules

/// Can a DiT block's weights be read from disk while the previous block is
/// still computing?
///
/// **This decides whether streaming exists**, and like `gemmCeiling` it costs
/// minutes rather than the weeks a production implementation would. The
/// motivating observation came from `antirez/h3.c`, which keeps two blocks
/// resident and reads the rest during GPU execution: 36.5 GiB of residency
/// becomes 2.0 GiB, at 26-84% slower. That penalty is a property of *their*
/// shape, not of streaming — 512x512x22 leaves about 37 ms of compute per block
/// to hide the read behind. This tree's control shape leaves 1.20 s.
///
/// The arithmetic that motivates the measurement, from the control sweep's
/// 60.0 s forward and the checkpoint header:
///
/// | | bytes/forward | at 3.18 GB/s | vs 60.0 s compute |
/// |---|---|---|---|
/// | every block, whole | 66.3 GB | 20.8 s | 35% |
/// | AdaLN precomputed | 38.5 GB | 12.1 s | 20% |
///
/// **AdaLN is 520.4 MB of each block's 1291.1 MB and 0.008% of its
/// arithmetic.** `docs/PERF_ROADMAP.md` records those 26 GB as weights that
/// "have to be read whatever happens" — true when reading everything costs 0.1%
/// of a step, false the moment reads are the constraint. `H3Transformer`
/// derives `tEmb` from `plan.values` alone, so every block's modulation is a
/// function of the timestep schedule and nothing else: it can be computed once
/// per render and the 26 GB never read again.
///
/// This suite therefore measures the tail alone — the 770.7 MB of attention and
/// MLP weights that genuinely must arrive per block per forward.
///
///     H3_BIG=1 swift test --filter blockStreaming
///
/// Never part of the normal suite: it needs the 66 GB checkpoint, it allocates
/// at production width, and it is meaningless under contention.
@Suite("block streaming", .serialized)
struct BlockStreamingTests {

    // MARK: - Checkpoint discovery

    /// The checkpoint, from `H3_CHECKPOINT` or the configured root. Absent
    /// means the suite skips rather than fails: not every machine that runs the
    /// tests holds 66 GB of weights.
    static let checkpoint: URL? = {
        let env = ProcessInfo.processInfo.environment
        if let p = env["H3_CHECKPOINT"] { return URL(fileURLWithPath: p) }
        let config = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/minimax-h3/config.json")
        guard let data = try? Data(contentsOf: config),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let ck = obj["checkpoints"] as? [String: Any],
              let root = ck["root"] as? String,
              let fl = ck["fl2va"] as? [String: Any],
              let name = fl["bf16"] as? String
        else { return nil }
        let url = URL(fileURLWithPath: root).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }()

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["H3_BIG"] != nil && checkpoint != nil
    }

    // MARK: - On-disk block geometry

    /// Where one block's tensors live in the file.
    ///
    /// Every block's tensors are contiguous, and `adaln_proj` sits at the
    /// *start* of the span — bias then weight, offsets +0.00 and +0.19 MB. That
    /// is what makes this cheap: skipping AdaLN leaves one contiguous read
    /// rather than a scatter, so a streamer issues a single `pread` per block.
    struct BlockSpan {
        let index: Int
        let start: Int           // absolute file offset, whole block
        let end: Int
        let tailStart: Int       // absolute file offset, first non-AdaLN byte
        var wholeBytes: Int { end - start }
        var tailBytes: Int { end - tailStart }
        /// Tensor name -> (offset within the tail, shape).
        let tail: [String: (offset: Int, shape: [Int])]
    }

    static func spans(_ archive: Safetensors.Archive, blobStart: Int, count: Int)
        -> [BlockSpan] {
        (0 ..< count).map { i in
            let prefix = "blocks.\(i)."
            let mine = archive.tensors.filter { $0.key.hasPrefix(prefix) }
            precondition(!mine.isEmpty, "checkpoint has no \(prefix)*")
            let start = blobStart + mine.values.map(\.begin).min()!
            let end = blobStart + mine.values.map(\.end).max()!
            let adalnEnd = blobStart + mine.filter { $0.key.contains("adaln_proj") }
                .values.map(\.end).max()!
            var tail: [String: (Int, [Int])] = [:]
            for (name, info) in mine where !name.contains("adaln_proj") {
                let absolute = blobStart + info.begin
                precondition(absolute >= adalnEnd,
                             "\(name) precedes the AdaLN weights; the tail is not contiguous")
                tail[String(name.dropFirst(prefix.count))] =
                    (absolute - adalnEnd, info.shape)
            }
            return BlockSpan(index: i, start: start, end: end, tailStart: adalnEnd, tail: tail)
        }
    }

    /// `Safetensors.Archive` keeps `blobStart` private and payload-relative
    /// offsets are all production needs. Recomputing it here costs 8 bytes and
    /// avoids widening a production API for a benchmark.
    static func blobStart(of url: URL) throws -> Int {
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        guard let head = try fh.read(upToCount: 8), head.count == 8 else {
            throw Safetensors.Error.tooSmall
        }
        return 8 + Int(head.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian })
    }

    // MARK: - Uncached reads

    /// A reader that the page cache cannot answer.
    ///
    /// This machine has 256 GB of RAM and the checkpoint is 66 GB, so a cached
    /// read reports memory bandwidth and the whole experiment becomes a
    /// tautology. `F_NOCACHE` is what makes the number mean anything, and it is
    /// also what a real streamer wants: caching pages it will not re-read
    /// evicts the activations it will.
    final class UncachedReader: @unchecked Sendable {
        private let fd: Int32
        init(_ url: URL) throws {
            fd = open(url.path, O_RDONLY)
            guard fd >= 0 else { throw POSIXError(.EIO) }
            guard fcntl(fd, F_NOCACHE, 1) >= 0 else { throw POSIXError(.EIO) }
        }
        deinit { close(fd) }

        /// Fills `buffer` from `offset`. `pread` keeps this thread-safe against
        /// a concurrent reader on the same descriptor.
        func read(into buffer: UnsafeMutableRawPointer, offset: Int, count: Int) {
            var done = 0
            while done < count {
                let n = pread(fd, buffer.advanced(by: done), count - done, off_t(offset + done))
                precondition(n > 0, "short read at \(offset + done): \(String(cString: strerror(errno)))")
                done += n
            }
        }
    }

    /// Page-aligned host staging buffer.
    final class Staging: @unchecked Sendable {
        let pointer: UnsafeMutableRawPointer
        let capacity: Int
        init(_ capacity: Int) {
            var p: UnsafeMutableRawPointer?
            precondition(posix_memalign(&p, 16384, capacity) == 0, "posix_memalign failed")
            self.pointer = p!
            self.capacity = capacity
        }
        deinit { free(pointer) }
    }

    // MARK: - Building a block out of bytes

    /// bf16 bytes in a host buffer -> `MLXArray`.
    ///
    /// `mlx_array_new_data` copies, so the staging buffer is free for the next
    /// prefetch the moment this returns. The copy is host-to-host at memory
    /// bandwidth; it is counted in the streamed arm's wall clock, not excluded
    /// from it.
    static func array(_ base: UnsafeMutableRawPointer, _ offset: Int, _ shape: [Int])
        -> MLXArray {
        let count = shape.reduce(1, *)
        let words = base.advanced(by: offset).bindMemory(to: UInt16.self, capacity: count)
        return MLXArray(UnsafeBufferPointer(start: words, count: count))
            .view(dtype: .bfloat16)
            .reshaped(shape)
    }

    /// A block whose attention and MLP weights come from `base`, sharing one
    /// resident AdaLN.
    ///
    /// **The shared AdaLN is deliberate and it is what isolates the variable.**
    /// Both arms run the identical AdaLN projection on identical weights, so
    /// its cost cancels; the only difference between them is whether the
    /// 770.7 MB of attention and MLP weights were resident or had just arrived
    /// from disk. Modelling the real design — precompute the schedule, drop the
    /// weights — belongs in the implementation, not in the measurement that
    /// decides whether to write one.
    static func block(from base: UnsafeMutableRawPointer, span: BlockSpan,
                      adaln: AdalnProj, config: H3Config) -> DiTBlock {
        func w(_ name: String) -> MLXArray {
            guard let t = span.tail[name] else { preconditionFailure("no \(name) in block tail") }
            return array(base, t.offset, t.shape)
        }
        return DiTBlock(
            norm1: H3RMSNorm(weight: w("norm1.weight"), eps: config.normEps),
            norm2: H3RMSNorm(weight: w("norm2.weight"), eps: config.normEps),
            attn: AttentionLayer(qkvWeight: H3Weights.permuteQKV(w("attn.qkv_proj.weight"),
                                                                 heads: config.numHeads,
                                                                 headDim: config.headDim),
                                 outWeight: w("attn.out_proj.weight"),
                                 qNormWeight: w("attn.q_norm.weight"),
                                 kNormWeight: w("attn.k_norm.weight"),
                                 heads: config.numHeads, headDim: config.headDim,
                                 eps: config.qkNormEps),
            mlp: H3MLP(fc1: w("mlp.fc1.weight"), fc2: w("mlp.fc2.weight")),
            adaln: adaln)
    }

    // MARK: - Production-shape inputs

    struct Inputs {
        let x: MLXArray
        let tEmb: MLXArray
        let index: ModulationIndex
        let rope: MLXArray
        let adaln: AdalnProj
    }

    static func inputs(_ config: H3Config, seed: UInt64 = 11) -> Inputs {
        let s = 15_731                                   // 864x480x124, the control shape
        MLXRandom.seed(seed)
        let segs = [ModSegment(start: 0, stop: 746, row: 0),
                    ModSegment(start: 746, stop: s, row: 6)]
        let rope = H3RoPE.rotationTable(
            angles: MLXRandom.normal([s, 96]).asType(.float32)).asType(.bfloat16)
        return Inputs(
            x: (MLXRandom.normal([s, config.hiddenSize]) * 0.02).asType(.bfloat16),
            tEmb: (MLXRandom.normal([3, config.timeEmbedDim]) * 0.02).asType(.bfloat16),
            index: ModulationIndex(segments: segs, tokenCount: s),
            rope: rope,
            adaln: AdalnProj(
                weight: (MLXRandom.normal([config.adalnOutFeatures, config.timeEmbedDim])
                         * 0.02).asType(.bfloat16),
                bias: nil, expand: 6, modalities: 3, hidden: config.hiddenSize,
                computeFP32: true))
    }

    // MARK: - 1. What the disk gives

    @Test(.enabled(if: Self.enabled))
    func blockStreamingIO() throws {
        let url = Self.checkpoint!
        let archive = try Safetensors.Archive(url: url)
        let spans = try Self.spans(archive, blobStart: Self.blobStart(of: url),
                                   count: H3Config().numLayers)

        let whole = spans[0].wholeBytes, tail = spans[0].tailBytes
        print("\n  block span   whole \(String(format: "%.1f", Double(whole) / 1e6)) MB"
              + "   tail \(String(format: "%.1f", Double(tail) / 1e6)) MB"
              + "   AdaLN \(String(format: "%.1f%%", 100 * Double(whole - tail) / Double(whole)))")

        let reader = try Self.UncachedReader(url)
        let staging = Self.Staging(whole)

        // Distinct blocks each time: re-reading one span would let the drive's
        // own cache answer, which is the same tautology as the page cache.
        func rate(_ pick: (BlockSpan) -> (Int, Int), _ n: Int, from: Int) -> Double {
            var bytes = 0
            let t0 = Date()
            for i in 0 ..< n {
                let (offset, count) = pick(spans[from + i])
                reader.read(into: staging.pointer, offset: offset, count: count)
                bytes += count
            }
            return Double(bytes) / Date().timeIntervalSince(t0) / 1e9
        }

        let wholeRate = rate({ ($0.start, $0.wholeBytes) }, 6, from: 0)
        let tailRate = rate({ ($0.tailStart, $0.tailBytes) }, 6, from: 12)

        print(String(format: "  uncached     whole %.2f GB/s (%.3f s/block)"
                     + "   tail %.2f GB/s (%.3f s/block)",
                     wholeRate, Double(whole) / 1e9 / wholeRate,
                     tailRate, Double(tail) / 1e9 / tailRate))
        let layers = Double(H3Config().numLayers)
        print(String(format: "  per forward  whole %.1f s   tail %.1f s   "
                     + "(compute is 60.0 s at this shape)",
                     Double(whole) * layers / 1e9 / wholeRate,
                     Double(tail) * layers / 1e9 / tailRate))
    }

    // MARK: - 2. Does a streamed block give the same answer?

    /// A speed-up that changes the result is not a speed-up. Bytes read through
    /// `pread` and wrapped with `view(dtype:)` must produce exactly what
    /// `MLX.loadArrays` produces from the same file — this is a reinterpretation
    /// of identical bytes, so anything short of bit-identical means the offset
    /// arithmetic or the qkv permute is wrong, and both fail silently.
    @Test(.enabled(if: Self.enabled))
    func streamedBlockIsExact() throws {
        let url = Self.checkpoint!
        let config = H3Config()
        let archive = try Safetensors.Archive(url: url)
        let spans = try Self.spans(archive, blobStart: Self.blobStart(of: url),
                                   count: config.numLayers)
        let span = spans[0]
        let f = Self.inputs(config)

        let reader = try Self.UncachedReader(url)
        let staging = Self.Staging(span.tailBytes)
        reader.read(into: staging.pointer, offset: span.tailStart, count: span.tailBytes)
        let streamed = Self.block(from: staging.pointer, span: span, adaln: f.adaln,
                                  config: config)

        let weights = try H3Weights(url: url, strict: false)
        let resident = DiTBlock(
            norm1: H3RMSNorm(weight: try weights.block(0, "norm1.weight"), eps: config.normEps),
            norm2: H3RMSNorm(weight: try weights.block(0, "norm2.weight"), eps: config.normEps),
            attn: AttentionLayer(qkvWeight: try weights.block(0, "attn.qkv_proj.weight"),
                                 outWeight: try weights.block(0, "attn.out_proj.weight"),
                                 qNormWeight: try weights.block(0, "attn.q_norm.weight"),
                                 kNormWeight: try weights.block(0, "attn.k_norm.weight"),
                                 heads: config.numHeads, headDim: config.headDim,
                                 eps: config.qkNormEps),
            mlp: H3MLP(fc1: try weights.block(0, "mlp.fc1.weight"),
                       fc2: try weights.block(0, "mlp.fc2.weight")),
            adaln: f.adaln)

        let a = streamed(f.x, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
        let b = resident(f.x, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
        MLX.eval(a, b)
        let identical = MLX.all(a .== b).item(Bool.self)
        print("\n  streamed block bit-identical to resident: \(identical)")
        #expect(identical)
    }

    // MARK: - 3. Does the read hide behind the compute?

    @Test(.enabled(if: Self.enabled))
    func blockStreamingOverlap() throws {
        let url = Self.checkpoint!
        let config = H3Config()
        let archive = try Safetensors.Archive(url: url)
        let spans = try Self.spans(archive, blobStart: Self.blobStart(of: url),
                                   count: config.numLayers)
        let f = Self.inputs(config)
        let k = 8                                        // 8 x ~1.2 s per arm
        let reader = try Self.UncachedReader(url)
        let tailBytes = spans[0].tailBytes

        // --- Arm A: resident. Eight distinct blocks, warmed so the mmap has
        // already faulted in and we are timing compute alone.
        let weights = try H3Weights(url: url, strict: false)
        var residentBlocks: [DiTBlock] = []
        for i in 0 ..< k {
            residentBlocks.append(DiTBlock(
                norm1: H3RMSNorm(weight: try weights.block(i, "norm1.weight"), eps: config.normEps),
                norm2: H3RMSNorm(weight: try weights.block(i, "norm2.weight"), eps: config.normEps),
                attn: AttentionLayer(qkvWeight: try weights.block(i, "attn.qkv_proj.weight"),
                                     outWeight: try weights.block(i, "attn.out_proj.weight"),
                                     qNormWeight: try weights.block(i, "attn.q_norm.weight"),
                                     kNormWeight: try weights.block(i, "attn.k_norm.weight"),
                                     heads: config.numHeads, headDim: config.headDim,
                                     eps: config.qkNormEps),
                mlp: H3MLP(fc1: try weights.block(i, "mlp.fc1.weight"),
                           fc2: try weights.block(i, "mlp.fc2.weight")),
                adaln: f.adaln))
        }
        func residentPass() -> Double {
            var h = f.x
            let t0 = Date()
            for b in residentBlocks {
                h = b(h, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
            }
            MLX.eval(h)
            return Date().timeIntervalSince(t0)
        }
        _ = residentPass()                               // warm: fault the mapping, compile
        let resident = residentPass()

        // --- Arm B: streamed, double-buffered. Block i computes while block
        // i+1 arrives. Two staging buffers is the whole residency claim: the
        // GPU never holds more than two blocks' worth of attention/MLP weights.
        let wholeBytes = spans[0].wholeBytes
        let staging = [Self.Staging(wholeBytes), Self.Staging(wholeBytes)]
        let queue = DispatchQueue(label: "h3.block-prefetch", qos: .userInitiated)
        var waited = 0.0                                 // time the compute lost to I/O

        /// `whole` reads the block's entire span including AdaLN; otherwise
        /// only the 770.7 MB tail, as a render with a precomputed modulation
        /// schedule would. **Compute is identical in both** — the block always
        /// runs the shared resident AdaLN — so the arms differ only in bytes
        /// moved, which is the variable under test.
        func streamedPass(blocks: [BlockSpan], whole: Bool) -> Double {
            var h = f.x
            var pending: DispatchSemaphore? = nil
            let t0 = Date()

            func prefetch(_ slot: Int, _ span: BlockSpan) -> DispatchSemaphore {
                let done = DispatchSemaphore(value: 0)
                let into = staging[slot]                 // Staging is Sendable; its pointer is not
                let offset = whole ? span.start : span.tailStart
                let count = whole ? span.wholeBytes : span.tailBytes
                queue.async {
                    reader.read(into: into.pointer, offset: offset, count: count)
                    done.signal()
                }
                return done
            }

            pending = prefetch(0, blocks[0])
            for i in 0 ..< blocks.count {
                let w0 = Date()
                pending!.wait()                          // block i's bytes must be here
                waited += Date().timeIntervalSince(w0)

                // The tail always begins at `tailStart`, so a whole-span read
                // puts it `tailStart - start` bytes in.
                let base = staging[i % 2].pointer
                    .advanced(by: whole ? blocks[i].tailStart - blocks[i].start : 0)
                let live = Self.block(from: base, span: blocks[i],
                                      adaln: f.adaln, config: config)
                // Issue block i+1's read before computing block i, so the drive
                // works while the GPU does. The staging buffer for i is free the
                // moment `block(from:)` returns — `mlx_array_new_data` copies.
                if i + 1 < blocks.count {
                    pending = prefetch((i + 1) % 2, blocks[i + 1])
                }
                h = live(h, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
                MLX.eval(h)                              // keep the pipeline honest: one block deep
            }
            return Date().timeIntervalSince(t0)
        }

        _ = streamedPass(blocks: Array(spans[8 ..< 8 + k]), whole: false)   // warm the path
        waited = 0
        let streamed = streamedPass(blocks: Array(spans[16 ..< 16 + k]), whole: false)
        let tailWaited = waited
        waited = 0
        let streamedWhole = streamedPass(blocks: Array(spans[24 ..< 24 + k]), whole: true)
        let wholeWaited = waited
        waited = tailWaited

        let perBlockResident = resident / Double(k)
        print(String(format: "\n  %d blocks at the control shape, %.3f s/block resident\n",
                     k, perBlockResident))
        for (label, bytes, total, stalled) in [
            ("tail only  (AdaLN precomputed)", tailBytes, streamed, tailWaited),
            ("whole span (AdaLN streamed)   ", wholeBytes, streamedWhole, wholeWaited),
        ] {
            print(String(format: "  %@  %6.1f MB/block  %.2f s  (%.3f s/block)  "
                         + "%+5.1f%%   stalled %.2f s of %.2f s of reads",
                         label, Double(bytes) / 1e6, total, total / Double(k),
                         100 * (total - resident) / resident, stalled,
                         Double(bytes) * Double(k) / 1e9 / 3.17))
        }
        print(String(format: "\n  extrapolated to 50 blocks: resident %.1f s, "
                     + "tail-streamed %.1f s, whole-streamed %.1f s",
                     perBlockResident * 50, streamed / Double(k) * 50,
                     streamedWhole / Double(k) * 50))
    }

    // MARK: - 4. Does the footprint actually stay at two blocks?

    /// Overlap without residency is worthless — the whole point is the memory,
    /// and MLX's allocator caches freed buffers by default, so "I dropped the
    /// reference" is not evidence that the machine got the memory back.
    ///
    /// Streams sixteen blocks and reports the footprint at each. The claim
    /// under test is that it is **flat**: if MLX retains released block weights
    /// the curve rises by 770.7 MB a step and the design does not work, whatever
    /// the timings said.
    @Test(.enabled(if: Self.enabled))
    func blockStreamingResidency() throws {
        let url = Self.checkpoint!
        let config = H3Config()
        let archive = try Safetensors.Archive(url: url)
        let spans = Self.spans(archive, blobStart: try Self.blobStart(of: url),
                               count: config.numLayers)
        let f = Self.inputs(config)
        let k = 16
        let reader = try Self.UncachedReader(url)
        let tailBytes = spans[0].tailBytes

        // Two blocks of headroom and nothing more. The default limit is sized
        // from physical memory, which on a 256 GB machine would happily cache
        // every block we release and report a flat curve for the wrong reason.
        Memory.clearCache()
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 2 * tailBytes
        defer { Memory.cacheLimit = previousLimit }

        let staging = [Self.Staging(tailBytes), Self.Staging(tailBytes)]
        let queue = DispatchQueue(label: "h3.block-prefetch", qos: .userInitiated)
        let baseline = Memory.activeMemory
        var footprints: [Int] = []

        var h = f.x
        var pending: DispatchSemaphore? = nil
        func prefetch(_ slot: Int, _ span: BlockSpan) -> DispatchSemaphore {
            let done = DispatchSemaphore(value: 0)
            let into = staging[slot]
            queue.async {
                reader.read(into: into.pointer, offset: span.tailStart, count: span.tailBytes)
                done.signal()
            }
            return done
        }

        pending = prefetch(0, spans[0])
        for i in 0 ..< k {
            pending!.wait()
            // Scoped so the block's weights are released before the footprint
            // is sampled — that is the behaviour being measured, not a trick to
            // flatter it.
            do {
                let live = Self.block(from: staging[i % 2].pointer, span: spans[i],
                                      adaln: f.adaln, config: config)
                if i + 1 < k { pending = prefetch((i + 1) % 2, spans[i + 1]) }
                h = live(h, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
                MLX.eval(h)
            }
            footprints.append(Memory.activeMemory - baseline)
        }

        let mb = { (b: Int) in String(format: "%.0f", Double(b) / 1e6) }
        print("\n  footprint above baseline, MB, per streamed block:")
        print("    " + footprints.map(mb).joined(separator: " "))

        // Growth is measured across the second half against the first: the
        // early blocks include one-off allocation the steady state does not.
        let firstHalf = footprints[2 ..< k / 2].reduce(0, +) / (k / 2 - 2)
        let secondHalf = footprints[(k / 2) ..< k].reduce(0, +) / (k / 2)
        let growth = Double(secondHalf - firstHalf) / Double(tailBytes)
        print("  steady state \(mb(firstHalf)) MB -> \(mb(secondHalf)) MB over \(k / 2) blocks"
              + String(format: " = %.2f blocks' worth of growth", growth))
        print(String(format: "  50 resident blocks would be %.1f GB; "
                     + "two-block streaming holds %.1f GB",
                     Double(spans[0].wholeBytes) * 50 / 1e9,
                     Double(secondHalf) / 1e9))

        // Half a block of drift is allocator noise; a block per block is the
        // failure this test exists to catch.
        #expect(growth < 0.5, "streamed block weights are being retained")
    }
}
