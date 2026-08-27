// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import Metal
import MLX
import H3ANEBridge
@testable import H3Modules

/// Can GPU and engine bind the same pages, and does a sequence-tile schedule
/// fit in the attention window?
///
/// Multi-GPU H3 pays an all-to-all for Ulysses. This machine's claim is that
/// IOSurface is already that all-to-all: wrap the engine's activation, let
/// MLX read and write it, never `write_prefix`. These tests check the claim
/// before any `DiTBlock` change.
///
///     swift test --filter ANESharedSurface
///     H3_BIG=1 swift test --filter tileOverlap
@Suite("ANE shared surface", .serialized, .enabled(if: h3_ane_is_available()))
struct ANESharedSurfaceTests {

    static let tolerance: Float = 2e-3

    static func halves(_ count: Int, seed: UInt64) -> [Float16] {
        var state = seed
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float16(0.02 + 0.06 * (Float((state >> 40) & 0xFFFF) / 65_535.0))
        }
    }

    static func write(_ t: OpaquePointer, _ values: [Float16], rows: Int, width: Int) -> Bool {
        values.withUnsafeBytes {
            h3_ane_tensor_write(t, $0.baseAddress!, Int32(rows), Int32(width))
        }
    }

    static func adopt(_ t: OpaquePointer, rows: Int, width: Int) -> MLXArray {
        MLXArray(rawPointer: h3_ane_tensor_ptr(t)!, [rows, width], dtype: .float16) { }
    }

    static func noCopyBase(_ a: MLXArray) -> UnsafeRawPointer? {
        a.eval()
        return a.asData(access: .noCopyIfContiguous).data.withUnsafeBytes { $0.baseAddress }
    }

    static func relRMS(_ reference: MLXArray, _ actual: MLXArray) -> Float {
        let r = reference.asType(.float32), a = actual.asType(.float32)
        let d = r - a
        let num = MLX.sqrt(MLX.sum(d * d)).item(Float.self)
        let den = MLX.sqrt(MLX.sum(r * r)).item(Float.self)
        return num / max(den, 1e-30)
    }

    static func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }

    // MARK: - Probe 1: same pages

    /// The engine's `x` is an IOSurface. Wrapping it as an `MLXArray` must
    /// leave MLX sitting on that address, which is the Ulysses all-to-all
    /// with nothing to send.
    @Test
    func wrapAliasesEnginePointer() {
        let k = 128, s = 64
        guard let x = h3_ane_tensor_create(Int32(k), Int32(s)) else {
            Issue.record("tensor create failed"); return
        }
        defer { h3_ane_tensor_free(x) }
        #expect(h3_ane_tensor_is_dense(x))

        let adopted = Self.adopt(x, rows: k, width: s)
        let mlx = Self.noCopyBase(adopted)
        let engine = UnsafeRawPointer(h3_ane_tensor_ptr(x))
        #expect(mlx == engine, "MLX adopted a copy, not the IOSurface")
    }

    /// GPU reads the engine's activation without `write_prefix` after the
    /// fill, and both processors compute the same matmul.
    @Test
    func gpuReadsEngineSurface() {
        let s = 64, k = 128, n = 32
        guard let program = h3_ane_program_create(Int32(s), Int32(k), Int32(n)),
              let xt = h3_ane_tensor_create(Int32(k), Int32(s)),
              let wt = h3_ane_tensor_create(Int32(k), Int32(n)),
              let yt = h3_ane_tensor_create(Int32(s), Int32(n))
        else {
            Issue.record("engine refused \(s)x\(k)x\(n)"); return
        }
        defer {
            h3_ane_program_free(program)
            h3_ane_tensor_free(xt); h3_ane_tensor_free(wt); h3_ane_tensor_free(yt)
        }

        #expect(Self.write(xt, Self.halves(k * s, seed: 1), rows: k, width: s))
        #expect(Self.write(wt, Self.halves(k * n, seed: 2), rows: k, width: n))

        let x = Self.adopt(xt, rows: k, width: s)
        let w = Self.adopt(wt, rows: k, width: n)
        #expect(Self.noCopyBase(x) == UnsafeRawPointer(h3_ane_tensor_ptr(xt)))
        #expect(Self.noCopyBase(w) == UnsafeRawPointer(h3_ane_tensor_ptr(wt)))

        // y = x^T @ w, the engine's contract. GPU reads the wrapped surfaces.
        let gpu = MLX.matmul(x.transposed(), w)
        #expect(h3_ane_run(program, xt, wt, yt, 1), "evaluate must succeed")
        let ane = Self.adopt(yt, rows: s, width: n)
        MLX.eval(gpu, ane)

        let rel = Self.relRMS(gpu, ane)
        #expect(rel < Self.tolerance, "GPU and engine disagree on a shared x, relRMS \(rel)")
    }

    /// Slice-update an adopted surface from a GPU array. If the pointer moves,
    /// MLX allocated a new buffer and the 0-copy write is not available through
    /// the array API — Metal blit is the remaining path.
    @Test
    func gpuWritesEngineSurface() {
        let k = 128, s = 64
        guard let xt = h3_ane_tensor_create(Int32(k), Int32(s)) else {
            Issue.record("tensor create failed"); return
        }
        defer { h3_ane_tensor_free(xt) }

        let src = MLXArray(Self.halves(k * s, seed: 7).map { Float($0) }, [k, s])
            .asType(.float16)
        MLX.eval(src)

        let dest = Self.adopt(xt, rows: k, width: s)
        dest[0 ..< k, 0 ..< s] = src
        MLX.eval(dest)

        let engine = UnsafeRawPointer(h3_ane_tensor_ptr(xt))
        let after = Self.noCopyBase(dest)
        let aliased = after == engine
        print(aliased
              ? "\n  MLX slice-update stayed on the IOSurface\n"
              : "\n  MLX slice-update allocated; pointer moved\n")

        if aliased {
            // The engine must see the GPU's bytes with no write_prefix.
            let sDim = 64, kDim = 128, n = 32
            guard let program = h3_ane_program_create(Int32(sDim), Int32(kDim), Int32(n)),
                  let wt = h3_ane_tensor_create(Int32(kDim), Int32(n)),
                  let yt = h3_ane_tensor_create(Int32(sDim), Int32(n))
            else { Issue.record("program setup failed"); return }
            defer {
                h3_ane_program_free(program)
                h3_ane_tensor_free(wt); h3_ane_tensor_free(yt)
            }
            #expect(Self.write(wt, Self.halves(kDim * n, seed: 3), rows: kDim, width: n))
            #expect(h3_ane_run(program, xt, wt, yt, 1))
            let gpu = MLX.matmul(src.transposed(), Self.adopt(wt, rows: kDim, width: n))
            let ane = Self.adopt(yt, rows: sDim, width: n)
            MLX.eval(gpu, ane)
            let rel = Self.relRMS(gpu, ane)
            #expect(rel < Self.tolerance, "engine did not see the GPU write, relRMS \(rel)")
        } else {
            #expect(Self.blit(src, into: xt, rows: k, width: s),
                    "Metal blit into the IOSurface failed; GPU cannot write the engine's pages")
            let roundTrip = Self.adopt(xt, rows: k, width: s)
            MLX.eval(roundTrip)
            let rel = Self.relRMS(src, roundTrip)
            #expect(rel < Self.tolerance, "blit did not land, relRMS \(rel)")
            print("  Metal blit into the IOSurface succeeded\n")
        }
    }

    /// GPU DMA into the IOSurface, then the engine reads it. This is the
    /// path if MLX slice-update copies: still one allocation, a blit not a
    /// host gather.
    static func blit(_ src: MLXArray, into tensor: OpaquePointer, rows: Int, width: Int) -> Bool {
        guard h3_ane_tensor_is_dense(tensor) else { return false }
        src.eval()
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        guard let srcBuf = src.asMTLBuffer(device: device, noCopy: true) else { return false }
        let dstPtr = h3_ane_tensor_ptr(tensor)!
        let nbytes = rows * width * MemoryLayout<Float16>.size
        let page = 16_384
        let length = max(((nbytes + page - 1) / page) * page, page)
        guard let dstBuf = device.makeBuffer(bytesNoCopy: dstPtr, length: length,
                                             options: [.storageModeShared],
                                             deallocator: nil)
        else { return false }
        guard let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else { return false }
        blit.copy(from: srcBuf, sourceOffset: 0, to: dstBuf, destinationOffset: 0, size: nbytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return cmd.status == .completed
    }

    // MARK: - Probe 3: live tile ping-pong

    /// Two sequence tiles, same weights. GPU GEMMs tile 0 from a wrapped
    /// surface while the engine GEMMs tile 1 on its own pages. Same math as
    /// Ulysses FFN; the question is whether the pair costs `max`, not `sum`.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func tileOverlap() throws {
        let s = 2_048, k = 5_376, n = 3_072
        guard let p0 = h3_ane_program_create(Int32(s), Int32(k), Int32(n)),
              let p1 = h3_ane_program_create(Int32(s), Int32(k), Int32(n)),
              let x0 = h3_ane_tensor_create(Int32(k), Int32(s)),
              let x1 = h3_ane_tensor_create(Int32(k), Int32(s)),
              let w0 = h3_ane_tensor_create(Int32(k), Int32(n)),
              let w1 = h3_ane_tensor_create(Int32(k), Int32(n)),
              let y0 = h3_ane_tensor_create(Int32(s), Int32(n)),
              let y1 = h3_ane_tensor_create(Int32(s), Int32(n))
        else { Issue.record("setup failed"); return }
        defer {
            h3_ane_program_free(p0); h3_ane_program_free(p1)
            for t in [x0, x1, w0, w1, y0, y1] { h3_ane_tensor_free(t) }
        }

        let xVals = Self.halves(k * s, seed: 1)
        let wVals = Self.halves(k * n, seed: 2)
        #expect(Self.write(x0, xVals, rows: k, width: s))
        #expect(Self.write(x1, xVals, rows: k, width: s))
        #expect(Self.write(w0, wVals, rows: k, width: n))
        #expect(Self.write(w1, wVals, rows: k, width: n))

        let xGPU = Self.adopt(x0, rows: k, width: s)
        let wGPU = Self.adopt(w0, rows: k, width: n)
        #expect(Self.noCopyBase(xGPU) == UnsafeRawPointer(h3_ane_tensor_ptr(x0)))

        func gpuTile() -> MLXArray { MLX.matmul(xGPU.transposed(), wGPU) }

        func time(_ body: () -> Void) -> Double {
            var v: [Double] = []
            for _ in 0 ..< 5 {
                let t = Date(); body(); v.append(Date().timeIntervalSince(t))
            }
            return Self.median(v) * 1000
        }

        MLX.eval(gpuTile())
        _ = h3_ane_run_pair(p0, x1, w0, y0, p1, x1, w1, y1)

        let gpuOnly = time { MLX.eval(gpuTile()) }
        let aneOnly = time { _ = h3_ane_run_pair(p0, x1, w0, y0, p1, x1, w1, y1) }
        let both = time {
            let out = Stream.withNewDefaultStream(device: .gpu) { () -> MLXArray in
                let v = gpuTile(); MLX.asyncEval(v); return v
            }
            _ = h3_ane_run_pair(p0, x1, w0, y0, p1, x1, w1, y1)
            MLX.eval(out)
        }

        let recovered = 100 * (gpuOnly + aneOnly - both) / min(gpuOnly, aneOnly)
        print(String(format: """

          GPU tile (wrapped IOSurface) %7.1f ms
          ANE tile                     %7.1f ms
          both together                %7.1f ms
          serial would be              %7.1f ms   ideal %7.1f ms
          overlap recovered            %6.1f%% of the smaller job

        """, gpuOnly, aneOnly, both, gpuOnly + aneOnly, max(gpuOnly, aneOnly), recovered))

        #expect(both < gpuOnly + aneOnly - 0.5 * min(gpuOnly, aneOnly),
                "tiles took turns; recovered \(recovered)%")

        let gpu = gpuTile()
        let ane = Self.adopt(y0, rows: s, width: n)
        MLX.eval(gpu, ane)
        let rel = Self.relRMS(gpu, ane)
        #expect(rel < Self.tolerance, "tile 0 GPU and tile 1 ANE die 0 disagree, relRMS \(rel)")
    }
}

/// Predicted block time for query-tile + sequence-parallel linears, from
/// isolated kernel medians already in `docs/PERF_ROADMAP.md` and the engine
/// rate in `docs/ANE_STATUS.md`. No silicon — if this is not under the
/// shipping 1047.5 ms, probe 3 cannot beat channel-split.
@Suite("ANE tile schedule sim")
struct ANETileScheduleSimTests {

    /// Isolated GPU kernel ms at production layout (`PERF_ROADMAP`).
    static let qkv = 209.2
    static let attn = 439.0
    static let out = 70.9
    static let fc1 = 279.9
    static let fc2 = 155.5

    static let gpuRate = 19.2
    static let aneRate = 7.9
    static let share = 0.286

    static func aneMs(_ gpuMs: Double) -> Double { gpuMs * (gpuRate / aneRate) }

    /// Channel-split: both processors see all S, split N. Combined GEMM is
    /// `max` of the two shards. Attention and `fc2` stay GPU.
    static func channelSplit() -> Double {
        func gemm(_ gpu: Double) -> Double {
            max(gpu * (1 - share), aneMs(gpu * share))
        }
        return gemm(qkv) + attn + gemm(out) + gemm(fc1) + fc2
    }

    /// Same 0.286 share on every linear. GPU cannot GEMM while it attends.
    /// ANE `out`+`fc1` on tile `i-1` hides in the attention window after tile 0.
    static func queryTileSameShare(tiles T: Int) -> Double {
        precondition(T >= 1)
        let qkvT = max(qkv * (1 - share), aneMs(qkv * share))
        let aneOutFc1 = aneMs((out + fc1) * share)
        let gpuTail = (out + fc1) * (1 - share) + fc2
        let window = attn - attn / Double(T)
        let aneTail = aneOutFc1 - min(aneOutFc1, window)
        return qkvT + attn + max(gpuTail, aneTail)
    }

    /// QKV stays at the serial balance (0.286). `out`+`fc1` take as much
    /// engine time as the attention window after tile 0. That is the extra
    /// work query-tiling can hide that channel-split cannot.
    static func queryTileFillWindow(tiles T: Int) -> Double {
        precondition(T >= 1)
        let qkvT = max(qkv * (1 - share), aneMs(qkv * share))
        let window = attn - attn / Double(T)
        let outFc1 = out + fc1
        let aneBudget = window * (aneRate / gpuRate)
        let aneShare = min(1.0, aneBudget / outFc1)
        let gpuTail = outFc1 * (1 - aneShare) + fc2
        return qkvT + attn + gpuTail
    }

    @Test
    func queryTileBeatsChannelSplitWhenTheWindowFits() {
        let serial = Self.qkv + Self.attn + Self.out + Self.fc1 + Self.fc2
        let shipped = Self.channelSplit()
        print("\n  GPU-only serial     \(String(format: "%7.1f", serial)) ms")
        print("  channel-split (sim) \(String(format: "%7.1f", shipped)) ms   measured shipping 1047.5")
        print("  tiles   same-share     fill-window    vs channel-split")
        for tileCount in [1, 2, 4, 8, 16] {
            let same = Self.queryTileSameShare(tiles: tileCount)
            let fill = Self.queryTileFillWindow(tiles: tileCount)
            print(String(format: "  %5d   %7.1f ms    %7.1f ms    %+.1f",
                         tileCount, same, fill, fill - shipped))
        }
        print("")
        #expect(shipped < serial, "channel-split sim should beat GPU-only")
        #expect(Self.queryTileSameShare(tiles: 8) <= shipped + 1.0)
        #expect(Self.queryTileFillWindow(tiles: 8) < shipped,
                "filling the attention window with extra engine work on out+fc1 should beat channel-split")
    }
}
