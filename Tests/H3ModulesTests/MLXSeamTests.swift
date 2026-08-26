// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import Metal
import MLX
import H3Foundation

/// Can work done outside MLX get back in without paying for it twice?
///
/// This is the seam the whole ANE route stands on, and `docs/PERF_ROADMAP.md`
/// already priced one half of it: a crossing costs about 28 ms, and four
/// projections a block is 98–112 ms against the 240 ms a hybrid block saves.
/// That lands near 1.12x against a 1.15x gate — close enough that where the
/// cost actually goes decides the project.
///
/// The API is asymmetric, and that asymmetry is the whole problem:
///
///  * **Out** — `asMTLBuffer(noCopy: true)` calls `eval()` and then wraps MLX's
///    own backing bytes. The `eval()` is the barrier; the wrapper is free.
///  * **In** — there is no constructor from an `MTLBuffer`. Nothing adopts a
///    buffer back. The obvious implementation copies the result home, and at
///    production QKV that is 677 MB per projection per block.
///
/// So the design worth testing is to never bring anything home: allocate the
/// output as an MLXArray first, take a no-copy Metal wrapper over *it*, and let
/// whatever runs outside MLX write into MLX's own memory. If that holds, the
/// inbound half costs nothing and only the `eval()` remains.
///
/// Two things have to be true for it to hold, and neither is documented:
///
///  1. `makeBuffer(bytesNoCopy:)` over an MLXArray's backing must alias it, so
///     a GPU write through the buffer is visible through the array.
///  2. MLX must not have cached anything about that array's contents, and must
///     not move or reuse the allocation while a reference is held.
///
/// Both hold. `metalWriteIsVisibleToMLX` stamps a constant through the wrapper
/// and reads it back through the array, so the inbound half of a crossing can
/// be made free — which matters, because the copy it replaces is 53 ms per
/// projection at production QKV.
///
///     H3_BIG=1 swift test --filter mlxSeam
@Suite("MLX seam", .serialized)
struct MLXSeamTests {

    static let s = 15_731                       // 864x480x124, the control shape
    static let k = 5_376                        // hidden
    static let n = 21_504                       // qkv inner, 3 x 7168

    /// Writes a constant into every element of a bf16 buffer, so a successful
    /// alias is unambiguous rather than a coincidence of uninitialised memory.
    private static func stamp(_ device: any MTLDevice) throws -> any MTLComputePipelineState {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void stamp(device ushort *out [[buffer(0)]],
                          constant uint &value [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) { out[i] = ushort(value); }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        return try device.makeComputePipelineState(function: library.makeFunction(name: "stamp")!)
    }

    /// Does a GPU write through a no-copy wrapper land in the MLXArray?
    ///
    /// Small on purpose. If aliasing works at all it works here, and a failure
    /// at production size would be ambiguous between "does not alias" and "did
    /// not fit".
    @Test
    func metalWriteIsVisibleToMLX() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let count = 4_096
        let array = MLXArray.zeros([count], dtype: .bfloat16)
        array.eval()

        guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
            Issue.record("asMTLBuffer(noCopy:) returned nil — no aliasing path exists")
            return
        }
        #expect(buffer.length == count * 2, "wrapper must cover the array, not a copy of it")

        // 0x3F80 is bf16 1.0. Chosen so the check cannot pass on zeroed memory.
        let pipeline = try Self.stamp(device)
        let queue = device.makeCommandQueue()!
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        var value: UInt32 = 0x3F80
        encoder.setBytes(&value, length: 4, index: 1)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

        let readBack = array.asType(.float32).asArray(Float.self)
        let ones = readBack.filter { $0 == 1.0 }.count
        #expect(ones == count,
                "GPU wrote through the wrapper but MLX sees \(ones)/\(count) — the buffer is a copy, so every result computed outside MLX has to be copied home")
    }

    /// What the two halves of a crossing actually cost at production shape.
    ///
    /// The roadmap's 28 ms is an *implied* barrier, backed out of whole-block
    /// timings for the MPSGraph arm. This measures the pieces directly, because
    /// the ANE design can only afford some of them:
    ///
    ///  * `eval()` on a pending projection — unavoidable, paid per crossing
    ///  * the no-copy wrap — expected free
    ///  * `asDataCopy` home — what writing into MLX's own buffer would avoid
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func crossingHalvesAtProductionShape() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        func time(_ rounds: Int = 5, _ body: () -> Void) -> Double {
            body()
            var samples: [Double] = []
            for _ in 0 ..< rounds {
                let t0 = Date(); body(); samples.append(Date().timeIntervalSince(t0))
            }
            return median(samples) * 1000
        }

        let x = MLXRandom.normal([Self.s, Self.k]).asType(.bfloat16)
        let w = MLXRandom.normal([Self.k, Self.n]).asType(.bfloat16)
        MLX.eval(x, w)

        // The barrier is the DIFFERENCE, not the total. A forced matmul takes
        // 190 ms whether or not anything crosses, because the arithmetic would
        // have happened anyway; what a crossing costs is the fusion and
        // scheduling MLX gives up when it has to materialise mid-graph. So run
        // the same four projections twice — draining once at the end, and
        // draining after each — and divide the gap by four.
        let projections = 4
        let lazyMS = time(3) {
            var pending: [MLXArray] = []
            for _ in 0 ..< projections { pending.append(MLX.matmul(x, w)) }
            MLX.eval(pending)
        }
        let forcedMS = time(3) {
            for _ in 0 ..< projections { MLX.eval(MLX.matmul(x, w)) }
        }
        let barrierMS = (forcedMS - lazyMS) / Double(projections)

        let y = MLX.matmul(x, w)
        y.eval()
        let wrapMS = time { _ = y.asMTLBuffer(device: device, noCopy: true) }
        let copyMS = time(3) { _ = y.asMTLBuffer(device: device, noCopy: false) }

        let bytes = Double(Self.s * Self.n * 2) / 1e6
        print("""

          production QKV crossing, S=\(Self.s) K=\(Self.k) N=\(Self.n), output \
        \(String(format: "%.0f", bytes)) MB

            4 projections, one drain   \(String(format: "%8.1f", lazyMS)) ms
            4 projections, drained each\(String(format: "%8.1f", forcedMS)) ms
            implied barrier per crossing \(String(format: "%6.1f", barrierMS)) ms

            wrap, no copy              \(String(format: "%8.3f", wrapMS)) ms
            wrap, copying home         \(String(format: "%8.1f", copyMS)) ms   <- avoided by
                                                        writing into MLX's own buffer

          The barrier here reads near zero, and that is a fact about this
          measurement rather than about a block. Four INDEPENDENT matmuls have
          nothing to fuse, so draining between them costs nothing. In a real
          block the projections sit among norms, modulation, RoPE and gating
          that MLX does fuse, and forcing a drain mid-block gives those up —
          which is why `crossingCost` measures 24.5-31 ms and this measures 0.
          Prefer that number; this one prices something nobody runs.

          What this test does establish is the other half, and it is the half
          that was unknown: the copy home costs \(String(format: "%.0f", copyMS)) ms
          per projection, four a block is \(String(format: "%.0f", copyMS * 4)) ms
          against the 240 ms a hybrid block saves, and `metalWriteIsVisibleToMLX`
          shows it is avoidable entirely.

        """)
        #expect(wrapMS < copyMS, "no-copy wrap must be cheaper than copying home")
    }
}
