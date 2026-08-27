// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXRandom
@testable import H3Modules

/// The tiled block must compute exactly what the untiled one computes.
///
/// Query tiling is free arithmetic only if every stage downstream of attention
/// is genuinely row-wise. The output projection, the residual gate, the second
/// norm and the MLP all are — but "is" is a claim about MLX's kernels choosing
/// the same accumulation for a 1,967-row GEMM as for a 15,731-row one, and that
/// is exactly the kind of claim that is true until it is not.
@Suite("query-tiled block")
struct QueryTilingTests {

    /// The saturation probe's whole and split bounds, against hand arithmetic.
    ///
    /// The probe is the instrument that decides whether `fc2` may be routed, so
    /// it gets checked against a case with a known answer before it is pointed
    /// at a render. With every operand equal to 1, `sum_k |a||w|` over `k` terms
    /// is exactly `k`, and over a piece of `k/8` it is exactly `k/8`.
    @Test("the saturation probe measures what it claims")
    func saturationProbeIsCorrect() throws {
        let probe = DiTBlock.saturationProbe
        guard probe.enabled else { return }        // only when H3_ANE_BOUND is set
        let k = 512, n = 256, s = 8
        probe.block = 0
        probe.splits = 8
        probe.record("unit", x: MLXArray.ones([s, k]).asType(.bfloat16),
                     weight: MLXArray.ones([n, k]).asType(.bfloat16))
        let json = try String(contentsOfFile: probe.path!, encoding: .utf8)
        #expect(json.contains("\"unit\""))
        // whole contraction: exactly k. one piece: exactly k/8.
        #expect(json.contains("\"bound\": \(Double(k))"), "whole-k bound must be exactly k")
        #expect(json.contains("\"bound\": \(Double(k / 8))"), "split bound must be exactly k/8")
    }

    /// Fusing the modulation must not move a single bit.
    ///
    /// `modScaleShift` and `modGate` are now compiled. Compilation is allowed
    /// to fuse kernels and is not allowed to reassociate the arithmetic, and
    /// the difference between those two is a different sample from the same
    /// seed. Checked against the expressions they replaced, at both dtypes a
    /// block actually uses.
    @Test("the fused modulation is bit-identical to the unfused")
    func fusedModulationMatchesUnfused() {
        for dtype in [DType.bfloat16, .float32] {
            let rows = MLXArray([0, 1, 2, 2, 1, 0, 1, 2].map { Int32($0) })
            let index = ModulationIndex(rows: rows)
            let h = (MLXRandom.normal([8, 64]) * 0.7).asType(dtype)
            let shift = (MLXRandom.normal([3, 64]) * 0.3).asType(dtype)
            let scale = (MLXRandom.normal([3, 64]) * 0.3).asType(dtype)
            let other = (MLXRandom.normal([8, 64]) * 0.5).asType(dtype)
            MLX.eval(h, shift, scale, other)

            let refScaleShift = h * (1.0 + scale[rows]) + shift[rows]
            let gotScaleShift = modScaleShift(h, shift: shift, scale: scale, index: index)
            let refGate = h + other * scale[rows]
            let gotGate = modGate(h, gate: scale, other: other, index: index)
            MLX.eval(refScaleShift, gotScaleShift, refGate, gotGate)

            #expect(MLX.all(refScaleShift .== gotScaleShift).item(Bool.self),
                    "fused modScaleShift changed the arithmetic at \(dtype)")
            #expect(MLX.all(refGate .== gotGate).item(Bool.self),
                    "fused modGate changed the arithmetic at \(dtype)")
        }
    }

    @Test("tiles cover the sequence exactly once, in order")
    func spansTile() {
        for (count, tiles) in [(16, 4), (15_731, 8), (100, 3), (7, 8), (15_731, 1)] {
            let spans = QueryTiling.spans(count: count, tiles: tiles)
            #expect(spans.first?.lowerBound == 0)
            #expect(spans.last?.upperBound == count)
            for (a, b) in zip(spans, spans.dropFirst()) {
                #expect(a.upperBound == b.lowerBound, "tiles must not overlap or skip rows")
            }
            #expect(spans.reduce(0) { $0 + $1.count } == count)
        }
    }

    /// **The pipeline must not be deeper than the pool it draws from.**
    ///
    /// `QueryTiling` begins tile `i`'s `fc1` before collecting tile `i-2`'s, so
    /// three jobs of one projection are live across that call. A slot pool of
    /// two deadlocks there: `take` waits for a slot that cannot be returned
    /// until the caller it is blocking returns. That shipped, briefly, and was
    /// found by a person watching a benchmark fail to finish.
    ///
    /// This asserts the invariant directly rather than hoping a benchmark
    /// notices, and the time limit means a regression fails in seconds instead
    /// of hanging a run.
    @Test("the slot pool is deeper than the tiling pipeline",
          .timeLimit(.minutes(1)))
    func pipelineDepthOutlivesTheSlotPool() {
        // What the schedule holds at once, read off the loop in `QueryTiling`:
        // out for tile i+1 and i, and fc1 for tiles i, i-1 and i-2.
        let deepestConsumer = 3
        #expect(ANELinearBackend.Session.slotCount(splits: 1) >= deepestConsumer
                || !QueryTiling.isEnabled,
                "query tiling needs \(deepestConsumer) slots of one projection")
        #expect(ANELinearBackend.Session.slotCount(splits: 4) >= deepestConsumer
                || !QueryTiling.isEnabled,
                "splitting the contraction must not shrink the pool below the pipeline")

        // And drive it, so the invariant is checked against the real schedule
        // rather than against a number someone wrote down.
        let (block, x, tEmb, index, rope) = CFGOverlapTests.block()
        ANELinearBackend.splitOverride = 4
        defer { ANELinearBackend.splitOverride = nil }
        let tiled = QueryTiling.block(block, x, tEmb: tEmb, index: index,
                                      ropeTable: rope, tiles: 4)
        let dense = block(x, tEmb: tEmb, index: index, ropeTable: rope)
        MLX.eval(tiled, dense)
        #expect(MLX.abs(tiled - dense).max().item(Float.self) == 0)
    }

    @Test("the tiled block matches the untiled one, bit for bit")
    func tiledMatchesDense() {
        let (block, x, tEmb, index, rope) = CFGOverlapTests.block()
        let dense = block(x, tEmb: tEmb, index: index, ropeTable: rope)
        MLX.eval(dense)
        for tiles in [2, 3, 4, 8] {
            let tiled = QueryTiling.block(block, x, tEmb: tEmb, index: index,
                                          ropeTable: rope, tiles: tiles)
            MLX.eval(tiled)
            #expect(tiled.shape == dense.shape)
            #expect(MLX.abs(tiled - dense).max().item(Float.self) == 0,
                    "T=\(tiles) changed the block's arithmetic")
        }
    }
}
