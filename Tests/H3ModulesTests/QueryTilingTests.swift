// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
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
