// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
@testable import H3Modules

/// The CFG overlap schedule splits a block into QKV / attention / out+MLP.
/// Those three stages must compose to the original block, or overlapping them
/// would be a different model.
@Suite("CFG overlap stages")
struct CFGOverlapTests {

    static let heads = 2
    static let headDim = 4
    static let hidden = 8
    static let tokens = 16
    static let ffn = 16
    static let timeDim = 32

    static func ramp(_ shape: [Int], scale: Float = 0.01) -> MLXArray {
        let n = shape.reduce(1, *)
        let v = (0 ..< n).map { Float($0 % 17) * scale - 0.08 }
        return MLXArray(v, shape).asType(.bfloat16)
    }

    static func block() -> (DiTBlock, MLXArray, MLXArray, ModulationIndex, MLXArray?) {
        let h = hidden
        let block = DiTBlock(
            norm1: H3RMSNorm(weight: ramp([h], scale: 0.5), eps: 1e-6),
            norm2: H3RMSNorm(weight: ramp([h], scale: 0.4), eps: 1e-6),
            attn: AttentionLayer(qkvWeight: ramp([3 * h, h], scale: 0.013),
                                 outWeight: ramp([h, h], scale: 0.021),
                                 qNormWeight: MLXArray.ones([headDim]),
                                 kNormWeight: MLXArray.ones([headDim]),
                                 heads: heads, headDim: headDim, eps: 1e-6),
            mlp: H3MLP(fc1: ramp([2 * ffn, h], scale: 0.02),
                       fc2: ramp([h, ffn], scale: 0.02)),
            adaln: AdalnProj(weight: ramp([6 * h * 3, timeDim], scale: 0.01),
                             bias: nil, expand: 6, modalities: 3, hidden: h,
                             computeFP32: false))
        let x = ramp([tokens, h], scale: 0.03)
        let tEmb = ramp([1, timeDim], scale: 0.05)
        let index = ModulationIndex(segments: [ModSegment(start: 0, stop: tokens, row: 0)],
                                    tokenCount: tokens)
        return (block, x, tEmb, index, nil as MLXArray?)
    }

    @Test("the three stages compose to the original block")
    func stagesMatchCallAsFunction() {
        let (block, x, tEmb, index, rope) = Self.block()
        let fused = block(x, tEmb: tEmb, index: index, ropeTable: rope)
        let prep = block.prepareAttention(x, tEmb: tEmb, index: index, ropeTable: rope)
        let staged = block.postAttention(prep, merged: block.attend(prep, context: nil),
                                         index: index)
        MLX.eval(fused, staged)
        let maxDiff = MLX.abs(fused - staged).max().item(Float.self)
        #expect(maxDiff == 0)
    }

    @Test("the scheduled block matches running both branches in series")
    func scheduledBlockMatchesSerial() {
        let (block, x, tEmb, index, rope) = Self.block()
        let y1 = block(x, tEmb: tEmb, index: index, ropeTable: rope)
        let y2 = block(x * 1.1, tEmb: tEmb, index: index, ropeTable: rope)
        let (c, u) = CFGOverlap.block(block, cond: x, uncond: x * 1.1,
                                      tEmbC: tEmb, tEmbU: tEmb,
                                      indexC: index, indexU: index,
                                      ropeC: rope, ropeU: rope,
                                      contextC: nil, contextU: nil)
        MLX.eval(y1, y2, c, u)
        #expect(MLX.abs(c - y1).max().item(Float.self) == 0)
        #expect(MLX.abs(u - y2).max().item(Float.self) == 0)
    }

    /// The pipelined stack, which is what production runs — and the thing the
    /// earlier version of this suite never actually exercised. It drove the
    /// stages by hand, in series, and asserted that a schedule it had not run
    /// produced the right answer.
    @Test("the pipelined stack matches running both branches in series")
    func pipelineMatchesSerial() {
        let depth = 3
        var blocks: [DiTBlock] = []
        for _ in 0 ..< depth { blocks.append(Self.block().0) }
        let (_, x, tEmb, index, rope) = Self.block()
        let xU = x * 1.1

        var c = x, u = xU
        for b in blocks {
            c = b(c, tEmb: tEmb, index: index, ropeTable: rope)
            u = b(u, tEmb: tEmb, index: index, ropeTable: rope)
        }

        var cond = CFGOverlap.Branch(h: x, tEmb: tEmb, table: rope, index: index,
                                     start: nil, prep: nil, merged: nil)
        var uncond = CFGOverlap.Branch(h: xU, tEmb: tEmb, table: rope, index: index,
                                       start: nil, prep: nil, merged: nil)
        var tapped: [Int: MLXArray] = [:]
        CFGOverlap.pipeline(blocks, range: 0 ..< depth, cond: &cond, uncond: &uncond,
                            contextC: { _ in nil }, contextU: { _ in nil },
                            tap: { i, h in tapped[i] = h })

        MLX.eval(c, u, cond.h, uncond.h)
        #expect(MLX.abs(cond.h - c).max().item(Float.self) == 0)
        #expect(MLX.abs(uncond.h - u).max().item(Float.self) == 0)
        // Every conditional block must be offered to the tap, in order, or the
        // contract's `block_NN` taps go missing exactly under the fast path.
        #expect(tapped.keys.sorted() == Array(0 ..< depth))
        #expect(MLX.abs(tapped[depth - 1]! - c).max().item(Float.self) == 0)
    }

    @Test("driving the stages by hand matches too")
    func overlapMatchesSerial() {
        let (block, x, tEmb, index, rope) = Self.block()
        let y1 = block(x, tEmb: tEmb, index: index, ropeTable: rope)
        let y2 = block(x * 1.1, tEmb: tEmb, index: index, ropeTable: rope)
        MLX.eval(y1, y2)

        // The transformer helper is on H3Transformer; drive the same schedule
        // by hand so this test does not need a 66 GB model.
        let cPrep = block.prepareAttention(x, tEmb: tEmb, index: index, ropeTable: rope)
        let uPrep = block.prepareAttention(x * 1.1, tEmb: tEmb, index: index, ropeTable: rope)
        let cMerged = block.attend(cPrep, context: nil)
        let uMerged = block.attend(uPrep, context: nil)
        let cOut = block.postAttention(cPrep, merged: cMerged, index: index)
        let uOut = block.postAttention(uPrep, merged: uMerged, index: index)
        MLX.eval(cOut, uOut)

        #expect(MLX.abs(cOut - y1).max().item(Float.self) == 0)
        #expect(MLX.abs(uOut - y2).max().item(Float.self) == 0)
    }
}
