// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import H3Foundation
@testable import H3Modules

/// What does the CFG pair actually cost, and where does the time sit?
///
/// The unit is a **pair of blocks** — one conditional, one unconditional —
/// because that is what a guided step repeats fifty times, and because the
/// overlap schedule has nothing to work with inside a single block. A per-block
/// number cannot express "the engine is busy while the GPU attends".
///
///     H3_BIG=1 swift test --filter cfgPair
///     H3_ANE=experimental H3_BIG=1 swift test --filter cfgPair
///
/// Out of the normal suite: production width allocates about 1.3 GB of block
/// weights plus two branches of activations.
@Suite("CFG pair at production width", .serialized)
struct CFGOverlapBenchTests {

    /// Both branches of one block, sharing the block's weights, as CFG does.
    static func pair(seed: UInt64 = 11)
        -> (block: DiTBlock, xC: MLXArray, xU: MLXArray, tEmb: MLXArray,
            index: ModulationIndex, rope: MLXArray?) {
        let f = CompiledBlockTests.productionBlock(seed: seed)
        // The unconditional branch is the same shapes against a different
        // hidden state. Scaling the conditional one keeps the magnitudes in the
        // range the saturation bound was measured on.
        return (f.block, f.x, f.x * 1.03, f.tEmb, f.index, f.rope)
    }

    private static func ms(_ label: String, _ n: Int, _ body: () -> [MLXArray]) -> Double {
        MLX.eval(body())
        var s: [Double] = []
        for _ in 0 ..< n {
            let t0 = Date()
            MLX.eval(body())
            s.append(Date().timeIntervalSince(t0))
        }
        let sorted = s.sorted()
        let med = sorted[sorted.count / 2] * 1000
        print(String(format: "  %@ %8.1f ms   (min %.1f  max %.1f)",
                     label.padding(toLength: 34, withPad: " ", startingAt: 0),
                     med, sorted.first! * 1000, sorted.last! * 1000))
        return med
    }

    /// Serial pair against overlapped pair, at production width.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func cfgPair() {
        let f = Self.pair()
        print("\n  ANE routing: \(ANELinearBackend.isEnabled ? "ON" : "off")")

        let serial = { () -> [MLXArray] in
            [f.block(f.xC, tEmb: f.tEmb, index: f.index, ropeTable: f.rope),
             f.block(f.xU, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)]
        }
        let overlapped = { () -> [MLXArray] in
            let (c, u) = CFGOverlap.block(f.block,
                                          cond: f.xC, uncond: f.xU,
                                          tEmbC: f.tEmb, tEmbU: f.tEmb,
                                          indexC: f.index, indexU: f.index,
                                          ropeC: f.rope, ropeU: f.rope,
                                          contextC: nil, contextU: nil)
            return [c, u]
        }

        // Math before speed. The schedule is allowed to change when work runs,
        // never what it computes.
        let a = serial(), b = overlapped()
        MLX.eval(a, b)
        for (i, name) in [(0, "cond"), (1, "uncond")] {
            let same = MLX.all(a[i] .== b[i]).item(Bool.self)
            #expect(same, "\(name) branch must be bit-identical under the overlap schedule")
        }

        let m = BenchmarkSupport.interleavedArrays(rounds: 5, first: serial, second: overlapped)
        let serialMs = m.first * 1000, overlapMs = m.second * 1000
        print(String(format: "\n  serial pair     %8.1f ms", serialMs))
        print(String(format: "  overlapped pair %8.1f ms", overlapMs))
        print(String(format: "  overlap gain    %8.3fx\n", serialMs / overlapMs))
    }

    /// The pipelined stack against the same blocks run back to back.
    ///
    /// A single block cannot show this. Half the engine's work in a block-pair
    /// sits at the two ends — the first `qkv`, which has no GPU work banked
    /// yet, and the last `fc1`, which has none left — and only offsetting the
    /// branches across a block boundary fills them. So the unit here is a
    /// stack, and four blocks is enough for the steady state to dominate.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func cfgStack() {
        let depth = 4
        let f = Self.pair()
        // Distinct weights per block, as a real stack has: one block reused
        // would let the engine's weight cache answer every call from the first
        // upload and measure a render nobody runs.
        var blocks = [f.block]
        for seed in 1 ..< depth {
            blocks.append(CompiledBlockTests.productionBlock(seed: UInt64(11 + seed)).block)
        }
        print("\n  ANE routing: \(ANELinearBackend.isEnabled ? "ON" : "off")"
              + "   engine share \(ANELinearBackend.share)   depth \(depth)")

        let serial = { () -> [MLXArray] in
            var c = f.xC, u = f.xU
            for b in blocks {
                c = b(c, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
                u = b(u, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
            }
            return [c, u]
        }
        let pipelined = { () -> [MLXArray] in
            var cond = CFGOverlap.Branch(h: f.xC, tEmb: f.tEmb, table: f.rope,
                                         index: f.index, start: nil, prep: nil, merged: nil)
            var uncond = CFGOverlap.Branch(h: f.xU, tEmb: f.tEmb, table: f.rope,
                                           index: f.index, start: nil, prep: nil, merged: nil)
            CFGOverlap.pipeline(blocks, range: 0 ..< depth, cond: &cond, uncond: &uncond,
                                contextC: { _ in nil }, contextU: { _ in nil }, tap: { _, _ in })
            return [cond.h, uncond.h]
        }

        let a = serial(), b = pipelined()
        MLX.eval(a, b)
        for (i, name) in [(0, "cond"), (1, "uncond")] {
            #expect(MLX.all(a[i] .== b[i]).item(Bool.self),
                    "\(name) branch must be bit-identical under the pipeline")
        }

        let m = BenchmarkSupport.interleavedArrays(rounds: 3, first: serial, second: pipelined)
        let serialMs = m.first * 1000, pipeMs = m.second * 1000
        print(String(format: "  serial     %8.1f ms  (%.1f a pair)", serialMs, serialMs / Double(depth)))
        print(String(format: "  pipelined  %8.1f ms  (%.1f a pair)", pipeMs, pipeMs / Double(depth)))
        print(String(format: "  gain       %8.3fx\n", serialMs / pipeMs))
    }

    /// Where a block's time actually sits, per stage, at production width.
    ///
    /// The overlap schedule can only hide engine work behind attention, so what
    /// bounds it is the ratio of the attention stage to everything else. That
    /// ratio has been quoted at 37% from a FLOP estimate; this measures it.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_BIG"] != nil))
    func stageBreakdown() {
        let f = Self.pair()
        print("\n  ANE routing: \(ANELinearBackend.isEnabled ? "ON" : "off")")

        let prep = f.block.prepareAttention(f.xC, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
        MLX.eval(prep.q, prep.k, prep.v)
        let merged = f.block.attend(prep, context: nil)
        MLX.eval(merged)

        let whole = Self.ms("whole block", 7) {
            [f.block(f.xC, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)]
        }
        let pMs = Self.ms("prepare (adaln+norm+qkv+rope)", 7) {
            let p = f.block.prepareAttention(f.xC, tEmb: f.tEmb, index: f.index, ropeTable: f.rope)
            return [p.q, p.k, p.v]
        }
        let aMs = Self.ms("attend (sdpa)", 7) { [f.block.attend(prep, context: nil)] }
        let oMs = Self.ms("post (out proj + mlp)", 7) {
            [f.block.postAttention(prep, merged: merged, index: f.index)]
        }
        print(String(format: "\n  stages sum %.1f ms against %.1f ms whole", pMs + aMs + oMs, whole))
        print(String(format: "  attention is %.1f%% of the block — the only window the engine has\n",
                     100 * aMs / whole))
    }
}
