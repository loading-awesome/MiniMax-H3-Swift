// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Attention
import H3Foundation

/// Overlap GPU attention on one CFG branch with Neural Engine linears on the other.
///
/// Classifier-free guidance is two independent forwards, and run back to back
/// the engine is idle for the 37.5% of each block that is GPU attention. The
/// schedule here submits **both** branches' projections before collecting
/// either, so the engine always has the other branch's work queued while the
/// GPU attends.
///
/// **The first version of this used two threads and measured 0.987x.** It put
/// GPU attention on a second thread and engine linears on the calling one, and
/// mlx-swift serialises every `eval` in the process behind one global recursive
/// lock — so whichever thread reached that lock first blocked the other for the
/// whole duration of its work. The two halves took turns. There is exactly one
/// MLX thread here for that reason; what runs beside it is the engine's own
/// thread, which never touches MLX at all. See `ANELinearBackend.Engine`.
///
/// The order is what does the work:
///
///     begin qkv (cond)  ─┐ engine
///     begin qkv (uncond) ┘        ← queued behind cond's, runs during attention
///     attend (cond)      ─┐ GPU, asyncEval — not waited for
///     attend (uncond)    ─┘        ← banks 880 ms of GPU work
///     begin out+fc1 (cond, then uncond)   ← engine, while the GPU drains that
///
/// Nothing here changes what is computed. The stages compose to
/// ``DiTBlock.callAsFunction`` and `CFGOverlapTests` pins that bit-for-bit.
enum CFGOverlap {

    /// One block of both CFG branches, engine and GPU busy at the same time.
    ///
    /// Lives on the schedule rather than on the transformer so it can be
    /// measured and tested against a single block. A copy of it living in a
    /// benchmark would be a benchmark of the copy.
    static func block(_ block: DiTBlock,
                      cond: MLXArray, uncond: MLXArray,
                      tEmbC: MLXArray, tEmbU: MLXArray,
                      indexC: ModulationIndex, indexU: ModulationIndex,
                      ropeC: MLXArray?, ropeU: MLXArray?,
                      contextC: AttentionContext?, contextU: AttentionContext?)
        -> (MLXArray, MLXArray) {
        // Both `qkv` projections go to the engine before either is collected.
        // Collecting the first one immediately is the synchronous path, and
        // measures like it.
        let startC = block.beginAttention(cond, tEmb: tEmbC, index: indexC)
        let startU = block.beginAttention(uncond, tEmb: tEmbU, index: indexU)

        // `asyncEval`, not `eval`: hand the GPU the cond attention and walk
        // away, so the engine has that window to finish the uncond `qkv`.
        let prepC = block.finishAttention(startC, ropeTable: ropeC)
        let mergedC = block.attend(prepC, context: contextC)
        MLX.asyncEval(mergedC)

        let prepU = block.finishAttention(startU, ropeTable: ropeU)

        // **The uncond attention is deliberately not submitted yet.** Queueing
        // both attentions up front banks 880 ms of GPU work, which sounds like
        // exactly what the engine wants — but Metal runs a stream in order, so
        // the cond branch's own post-attention elementwise then sits behind the
        // *uncond* attention it does not depend on, and the engine stalls
        // waiting for numbers the GPU is not allowed to compute yet. Measured,
        // that ordering cost 2616 ms against 2193 for this one.
        //
        // So cond's post-attention work goes in first, and uncond's attention
        // is submitted once it is queued — where it covers `fc1`, the largest
        // engine job in the block.
        let postC = block.beginPost(prepC, merged: mergedC)
        let mlpC = block.beginMLP(postC, index: indexC)

        let mergedU = block.attend(prepU, context: contextU)
        MLX.asyncEval(mergedU)

        let cOut = block.finishBlock(mlpC, index: indexC)

        let postU = block.beginPost(prepU, merged: mergedU)
        let mlpU = block.beginMLP(postU, index: indexU)
        let uOut = block.finishBlock(mlpU, index: indexU)

        MLX.eval(cOut, uOut)
        return (cOut, uOut)
    }

    /// The two branches walking the stack half a block out of phase.
    ///
    /// ``block`` overlaps what it can *inside* one block, and that leaves two
    /// holes it cannot fill: the first `qkv` of a pair has no GPU work banked
    /// yet, and the last `fc1` has none left. Together those are about half the
    /// engine's work, which is why raising the engine's share stopped paying.
    ///
    /// Offsetting the branches by half a block closes both. In steady state the
    /// conditional branch is attending while the unconditional one is on the
    /// engine, and then they swap — so every engine job has GPU work already
    /// queued to hide behind, and every attention has engine work to cover it.
    ///
    /// The ordering rule that makes it work, and the one that is easy to get
    /// wrong: **a branch's own post-attention elementwise must be queued before
    /// the other branch's attention is submitted.** Metal runs a stream in
    /// order, so an `fc1` whose `h2` sits behind 441 ms of the *other* branch's
    /// attention cannot start until that attention ends, and the overlap it was
    /// supposed to provide is exactly cancelled.
    struct Branch {
        var h: MLXArray
        let tEmb: MLXArray
        let table: MLXArray?
        let index: ModulationIndex
        /// Set while this branch's `qkv` for the next block is on the engine.
        var start: DiTBlock.AttentionStart?
        /// Set while this branch's attention for the current block is on the GPU.
        var prep: DiTBlock.AttentionPrep?
        var merged: MLXArray?
    }

    /// Runs `blocks[range]` for both branches, pipelined.
    ///
    /// Returns each branch's hidden state after the last block. `tap` is called
    /// with every conditional block index and output, so the caller's tap
    /// contract does not have to know the schedule.
    static func pipeline(_ blocks: [DiTBlock], range: Range<Int>,
                         cond: inout Branch, uncond: inout Branch,
                         contextC: (Int) -> AttentionContext?,
                         contextU: (Int) -> AttentionContext?,
                         tap: (Int, MLXArray) -> Void) {
        guard !range.isEmpty else { return }

        // Prologue: put the conditional branch half a block ahead.
        cond.start = blocks[range.lowerBound].beginAttention(
            cond.h, tEmb: cond.tEmb, index: cond.index)
        var uncondBlock = range.lowerBound - 1      // the block uncond is finishing

        for i in range {
            let block = blocks[i]

            // 1. Collect cond's qkv for block i. The GPU still has uncond's
            //    attention for block i-1 banked from the previous iteration.
            let prepC = block.finishAttention(cond.start!, ropeTable: cond.table)
            cond.start = nil

            // 2. Uncond finishes block i-1 on the engine — before cond's
            //    attention is submitted, so its elementwise is not queued behind
            //    441 ms it does not depend on.
            if uncondBlock >= range.lowerBound, let prepU = uncond.prep,
               let mergedU = uncond.merged {
                let postU = blocks[uncondBlock].beginPost(prepU, merged: mergedU)
                let mlpU = blocks[uncondBlock].beginMLP(postU, index: uncond.index)
                // 3. Cond's attention goes in now, so it covers uncond's `fc1`.
                let mergedC = block.attend(prepC, context: contextC(i))
                MLX.asyncEval(mergedC)
                cond.prep = prepC
                cond.merged = mergedC
                uncond.h = blocks[uncondBlock].finishBlock(mlpU, index: uncond.index)
            } else {
                let mergedC = block.attend(prepC, context: contextC(i))
                MLX.asyncEval(mergedC)
                cond.prep = prepC
                cond.merged = mergedC
            }

            // 4. Uncond starts block i on the engine, under cond's attention.
            let startU = block.beginAttention(uncond.h, tEmb: uncond.tEmb,
                                              index: uncond.index)
            let prepU = block.finishAttention(startU, ropeTable: uncond.table)

            // 5. Cond finishes block i on the engine, with uncond's attention
            //    submitted between its elementwise and its `fc1` wait.
            let postC = block.beginPost(cond.prep!, merged: cond.merged!)
            let mlpC = block.beginMLP(postC, index: cond.index)

            let mergedU = block.attend(prepU, context: contextU(i))
            MLX.asyncEval(mergedU)
            uncond.prep = prepU
            uncond.merged = mergedU
            uncondBlock = i

            cond.h = block.finishBlock(mlpC, index: cond.index)
            tap(i, cond.h)

            // 6. Cond's next block goes on the engine under uncond's attention.
            if i + 1 < range.upperBound {
                cond.start = blocks[i + 1].beginAttention(
                    cond.h, tEmb: cond.tEmb, index: cond.index)
            }
        }

        // Epilogue: uncond is still half a block behind.
        if let prepU = uncond.prep, let mergedU = uncond.merged {
            let postU = blocks[uncondBlock].beginPost(prepU, merged: mergedU)
            let mlpU = blocks[uncondBlock].beginMLP(postU, index: uncond.index)
            uncond.h = blocks[uncondBlock].finishBlock(mlpU, index: uncond.index)
            uncond.prep = nil
            uncond.merged = nil
        }
        MLX.eval(cond.h, uncond.h)
    }

}

extension H3Transformer {

    func overlapStacks(cond: inout PackedPass, uncond: inout PackedPass,
                       condCache: H3StepCache?, uncondCache: H3StepCache?,
                       stepIndex: Int?, stepCount: Int?,
                       condTaps: inout Taps) {
        let cIn = cond.h
        let uIn = uncond.h

        func run(_ i: Int, contextC: AttentionContext?, contextU: AttentionContext?) {
            let (c, u) = CFGOverlap.block(blocks[i],
                                          cond: cond.h, uncond: uncond.h,
                                          tEmbC: cond.tEmb, tEmbU: uncond.tEmb,
                                          indexC: cond.index, indexU: uncond.index,
                                          ropeC: cond.table, ropeU: uncond.table,
                                          contextC: contextC, contextU: contextU)
            cond.h = c
            uncond.h = u
            if Self.tappedBlocks.contains(i) { condTaps.blocks[i] = c }
        }

        func rest(_ from: Int, pass: inout PackedPass, cache: H3StepCache?, hIn: MLXArray,
                  recordTaps: Bool) {
            for i in from ..< blocks.count {
                pass.h = blocks[i](pass.h, tEmb: pass.tEmb, index: pass.index,
                                   ropeTable: pass.table,
                                   context: attentionContext(block: i, stepIndex: stepIndex,
                                                             stepCount: stepCount,
                                                             layout: pass.layout))
                if recordTaps, Self.tappedBlocks.contains(i) {
                    condTaps.blocks[i] = pass.h
                }
            }
            cache?.record(totalResidual: pass.h - hIn)
        }

        // Block 0 runs dense **when a cache is probing it**, and only then —
        // the same rule as `applyBlocks`, for the same measured reason. Passing
        // nil unconditionally would make this path disagree with the sequential
        // one under a sparse backend, at `--quality faithful`, which is the
        // recipe anyone benchmarking the engine is told to use.
        let probing = (condCache != nil || uncondCache != nil)
            && stepIndex != nil && stepCount != nil
        run(0,
            contextC: probing ? nil : attentionContext(block: 0, stepIndex: stepIndex,
                                                       stepCount: stepCount,
                                                       layout: cond.layout),
            contextU: probing ? nil : attentionContext(block: 0, stepIndex: stepIndex,
                                                       stepCount: stepCount,
                                                       layout: uncond.layout))

        var skipC = false, skipU = false
        if let cache = condCache, let step = stepIndex, let total = stepCount {
            switch cache.decide(probe: cond.h - cIn, audioRange: cond.layout.audioRange,
                                videoRange: cond.layout.videoRange,
                                step: step, totalSteps: total,
                                sigma: Double(cond.plan.sigmaVideo)) {
            case .reuse(let residual):
                cond.h = cIn + residual
                skipC = true
            case .runFull:
                break
            }
        }
        if let cache = uncondCache, let step = stepIndex, let total = stepCount {
            switch cache.decide(probe: uncond.h - uIn, audioRange: uncond.layout.audioRange,
                                videoRange: uncond.layout.videoRange,
                                step: step, totalSteps: total,
                                sigma: Double(uncond.plan.sigmaVideo)) {
            case .reuse(let residual):
                uncond.h = uIn + residual
                skipU = true
            case .runFull:
                break
            }
        }

        if skipC && skipU { return }
        if skipC {
            rest(1, pass: &uncond, cache: uncondCache, hIn: uIn, recordTaps: false)
            return
        }
        if skipU {
            rest(1, pass: &cond, cache: condCache, hIn: cIn, recordTaps: true)
            return
        }

        // Blocks 1..49 run pipelined, the branches half a block out of phase.
        // Block 0 is deliberately not in here: it is the cache probe, both
        // branches' outputs are needed before either decision can be made, and
        // one block of lost overlap is 1/50th of the stack.
        var condBranch = CFGOverlap.Branch(h: cond.h, tEmb: cond.tEmb, table: cond.table,
                                           index: cond.index, start: nil, prep: nil, merged: nil)
        var uncondBranch = CFGOverlap.Branch(h: uncond.h, tEmb: uncond.tEmb,
                                             table: uncond.table, index: uncond.index,
                                             start: nil, prep: nil, merged: nil)
        CFGOverlap.pipeline(
            blocks, range: 1 ..< blocks.count,
            cond: &condBranch, uncond: &uncondBranch,
            contextC: { attentionContext(block: $0, stepIndex: stepIndex,
                                         stepCount: stepCount, layout: cond.layout) },
            contextU: { attentionContext(block: $0, stepIndex: stepIndex,
                                         stepCount: stepCount, layout: uncond.layout) },
            tap: { i, h in if Self.tappedBlocks.contains(i) { condTaps.blocks[i] = h } })
        cond.h = condBranch.h
        uncond.h = uncondBranch.h
        condCache?.record(totalResidual: cond.h - cIn)
        uncondCache?.record(totalResidual: uncond.h - uIn)
    }
}
