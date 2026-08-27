// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Attention
import H3Foundation

/// Attention in query tiles, so the engine has work during the 37.5% of a block
/// that is attention.
///
/// Serial, a block gives the engine nothing to do while it attends: `attn out`
/// depends on the whole attention, and `qkv` is already finished. That idle
/// window is the entire remaining gap to the 1.15x gate.
///
/// Softmax runs over the key axis independently for every query row, so
/// splitting the **queries** while keeping the whole KV is arithmetically free
/// — no running maximum, no rescaling, nothing to recombine, and measured
/// bit-identical at T = 2, 4, 8 and 16 (`TiledAttentionTests`). Everything
/// after attention in a block is row-wise too: the output projection, the
/// residual and its gate, the second norm, the MLP. So tile `i-1`'s entire
/// post-attention chain can run on the engine while the GPU attends tile `i`.
///
///     GPU     attend 0   attend 1   attend 2   attend 3
///     engine             out+fc1 0  out+fc1 1  out+fc1 2  out+fc1 3
///
/// Two things this does **not** change: `qkv` still runs whole and still races
/// the GPU at 0.286, because nothing is available to cover it; and `fc2` stays
/// on the GPU, because its interior partials breach the engine's 2^15 cliff at
/// block 49 and the failure is silent zeros.
///
/// The tiling itself is not free — attention measured 434.7 ms dense against
/// 448.3 at T=8 — so the window it opens has to be worth more than the 13.6 ms
/// it costs. That is what `H3_BIG=1 swift test --filter tiledBlock` decides.
package enum QueryTiling {

    /// How many query tiles. `H3_ANE_TILES_T` sweeps it; 0 or 1 disables tiling
    /// and takes the ordinary block.
    package static let tiles: Int = {
        guard let raw = ProcessInfo.processInfo.environment["H3_ANE_TILES_T"],
              let value = Int(raw), value >= 0 else { return 8 }
        return value
    }()

    package static var isEnabled: Bool {
        ANELinearBackend.isEnabled && tiles > 1
    }

    /// Row spans of `count` rows split into `tiles` tiles.
    ///
    /// The last tile carries the remainder rather than spreading it, so every
    /// tile but one compiles to the same engine program and the session cache
    /// holds two shapes instead of `T`.
    package static func spans(count: Int, tiles: Int) -> [Range<Int>] {
        guard tiles > 1, count > tiles else { return [0 ..< count] }
        let span = (count + tiles - 1) / tiles
        var out: [Range<Int>] = []
        var start = 0
        while start < count {
            out.append(start ..< min(start + span, count))
            start += span
        }
        return out
    }

    /// One block, attention tiled, the engine fed from the tile behind.
    ///
    /// Composes to ``DiTBlock.callAsFunction`` exactly; `QueryTilingTests` pins
    /// that bit-for-bit.
    package static func block(_ block: DiTBlock, _ x: MLXArray, tEmb: MLXArray,
                              index: ModulationIndex, ropeTable: MLXArray?,
                              context: AttentionContext? = nil,
                              tiles: Int = QueryTiling.tiles) -> MLXArray {
        let prep = block.prepareAttention(x, tEmb: tEmb, index: index, ropeTable: ropeTable)
        let rows = prep.q.dim(0)
        let spans = spans(count: rows, tiles: tiles)
        guard spans.count > 1 else {
            return block.postAttention(prep, merged: block.attend(prep, context: context),
                                       index: index)
        }

        // `q`, `k` and `v` are queued, not waited for. They must be *submitted*
        // before the tiles so the engine's `qkv` shard lands before the window
        // rather than inside it, but a blocking `eval` here also drains the GPU
        // at the one moment the schedule wants it full.
        MLX.asyncEval(prep.q, prep.k, prep.v)

        // **Every tile's attention is queued up front, on this stream.** They
        // depend only on q/k/v, so there is nothing to wait for and the GPU has
        // the block's whole 434 ms of attention banked.
        var merged: [MLXArray] = []
        merged.reserveCapacity(spans.count)
        for span in spans {
            merged.append(block.attn.attend(q: prep.q[span], k: prep.k, v: prep.v,
                                            context: context))
        }
        MLX.asyncEval(merged)

        // **The post-attention chain runs on a stream of its own, and that is
        // the whole schedule.** Metal runs a stream in order: on the attention
        // stream, tile 0's residual and norm would queue behind tiles 1..T-1's
        // attention and every projection would serialise after all of it — the
        // exact opposite of the intent. On a separate stream each tile waits
        // only for its own `merged`, through the cross-stream dependency MLX
        // inserts, so tile 0's `out` and `fc1` run on the engine while the GPU
        // is still attending tiles 1..T-1.
        //
        // One scope for the whole loop, not one per tile: `withNewDefaultStream`
        // builds a Metal command queue each time it is entered, and a block is
        // executed fifty times a step.
        // **Two engine jobs in flight at all times.** Submitting one and
        // waiting on it immediately leaves the engine idle for every upload,
        // every gather and every norm the single MLX thread has to run in
        // between — measured, that is most of the window. So each tile's `out`
        // is submitted one iteration early and each tile's `fc1` is collected
        // one iteration late, which keeps a job queued behind the one being
        // waited on without needing more than the two slots a session owns.
        let m = prep.m
        var outputs = [MLXArray?](repeating: nil, count: spans.count)
        var residual = [MLXArray?](repeating: nil, count: spans.count)
        var mlpJob = [ANELinearBackend.Pending?](repeating: nil, count: spans.count)
        var outJob = [ANELinearBackend.Pending?](repeating: nil, count: spans.count)
        var rowIndex = [ModulationIndex?](repeating: nil, count: spans.count)

        Stream.withNewDefaultStream(device: .gpu) {
            outJob[0] = block.attn.beginOut(merged[0])
            for i in spans.indices {
                if i + 1 < spans.count {
                    outJob[i + 1] = block.attn.beginOut(merged[i + 1])
                }
                let span = spans[i]
                let rows = ModulationIndex(rows: index.rows[span])
                rowIndex[i] = rows

                let out = outJob[i]!.value()
                outJob[i] = nil
                let x1 = modGate(prep.x[span], gate: m[2], other: out, index: rows)
                let h2 = modScaleShift(block.norm2(x1), shift: m[3], scale: m[4], index: rows)
                residual[i] = x1
                mlpJob[i] = block.mlp.begin(h2)

                // Collected two tiles late, not one: with a third slot the
                // engine can hold `fc1` for tile i-2, `out` for tile i+1 and
                // the job being waited on, so it never runs dry while this
                // thread is uploading or gathering.
                if i >= 2 {
                    outputs[i - 2] = modGate(residual[i - 2]!, gate: m[5],
                                             other: block.mlp.finish(mlpJob[i - 2]!),
                                             index: rowIndex[i - 2]!)
                    mlpJob[i - 2] = nil
                }
            }
            for i in max(0, spans.count - 2) ..< spans.count {
                outputs[i] = modGate(residual[i]!, gate: m[5],
                                     other: block.mlp.finish(mlpJob[i]!),
                                     index: rowIndex[i]!)
                mlpJob[i] = nil
            }
        }

        return concatenated(outputs.map { $0! }, axis: 0)
    }
}
