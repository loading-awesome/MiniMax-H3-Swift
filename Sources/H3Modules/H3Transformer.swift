// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXNN
import H3Foundation
import H3Attention

/// Packing helpers. These decide the ROW ORDER of the packed sequence, so an
/// error here misaligns every downstream tap while keeping all the shapes
/// correct — the most expensive kind of bug to find late.
package enum H3Packing {
    /// `[B,C,T,H,W] -> [B*t*h*w, C*pt*ph*pw]`, rows ordered t-major then h then w.
    ///
    /// Reference: `einsum("nctrhpwq->nthwcrpq")`. Feature order within a row is
    /// `(c, pt, ph, pw)` — channel outermost, patch offsets innermost.
    package static func patchifyVideo(_ latent: MLXArray, patch: [Int] = [1, 2, 2]) -> MLXArray {
        let b = latent.dim(0), c = latent.dim(1)
        let (pt, ph, pw) = (patch[0], patch[1], patch[2])
        let t = latent.dim(2) / pt, h = latent.dim(3) / ph, w = latent.dim(4) / pw
        // n c (t pt) (h ph) (w pw) -> n t h w c pt ph pw
        return latent.reshaped([b, c, t, pt, h, ph, w, pw])
                     .transposed(0, 2, 4, 6, 1, 3, 5, 7)
                     .reshaped([b * t * h * w, c * pt * ph * pw])
    }

    package static func unpatchifyVideo(_ rows: MLXArray, t: Int, h: Int, w: Int,
                                       channels: Int = 24, patch: [Int] = [1, 2, 2]) -> MLXArray {
        let (pt, ph, pw) = (patch[0], patch[1], patch[2])
        return rows.reshaped([-1, t, h, w, channels, pt, ph, pw])
                   .transposed(0, 4, 1, 5, 2, 6, 3, 7)
                   .reshaped([-1, channels, t * pt, h * ph, w * pw])
    }

    /// `[B,32,2,T] -> [2*T, 32]`, **channel-major**: ch0 t0..T-1 then ch1 t0..T-1.
    /// Reference: `latent[0].permute(1,2,0).reshape(ch*t, c)`.
    package static func packAudio(_ latent: MLXArray) -> MLXArray {
        let c = latent.dim(1), ch = latent.dim(2), t = latent.dim(3)
        return latent[0].transposed(1, 2, 0).reshaped([ch * t, c])
    }

    package static func unpackAudio(_ rows: MLXArray, channels: Int = 2) -> MLXArray {
        let t = rows.dim(0) / channels
        return rows.reshaped([channels, t, rows.dim(-1)])
                   .transposed(2, 0, 1)
                   .expandedDimensions(axis: 0)
    }
}

/// RoPE frequency construction.
///
///     per_axis = pos[:, :, None] * inv_freq        [S, 3, 16]
///     half     = cat(t, h, w)                      [S, 48]
///     angles   = cat(half, half)                   [S, 96]
///
/// The duplicated halves are why `rope_rotation_table` only reads the first
/// half — and why `rot` is 96 while `headDim` is 128, leaving 32 channels
/// unrotated.
package enum H3RoPE {
    /// `positionIds` is `[S, 3]` (t, h, w). Returns `[S, 96]` angles.
    package static func angles(positionIds: MLXArray, invFreq: MLXArray) -> MLXArray {
        let pos = positionIds.asType(.float32)
        let inv = invFreq.asType(.float32).reshaped([1, 1, -1])
        let perAxis = pos.expandedDimensions(axis: -1) * inv          // [S,3,16]
        let s = perAxis.dim(0)
        let half = perAxis.reshaped([s, -1])                          // [S,48] = t|h|w
        return concatenated([half, half], axis: -1)                   // [S,96]
    }

    /// `[S, rot] angles -> [1, S, 1, rot/2, 2, 2]` holding `[[c, -s], [s, c]]`.
    package static func rotationTable(angles: MLXArray) -> MLXArray {
        let s = angles.dim(0)
        let half = angles.dim(-1) / 2
        let ang = angles[0..., 0 ..< half]
        let c = cos(ang), sn = sin(ang)
        return stacked([c, -sn, sn, c], axis: -1).reshaped([1, s, 1, half, 2, 2])
    }
}

/// Sinusoidal-style timestep embedding: `proj_out(silu(proj_in(t)))`.
/// Only used when the checkpoint has no `adaln_t_table`; ours does not, which
/// the inventory confirms by deriving `timestepInputDim` from `proj_in`.
package struct TimeEmbedder {
    package let projInWeight: MLXArray
    package let projInBias: MLXArray?
    package let projOutWeight: MLXArray
    package let projOutBias: MLXArray?
    package let inputDim: Int

    package init(projInWeight: MLXArray, projInBias: MLXArray?,
                projOutWeight: MLXArray, projOutBias: MLXArray?, inputDim: Int) {
        self.projInWeight = projInWeight
        self.projInBias = projInBias
        self.projOutWeight = projOutWeight
        self.projOutBias = projOutBias
        self.inputDim = inputDim
    }

    /// `t` is `[M]` timestep values in [0, 1].
    package func callAsFunction(_ t: MLXArray) -> MLXArray {
        var h = matmul(sinusoid(t), projInWeight.T)
        if let projInBias { h = h + projInBias }
        h = silu(h)
        var o = matmul(h, projOutWeight.T)
        if let projOutBias { o = o + projOutBias }
        return o
    }

    /// Standard half-cos/half-sin frequency embedding of width `inputDim`,
    /// **cos before sin**.
    ///
    /// The association is the reference's, not the algebraically tidier one:
    /// `exp(-log(10000) * i / half)` multiplies before dividing. Folding the
    /// constant first — `i * (-log(10000)/half)` — is the same number in real
    /// arithmetic and a different one in fp32, and it showed up as a 2.5e-06
    /// discrepancy on a tap that is otherwise bit-exact.
    func sinusoid(_ t: MLXArray) -> MLXArray {
        let half = inputDim / 2
        let scale = Float(-Foundation.log(10000.0))
        let freqs = exp(MLXArray(0 ..< half).asType(.float32) * scale / Float(half))
        let a = t.asType(.float32).expandedDimensions(axis: -1) * freqs.reshaped([1, -1])
        return concatenated([cos(a), sin(a)], axis: -1)
    }
}

/// Final layer: `video_out` / `audio_out` heads over the target segments.
///
/// The heads are the checkpoint's **fp32 island**. Casting them to bf16 with
/// everything else silently changes the output.
///
/// Its AdaLN has `modalities = 1`, unlike a block's 3 — so `ModSegment.row`
/// here is the **timestep row alone**, not `timestepRow * 3 + tag`.
package struct FinalLayer {
    package let norm: H3RMSNorm
    package let adaln: AdalnProj
    package let videoOutWeight: MLXArray
    package let videoOutBias: MLXArray?
    package let audioOutWeight: MLXArray
    package let audioOutBias: MLXArray?

    package init(norm: H3RMSNorm, adaln: AdalnProj,
                videoOutWeight: MLXArray, videoOutBias: MLXArray?,
                audioOutWeight: MLXArray, audioOutBias: MLXArray?) {
        self.norm = norm
        self.adaln = adaln
        self.videoOutWeight = videoOutWeight
        self.videoOutBias = videoOutBias
        self.audioOutWeight = audioOutWeight
        self.audioOutBias = audioOutBias
    }

    /// `seg` entries are `(start, stop, modRow)`.
    package func callAsFunction(_ h: MLXArray, tEmb: MLXArray,
                               videoSeg: ModSegment, audioSeg: ModSegment)
        -> (video: MLXArray, audio: MLXArray) {
        let m = adaln(tEmb)
        precondition(m.count == 2, "FinalLayer AdaLN must expand to 2, got \(m.count)")
        let shift = m[0], scale = m[1]

        func head(_ seg: ModSegment, _ w: MLXArray, _ b: MLXArray?) -> MLXArray {
            let slice = h.ndim == 3 ? h[0..., seg.start ..< seg.stop] : h[seg.start ..< seg.stop]
            let sc = scale[seg.row].expandedDimensions(axis: 0)
            let sh = shift[seg.row].expandedDimensions(axis: 0)
            let x = (norm(slice) * (1.0 + sc) + sh).asType(.float32)
            var o = matmul(x, w.asType(.float32).T)
            if let b { o = o + b.asType(.float32) }
            return o
        }
        return (head(videoSeg, videoOutWeight, videoOutBias),
                head(audioSeg, audioOutWeight, audioOutBias))
    }
}

/// The assembled DiT.
///
/// Covers the span the parity contract makes testable without a text encoder:
/// conditioning in, sampled-latent-shaped velocity out. Both VAEs are separate
/// models and are not part of this type.
package struct H3Transformer {
    /// Immutable tensors shared by every denoise step of one render.
    ///
    /// The conditioning projection/refiner and RoPE table depend on the prompt
    /// and packed geometry, not on sigma or the evolving latents. Keeping them
    /// alive avoids rebuilding the position table and re-reading the refiner's
    /// weights on every step. It is deliberately supplied by the caller rather
    /// than kept as mutable state on the model: one model may serve multiple
    /// prompts or geometries without a stale-cache hazard.
    package final class RenderState: @unchecked Sendable {
        fileprivate let layout: PackedLayout
        fileprivate let contextTokens: Int
        fileprivate let textStates: MLXArray
        fileprivate let refined: MLXArray
        fileprivate let ropeTable: MLXArray

        fileprivate init(layout: PackedLayout, contextTokens: Int,
                         textStates: MLXArray, refined: MLXArray, ropeTable: MLXArray) {
            self.layout = layout
            self.contextTokens = contextTokens
            self.textStates = textStates
            self.refined = refined
            self.ropeTable = ropeTable
        }
    }

    package let config: H3Config
    package let conditionProj: (weight: MLXArray, bias: MLXArray?)
    package let videoPatchProj: (weight: MLXArray, bias: MLXArray?)
    package let audioPatchProj: (weight: MLXArray, bias: MLXArray?)
    package let tokenRefiner: TokenRefiner
    package let timeEmbedder: TimeEmbedder
    package let blocks: [DiTBlock]
    package let finalLayer: FinalLayer
    package let ropeInvFreq: MLXArray
    /// The dtype the block stack runs in. The reference takes it from the
    /// incoming context, which is bf16 in every real run.
    package let computeDType: DType

    package init(config: H3Config,
                conditionProj: (weight: MLXArray, bias: MLXArray?),
                videoPatchProj: (weight: MLXArray, bias: MLXArray?),
                audioPatchProj: (weight: MLXArray, bias: MLXArray?),
                tokenRefiner: TokenRefiner, timeEmbedder: TimeEmbedder,
                blocks: [DiTBlock], finalLayer: FinalLayer, ropeInvFreq: MLXArray,
                computeDType: DType = .bfloat16) {
        self.config = config
        self.conditionProj = conditionProj
        self.videoPatchProj = videoPatchProj
        self.audioPatchProj = audioPatchProj
        self.tokenRefiner = tokenRefiner
        self.timeEmbedder = timeEmbedder
        self.blocks = blocks
        self.finalLayer = finalLayer
        self.ropeInvFreq = ropeInvFreq
        self.computeDType = computeDType
    }

    /// Taps recorded while running, keyed by the parity contract's names.
    package struct Taps {
        package var conditionProj: MLXArray?
        package var tokenRefiner: MLXArray?
        package var videoPatchProj: MLXArray?
        package var audioPatchProj: MLXArray?
        package var timeEmbedder: MLXArray?
        package var blocks: [Int: MLXArray] = [:]
        package var finalVideo: MLXArray?
        package var finalAudio: MLXArray?
        package init() {}
    }

    private func linear(_ x: MLXArray, _ p: (weight: MLXArray, bias: MLXArray?)) -> MLXArray {
        var o = matmul(x, p.weight.T)
        if let b = p.bias { o = o + b }
        return o
    }

    /// Which block outputs get recorded. Matches the contract's `block_NN` taps.
    package static let tappedBlocks: Set<Int> = [0, 1, 24, 49]

    /// Packed hidden state and the per-pass tables the block stack consumes.
    ///
    /// Split out of ``packedForward`` so classifier-free guidance can run two
    /// packs and then overlap their block stacks, rather than two full forwards
    /// back to back.
    struct PackedPass {
        var h: MLXArray
        let tEmb: MLXArray
        let table: MLXArray
        let layout: PackedLayout
        let plan: TimestepPlan
        let index: ModulationIndex
    }

    func attentionContext(block: Int, stepIndex: Int?, stepCount: Int?,
                          layout: PackedLayout) -> AttentionContext? {
        guard let step = stepIndex, let total = stepCount, total > 0 else { return nil }
        return AttentionContext(blockIndex: block, blockCount: blocks.count,
                                scheduleProgress: Double(step) / Double(total),
                                sequenceLength: layout.totalTokens,
                                videoSpan: layout.videoRange)
    }

    /// Precompute the exact prompt- and geometry-invariant DiT inputs for one
    /// render. Call once before the sampler loop and pass the result to every
    /// ``velocity`` / ``guidedVelocity`` invocation in that loop.
    package func prepareRender(context: MLXArray, geometry: LatentGeometry,
                              keyframes: [KeyframeConfig] = [],
                              refs: [ReferenceBlock] = []) throws -> RenderState {
        let layout = try PackedLayout(textTokens: context.dim(1), geometry: geometry,
                                      keyframes: keyframes, refs: refs)
        let textStates = linear(context[0].asType(computeDType), conditionProj)
        let refined = tokenRefiner(textStates)
        let pos = MLXArray(layout.positionIds.map { Float($0) }, [layout.totalTokens, 3])
        let rope = H3RoPE.rotationTable(
            angles: H3RoPE.angles(positionIds: pos, invFreq: ropeInvFreq)
        ).asType(computeDType)
        return RenderState(layout: layout, contextTokens: context.dim(1), textStates: textStates,
                           refined: refined, ropeTable: rope)
    }

    func applyBlocks(to h: inout MLXArray, tEmb: MLXArray, index: ModulationIndex,
                     table: MLXArray, layout: PackedLayout, plan: TimestepPlan,
                     tapsOut: inout Taps,
                     stepCache: H3StepCache?,
                     stepIndex: Int?, stepCount: Int?) {
        // Cross-step residual reuse. Block 0 always runs and doubles as the
        // probe; if its residual barely moved since the previous step, the
        // other 49 blocks are skipped and last step's total residual is
        // re-applied. See `H3StepCache` for why the *total residual* is what
        // gets cached rather than the output.
        //
        // With no cache this is the plain loop, unchanged and bit-identical.
        if let cache = stepCache, let step = stepIndex, let total = stepCount {
            let hIn = h
            // **Block 0 runs dense whenever the cache is on, and this is not a
            // conservatism — it is measured.** Block 0's residual is the cache's
            // probe. Sol-Attn's error at block 0 is rel_rms 0.132 at beta 1.2,
            // and the probe signal the cache thresholds on is 0.077: the
            // approximation is 1.7x the quantity being measured. Sparsify here
            // and the cache stops thresholding on how much the step moved and
            // starts thresholding on backend noise — reusing when it should not
            // and refusing when it should, with every shape still correct.
            //
            // The cost is 1/50th of the stack. Passing nil, rather than asking
            // the backend to exclude block 0 itself, keeps the rule where its
            // reason lives instead of in each backend's policy.
            h = blocks[0](h, tEmb: tEmb, index: index, ropeTable: table, context: nil)
            if Self.tappedBlocks.contains(0) { tapsOut.blocks[0] = h }

            switch cache.decide(probe: h - hIn, audioRange: layout.audioRange,
                                videoRange: layout.videoRange,
                                step: step, totalSteps: total,
                                sigma: Double(plan.sigmaVideo)) {
            case .reuse(let residual):
                h = hIn + residual
            case .runFull:
                for i in 1 ..< blocks.count {
                    h = blocks[i](h, tEmb: tEmb, index: index, ropeTable: table,
                                  context: attentionContext(block: i, stepIndex: stepIndex,
                                                            stepCount: stepCount, layout: layout))
                    if Self.tappedBlocks.contains(i) { tapsOut.blocks[i] = h }
                }
                cache.record(totalResidual: h - hIn)
            }
        } else {
            for (i, block) in blocks.enumerated() {
                h = block(h, tEmb: tEmb, index: index, ropeTable: table,
                          context: attentionContext(block: i, stepIndex: stepIndex,
                                                    stepCount: stepCount, layout: layout))
                if Self.tappedBlocks.contains(i) { tapsOut.blocks[i] = h }
            }
        }
    }

    func makePackedPass(videoLatent: MLXArray, audioLatent: MLXArray,
                        context: MLXArray, layout: PackedLayout,
                        plan: TimestepPlan, index: ModulationIndex,
                        tapsOut: inout Taps,
                        renderState: RenderState?,
                        condVideo: MLXArray?,
                        condAudio: MLXArray?) throws -> PackedPass {
        // text path — the context sets the compute dtype in the reference, so
        // cast it here rather than inheriting whatever the caller supplied. A
        // fp32 golden read straight in would silently run the whole stack at
        // fp32 and land outside the bf16 equivalence class.
        let textStates: MLXArray
        let refined: MLXArray
        if let renderState {
            precondition(renderState.layout == layout && renderState.contextTokens == context.dim(1),
                         "RenderState does not match this render's packed layout or context length")
            refined = renderState.refined
            textStates = renderState.textStates
        } else {
            textStates = linear(context[0].asType(computeDType), conditionProj)
            refined = tokenRefiner(textStates)
        }
        tapsOut.conditionProj = textStates
        tapsOut.tokenRefiner = refined

        let videoRows = H3Packing.patchifyVideo(videoLatent.asType(.float32),
                                                patch: config.patchSize)
        let audioRows = H3Packing.packAudio(audioLatent.asType(.float32))

        var allVideoRows = [MLXArray]()
        var condVideoOffset = 0
        for s in layout.segments {
            if s.kind == .cond || s.kind == .refImage {
                guard let condVideo = condVideo else {
                    throw H3Error.invalidRequest(
                        rule: "missing conditioning rows",
                        detail: "the packed layout declares a \(s.kind.rawValue) segment of "
                              + "\(s.count) row(s), and no conditioning video was supplied",
                        remedy: "encode every declared condition before sampling; the layout "
                              + "and the rows are built from the same latents for this reason.")
                }
                let slice = condVideo[condVideoOffset ..< (condVideoOffset + s.count)]
                allVideoRows.append(slice)
                condVideoOffset += s.count
            } else if s.kind == .video {
                allVideoRows.append(videoRows)
            }
        }
        let videoEmbed = linear(concatenated(allVideoRows, axis: 0), videoPatchProj)

        var allAudioRows = [MLXArray]()
        var condAudioOffset = 0
        for s in layout.segments {
            if s.kind == .refAudio {
                guard let condAudio = condAudio else {
                    throw H3Error.invalidRequest(
                        rule: "missing conditioning rows",
                        detail: "the packed layout declares a \(s.kind.rawValue) segment of "
                              + "\(s.count) row(s), and no conditioning audio was supplied",
                        remedy: "encode every declared condition before sampling; the layout "
                              + "and the rows are built from the same latents for this reason.")
                }
                let slice = condAudio[condAudioOffset ..< (condAudioOffset + s.count)]
                allAudioRows.append(slice)
                condAudioOffset += s.count
            } else if s.kind == .audio {
                allAudioRows.append(audioRows)
            }
        }
        let audioEmbed = linear(concatenated(allAudioRows, axis: 0), audioPatchProj)

        tapsOut.videoPatchProj = videoEmbed
        tapsOut.audioPatchProj = audioEmbed

        let dtype = computeDType
        var hSegments = [MLXArray]()
        var vEmbedOffset = 0
        var aEmbedOffset = 0
        for s in layout.segments {
            switch s.kind {
            case .text:
                hSegments.append(refined.asType(dtype))
            case .cond, .refImage, .video:
                let slice = videoEmbed[vEmbedOffset ..< (vEmbedOffset + s.count)].asType(dtype)
                hSegments.append(slice)
                vEmbedOffset += s.count
            case .refAudio, .audio:
                let slice = audioEmbed[aEmbedOffset ..< (aEmbedOffset + s.count)].asType(dtype)
                hSegments.append(slice)
                aEmbedOffset += s.count
            }
        }
        let h = concatenated(hSegments, axis: 0)
        precondition(h.dim(0) == layout.totalTokens,
                     "packed \(h.dim(0)) rows, layout says \(layout.totalTokens)")

        let tEmbFP32 = timeEmbedder(MLXArray(plan.values))
        tapsOut.timeEmbedder = tEmbFP32
        let tEmb = tEmbFP32.asType(dtype)

        let table: MLXArray
        if let renderState {
            table = renderState.ropeTable
        } else {
            let pos = MLXArray(layout.positionIds.map { Float($0) }, [layout.totalTokens, 3])
            table = H3RoPE.rotationTable(
                angles: H3RoPE.angles(positionIds: pos, invFreq: ropeInvFreq)).asType(dtype)
        }
        return PackedPass(h: h, tEmb: tEmb, table: table, layout: layout, plan: plan, index: index)
    }

    func finalize(_ pass: PackedPass, tapsOut: inout Taps) -> (video: MLXArray, audio: MLXArray) {
        let videoSeg = ModSegment(start: pass.layout.videoRange.lowerBound,
                                  stop: pass.layout.videoRange.upperBound,
                                  row: pass.plan.row(for: .video))
        let audioSeg = ModSegment(start: pass.layout.audioRange.lowerBound,
                                  stop: pass.layout.audioRange.upperBound,
                                  row: pass.plan.row(for: .audio))
        let (v, a) = finalLayer(pass.h, tEmb: pass.tEmb, videoSeg: videoSeg, audioSeg: audioSeg)
        tapsOut.finalVideo = v
        tapsOut.finalAudio = a
        return (v, a)
    }

    /// One forward pass, in packed-row space.
    ///
    /// Returns the final layer's raw head outputs — `[videoTokens, 96]` and
    /// `[audioTokens, 32]` — which is where the contract's `final_layer.0` and
    /// `final_layer.1` taps sit. Latent-shaped velocity comes from
    /// ``velocity(videoLatent:audioLatent:context:sigmaVideo:geometry:textTags:taps:)``.
    ///
    /// - Parameters:
    ///   - videoLatent:    ///   - audioLatent: `[1,32,2,audioT]`
    ///   - context: `[1, textLen, textDim]` — the conditioning the contract feeds you
    ///   - layout: carries the segment table and `[S,3]` position ids
    ///   - stepIndex: where in the sampling schedule this call sits. Two things
    ///     need it — the cross-step cache, which will not reuse a residual
    ///     during warm-up or on the final step, and a sparse attention backend,
    ///     which has a dense warm-up of its own. One pair of arguments serves
    ///     both; two pairs would eventually disagree. (These were `cacheStep`
    ///     and `cacheTotalSteps`, named for their first consumer.) Omitted, the
    ///     cache is inactive and attention runs dense.
    package func packedForward(videoLatent: MLXArray, audioLatent: MLXArray,
                              context: MLXArray, layout: PackedLayout,
                              plan: TimestepPlan, index: ModulationIndex,
                              tapsOut: inout Taps,
                              renderState: RenderState? = nil,
                              condVideo: MLXArray? = nil,
                              condAudio: MLXArray? = nil,
                              stepCache: H3StepCache? = nil,
                              stepIndex: Int? = nil,
                              stepCount: Int? = nil)
        throws -> (video: MLXArray, audio: MLXArray) {
        var pass = try makePackedPass(videoLatent: videoLatent, audioLatent: audioLatent,
                                      context: context, layout: layout, plan: plan, index: index,
                                      tapsOut: &tapsOut, renderState: renderState,
                                      condVideo: condVideo, condAudio: condAudio)
        applyBlocks(to: &pass.h, tEmb: pass.tEmb, index: pass.index, table: pass.table,
                    layout: pass.layout, plan: pass.plan, tapsOut: &tapsOut,
                    stepCache: stepCache, stepIndex: stepIndex, stepCount: stepCount)
        return finalize(pass, tapsOut: &tapsOut)
    }

    /// Latent-shaped velocity for one sampler step, matching the reference's
    /// return value exactly.
    ///
    /// Two sign conventions are baked in and neither is cosmetic: **both streams
    /// are negated**, and audio is additionally scaled by `d(sigma_a)/d(sigma_v)`
    /// so that the single flat ODE the sampler integrates is each stream's true
    /// ODE on its own shifted schedule.
    package func velocity(videoLatent: MLXArray, audioLatent: MLXArray,
                          context: MLXArray, sigmaVideo: Double,
                          geometry: LatentGeometry, textTags: [Int]? = nil,
                          keyframes: [KeyframeConfig] = [],
                          refs: [ReferenceBlock] = [],
                          condVideo: MLXArray? = nil,
                          condAudio: MLXArray? = nil,
                          renderState: RenderState? = nil,
                          stepCache: H3StepCache? = nil,
                          stepIndex: Int? = nil,
                          stepCount: Int? = nil,
                          taps: inout Taps) throws -> (video: MLXArray, audio: MLXArray) {
        let layout = try renderState?.layout ?? PackedLayout(textTokens: context.dim(1), geometry: geometry,
                                                             keyframes: keyframes, refs: refs)
        let plan = TimestepPlan(sigmaVideo: sigmaVideo, segments: layout.segments)
        let index = ModulationIndex(layout: layout, plan: plan, textTags: textTags)
        let (v, a) = try packedForward(videoLatent: videoLatent, audioLatent: audioLatent,
                                   context: context, layout: layout, plan: plan,
                                   index: index, tapsOut: &taps,
                                   renderState: renderState,
                                   condVideo: condVideo, condAudio: condAudio,
                                   stepCache: stepCache,
                                   stepIndex: stepIndex,
                                   stepCount: stepCount)
        let video = H3Packing.unpatchifyVideo(v, t: geometry.latentT,
                                              h: geometry.latentH / config.patchSize[1],
                                              w: geometry.latentW / config.patchSize[2],
                                              channels: config.videoLatentDim,
                                              patch: config.patchSize)
        let audio = H3Packing.unpackAudio(a)
        return (-video, MLXArray(-plan.audioSlope) * audio)
    }

    /// Classifier-free guidance over one sampler step.
    ///
    ///     guided = neg + scale * (cond - neg)
    ///
    /// With `negative` nil the unconditional term is a zero-length context,
    /// which is the null-context form; with `negative` supplied it is the
    /// negative-prompt form, and the trajectory is pushed away from those
    /// concepts at every step rather than merely not toward them.
    ///
    /// The two passes are independent — this model is batch-1 only, and the
    /// packed sequence has no batch axis to widen — so guidance is a second
    /// full forward. At production shape that is 61 s per pass if they run
    /// back to back.
    ///
    /// Under `H3_ANE_CFG_OVERLAP=1` the two block stacks are **pipelined**, the
    /// branches half a block out of phase, so the engine works on one while the
    /// GPU attends on the other. The math of each branch is unchanged — the
    /// schedule is bit-identical to the sequential path — and it is off by
    /// default because it does not win: the engine runs at 7.9 TFLOP/s against
    /// the GPU's 19.4, and the routing overhead of moving more columns to it
    /// exceeds what overlapping them recovers. `docs/ANE_STATUS.md` carries the
    /// table and the arithmetic.
    ///
    /// Taps come from the CONDITIONAL pass, so a parity run at scale 1.0 is
    /// byte-comparable with a run with guidance off.
    ///
    /// **Unverified against the reference.** The ComfyUI H3 graph uses
    /// `BasicGuider`, i.e. no CFG, so no golden exercises this path. The
    /// combination is the standard one; the claim that it is what MiniMax's own
    /// stack does is not something this repo has measured.
    package func guidedVelocity(videoLatent: MLXArray, audioLatent: MLXArray,
                               context: MLXArray, negative: MLXArray?,
                               scale: Float, sigmaVideo: Double,
                               geometry: LatentGeometry,
                               textTags: [Int]? = nil,
                               negativeTextTags: [Int]? = nil,
                               keyframes: [KeyframeConfig] = [],
                               refs: [ReferenceBlock] = [],
                               condVideo: MLXArray? = nil,
                               condAudio: MLXArray? = nil,
                               renderState: RenderState? = nil,
                               negativeRenderState: RenderState? = nil,
                               stepCache: H3StepCache? = nil,
                               negativeStepCache: H3StepCache? = nil,
                               stepIndex: Int? = nil,
                               stepCount: Int? = nil,
                               taps: inout Taps) throws -> (video: MLXArray, audio: MLXArray) {
        // scale 1.0 is the identity; skip the second pass rather than paying
        // 61 s to multiply by one.
        guard scale != 1.0 else {
            return try velocity(videoLatent: videoLatent, audioLatent: audioLatent,
                                context: context, sigmaVideo: sigmaVideo,
                                geometry: geometry, textTags: textTags,
                                keyframes: keyframes, refs: refs,
                                condVideo: condVideo, condAudio: condAudio,
                                renderState: renderState,
                                stepCache: stepCache, stepIndex: stepIndex,
                                stepCount: stepCount, taps: &taps)
        }

        if ANELinearBackend.isEnabled && ANELinearBackend.cfgOverlapEnabled {
            return try overlappedGuidedVelocity(
                videoLatent: videoLatent, audioLatent: audioLatent,
                context: context, negative: negative, scale: scale,
                sigmaVideo: sigmaVideo, geometry: geometry,
                textTags: textTags, negativeTextTags: negativeTextTags,
                keyframes: keyframes, refs: refs,
                condVideo: condVideo, condAudio: condAudio,
                renderState: renderState, negativeRenderState: negativeRenderState,
                stepCache: stepCache, negativeStepCache: negativeStepCache,
                stepIndex: stepIndex, stepCount: stepCount, taps: &taps)
        }

        let cond = try velocity(videoLatent: videoLatent, audioLatent: audioLatent,
                            context: context, sigmaVideo: sigmaVideo,
                            geometry: geometry, textTags: textTags,
                            keyframes: keyframes, refs: refs,
                            condVideo: condVideo, condAudio: condAudio,
                            renderState: renderState,
                            stepCache: stepCache, stepIndex: stepIndex,
                            stepCount: stepCount,
                            taps: &taps)

        var negTaps = Taps()
        let uncondContext = negative ?? MLXArray.zeros(
            [context.dim(0), context.dim(1), context.dim(2)], dtype: context.dtype)
        let uncondTags = negative == nil ? textTags : negativeTextTags
        let uncond = try velocity(videoLatent: videoLatent, audioLatent: audioLatent,
                              context: uncondContext, sigmaVideo: sigmaVideo,
                              geometry: geometry, textTags: uncondTags,
                              keyframes: keyframes, refs: refs,
                              condVideo: condVideo, condAudio: condAudio,
                              renderState: negativeRenderState,
                              stepCache: negativeStepCache, stepIndex: stepIndex,
                              stepCount: stepCount,
                              taps: &negTaps)

        let s = MLXArray(scale)
        return (uncond.video + s * (cond.video - uncond.video),
                uncond.audio + s * (cond.audio - uncond.audio))
    }

    func overlappedGuidedVelocity(videoLatent: MLXArray, audioLatent: MLXArray,
                                  context: MLXArray, negative: MLXArray?,
                                  scale: Float, sigmaVideo: Double,
                                  geometry: LatentGeometry,
                                  textTags: [Int]?, negativeTextTags: [Int]?,
                                  keyframes: [KeyframeConfig], refs: [ReferenceBlock],
                                  condVideo: MLXArray?, condAudio: MLXArray?,
                                  renderState: RenderState?,
                                  negativeRenderState: RenderState?,
                                  stepCache: H3StepCache?,
                                  negativeStepCache: H3StepCache?,
                                  stepIndex: Int?, stepCount: Int?,
                                  taps: inout Taps) throws -> (video: MLXArray, audio: MLXArray) {
        let layoutC = try renderState?.layout ?? PackedLayout(textTokens: context.dim(1),
                                                             geometry: geometry,
                                                             keyframes: keyframes, refs: refs)
        let planC = TimestepPlan(sigmaVideo: sigmaVideo, segments: layoutC.segments)
        let indexC = ModulationIndex(layout: layoutC, plan: planC, textTags: textTags)
        var condPass = try makePackedPass(videoLatent: videoLatent, audioLatent: audioLatent,
                                          context: context, layout: layoutC, plan: planC,
                                          index: indexC, tapsOut: &taps,
                                          renderState: renderState,
                                          condVideo: condVideo, condAudio: condAudio)

        let uncondContext = negative ?? MLXArray.zeros(
            [context.dim(0), context.dim(1), context.dim(2)], dtype: context.dtype)
        let uncondTags = negative == nil ? textTags : negativeTextTags
        let layoutU = try negativeRenderState?.layout
            ?? PackedLayout(textTokens: uncondContext.dim(1), geometry: geometry,
                            keyframes: keyframes, refs: refs)
        let planU = TimestepPlan(sigmaVideo: sigmaVideo, segments: layoutU.segments)
        let indexU = ModulationIndex(layout: layoutU, plan: planU, textTags: uncondTags)
        var negTaps = Taps()
        var uncondPass = try makePackedPass(videoLatent: videoLatent, audioLatent: audioLatent,
                                            context: uncondContext, layout: layoutU, plan: planU,
                                            index: indexU, tapsOut: &negTaps,
                                            renderState: negativeRenderState,
                                            condVideo: condVideo, condAudio: condAudio)

        ANELinearBackend.markCFGOverlap()
        overlapStacks(cond: &condPass, uncond: &uncondPass,
                      condCache: stepCache, uncondCache: negativeStepCache,
                      stepIndex: stepIndex, stepCount: stepCount,
                      condTaps: &taps)

        let cond = decodeVelocity(finalize(condPass, tapsOut: &taps),
                                  geometry: geometry, plan: planC)
        let uncond = decodeVelocity(finalize(uncondPass, tapsOut: &negTaps),
                                    geometry: geometry, plan: planU)
        let s = MLXArray(scale)
        return (uncond.video + s * (cond.video - uncond.video),
                uncond.audio + s * (cond.audio - uncond.audio))
    }

    func decodeVelocity(_ heads: (video: MLXArray, audio: MLXArray),
                        geometry: LatentGeometry, plan: TimestepPlan)
        -> (video: MLXArray, audio: MLXArray) {
        let video = H3Packing.unpatchifyVideo(heads.video, t: geometry.latentT,
                                              h: geometry.latentH / config.patchSize[1],
                                              w: geometry.latentW / config.patchSize[2],
                                              channels: config.videoLatentDim,
                                              patch: config.patchSize)
        let audio = H3Packing.unpackAudio(heads.audio)
        return (-video, MLXArray(-plan.audioSlope) * audio)
    }
}

/// Two pre-norm blocks with plain residuals, then a final RMSNorm. No AdaLN,
/// no RoPE — the refiner sees text only.
package struct TokenRefiner {
    package struct Block {
        package let norm1: H3RMSNorm
        package let norm2: H3RMSNorm
        package let attn: AttentionLayer
        package let mlp: H3MLP
        package init(norm1: H3RMSNorm, norm2: H3RMSNorm, attn: AttentionLayer, mlp: H3MLP) {
            self.norm1 = norm1
            self.norm2 = norm2
            self.attn = attn
            self.mlp = mlp
        }
        package func callAsFunction(_ x: MLXArray) -> MLXArray {
            let a = attn(norm1(x), ropeTable: nil) + x
            return mlp(norm2(a)) + a
        }
    }
    package let blocks: [Block]
    package let finalNorm: H3RMSNorm
    package init(blocks: [Block], finalNorm: H3RMSNorm) {
        self.blocks = blocks
        self.finalNorm = finalNorm
    }
    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for b in blocks { h = b(h) }
        return finalNorm(h)
    }
}
