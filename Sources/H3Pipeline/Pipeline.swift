// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Catalog
import H3Foundation
import H3Hardware
import H3Attention
import H3Modules

/// A render, start to finish.
///
/// **The phase order below is a memory contract, not a narrative one.** Each
/// stage is loaded, used and released before the next begins, and the single
/// most important consequence is that every condition is encoded *before* the
/// DiT is loaded. Hold the 51.5 GB text encoder and the 66.3 GB DiT at once and
/// a 275 GB machine is carrying 118 GB of mapped checkpoint before it has
/// allocated a single activation.
///
///   1. text conditioning   -> release, clear the cache
///   2. VAE encode          -> release, clear the cache
///   3. sample              -> keep the cache; the sampler wants it
///   4. decode, then write
///
/// The cache is *not* cleared after the DiT loads, and that asymmetry is
/// deliberate: MLX pools freed buffers rather than returning them to the OS, so
/// clearing at a boundary the sampler is about to allocate across would only
/// make it allocate again.
package enum PipelineRuntime {

    /// The dtype the DiT block stack runs in, overridable for one specific
    /// diagnostic and nothing else.
    ///
    /// Contract 8 pins production at bf16 and this does not change that: the
    /// default is unchanged and the override has to be asked for by name. It
    /// exists because `docs/ANE.md` records the ANE's
    /// arithmetic as *more* accurate than bf16 per projection — 7e-5 to 5e-4
    /// against the bf16 GPU path's 1.66e-3 — and that result is single-shot.
    /// A forward is 50 blocks and a render is 20 of them, so the question it
    /// cannot answer is whether a per-projection improvement survives a
    /// thousand evaluations of a diffusion trajectory, which can compound an
    /// error as easily as it can wash one out.
    ///
    /// fp16 with a wide accumulator is the closest proxy for the engine's
    /// datapath that runs on the GPU, so this answers the propagation question
    /// without any private API, any bridge, or any of the integration work the
    /// ANE route would need. If a full render holds at fp16, the arithmetic is
    /// not what stands in the way; if it drifts, none of the rest matters.
    ///
    ///     H3_DIT_DTYPE=fp16 h3 render ...
    package static func diagnosticComputeDType(log: (String) -> Void) -> DType {
        switch ProcessInfo.processInfo.environment["H3_DIT_DTYPE"] {
        case "fp16", "float16":
            log("DiT compute dtype overridden to fp16 — diagnostic only, "
                + "contract 8 pins production at bf16")
            return .float16
        case "fp32", "float32":
            log("DiT compute dtype overridden to fp32 — diagnostic only, "
                + "2x residency")
            return .float32
        default:
            return .bfloat16
        }
    }

    /// How this process computed a DiT block, for the render receipt.
    ///
    /// Two things change the arithmetic and therefore reselect the diffusion
    /// sample: the compute dtype, and whether projections ran on the Neural
    /// Engine. Neither was recorded, so two renders from the same seed and
    /// checkpoint that produced visibly different videos were indistinguishable
    /// in their receipts — which is precisely the claim a receipt exists to
    /// support.
    package struct ArithmeticProfile: Sendable, Codable, Equatable {
        /// `bf16`, `fp16` or `fp32`.
        package let ditDType: String
        /// Projections the engine actually computed, observed rather than
        /// declared.
        package let aneRoutedProjections: [String]
        /// Projections offered to the engine that fell back to MLX. A render
        /// that declined partway is not the render that did not offer at all.
        package let aneDeclinedProjections: [String]
        /// GPU attention on one CFG branch ran beside engine linears on the other.
        package let aneCFGOverlap: Bool
        /// Pieces the contraction was cut into, 0 if it was never split.
        package let aneSplitContraction: Int

        package static func observed() -> ArithmeticProfile {
            let dtype: String
            switch diagnosticComputeDType(log: { _ in }) {
            case .float16: dtype = "fp16"
            case .float32: dtype = "fp32"
            default:       dtype = "bf16"
            }
            if ANELinearBackend.fc2Verify {
                FileHandle.standardError.write(Data(DiTBlock.fc2Audit.report().utf8))
            }
            // Busy-seconds against wall time says whether the unrealized linear
            // gain is a bad GPU/ANE split or dies sitting idle between
            // submissions. Those want opposite fixes, so measure before tuning.
            let phases = ANELinearBackend.PhaseMeter.report()
            if !phases.isEmpty { FileHandle.standardError.write(Data(phases.utf8)) }
            if ProcessInfo.processInfo.environment["H3_ANE_UTILISATION"] == "1" {
                let busy = ANELinearBackend.EngineMeter.busySeconds
                FileHandle.standardError.write(Data(
                    String(format: "ane engine busy %.1f s\n", busy).utf8))
            }
            return ArithmeticProfile(
                ditDType: dtype,
                aneRoutedProjections: ANELinearBackend.routedProjections,
                aneDeclinedProjections: ANELinearBackend.declinedProjections,
                aneCFGOverlap: ANELinearBackend.overlappedCFG,
                aneSplitContraction: ANELinearBackend.splitContractions)
        }
    }

    /// Where the checkpoints are.
    ///
    /// Resolved by the caller — usually from `H3Configuration` and `Catalog` —
    /// so this type does no discovery of its own and can be pointed anywhere.
    package struct Checkpoints: Sendable {
        package let dit: URL
        package let textEncoder: URL
        package let tokenizer: URL
        package let videoVAE: URL
        package let audioVAE: URL

        package init(dit: URL, textEncoder: URL, tokenizer: URL,
                    videoVAE: URL, audioVAE: URL) {
            self.dit = dit
            self.textEncoder = textEncoder
            self.tokenizer = tokenizer
            self.videoVAE = videoVAE
            self.audioVAE = audioVAE
        }
    }

    package static let phaseOrder = RenderProgress.Phase.allCases

    /// - Parameters:
    ///   - progress: called on the calling thread at each phase boundary and
    ///     after each sampler step.
    ///   - cancellation: observed between sampler steps. A step is minutes at
    ///     production shape, so this cannot be instant, and pretending otherwise
    ///     would mean abandoning a half-finished forward for nothing.
    ///   - log: diagnostics. The CLI prints these; a library caller can ignore
    ///     them without losing anything a thrown error would have carried.
    package static func render(request: RenderRequest,
                              checkpoints: Checkpoints,
                              conditioningLatents: (video: URL?, audio: URL?) = (nil, nil),
                              progress: (RenderProgress) -> Void = { _ in },
                              cancellation: RenderCancellation? = nil,
                              log: (String) -> Void = { _ in }) async throws -> RenderResult {

        try request.validate()
        try checkCancellation(cancellation, phase: .textConditioning,
                              detail: "before checkpoint preflight")
        // Before anything expensive: MLX cannot run at all without its Metal
        // kernels, and the error it raises on its own is an untyped C++ throw
        // with no path in it.
        try MetalLibrary.preflight()

        let (width, height) = try request.dimensions()
        let geometry = LatentGeometry(width: width, height: height,
                                      length: request.seconds * H3Video.fps)

        // The render policy. These were warnings once; the warning was printed,
        // ignored by its own author, and cost an evening chasing a decoder bug
        // that did not exist. They refuse now.
        let violations = request.policyViolations(geometry: geometry)
        if !violations.isEmpty {
            if request.allowSuboptimal {
                log("render policy: \(violations.count) problem(s), proceeding anyway.")
                for v in violations { log("  * \(v.rule): \(v.reason)") }
                log("  Do not read the output as a quality or correctness signal.")
            } else {
                throw H3Error.policyViolations(violations)
            }
        }

        // The attention backend is resolved once, here, and logged. A render
        // that is half one backend and half another is not something this
        // should be able to express.
        let selection = try AttentionRegistry.resolve(requested: request.attentionBackend,
                                                     machine: Machine.detect())
        log("  attention: \(selection.identifier) — \(selection.reason)")

        var timings = RenderResult.Timings()
        let began = Date()
        func mark(_ phase: RenderProgress.Phase, _ detail: String,
                  completed: Int = 0, total: Int = 0) {
            progress(RenderProgress(phase: phase, completed: completed, total: total,
                                    detail: detail, elapsed: Date().timeIntervalSince(began)))
        }

        // Reference video is decoded once and used twice — Qwen sees a 2 fps
        // subsample of these exact frames, the video VAE encodes all of them —
        // so decoding per consumer would let the two halves drift apart while
        // every shape stayed valid.
        var referenceVideoFrames: [MLXArray] = []
        for url in request.referenceVideos {
            referenceVideoFrames.append(
                try await ConditionEncoder.loadReferenceVideo(
                    at: url, frameCount: geometry.frameCount, log: log))
        }

        // ---- 1. text conditioning
        mark(.textConditioning, "loading the text encoder")
        var phase = Date()
        let conditioning = try ConditioningEncoder.encode(
            request: request, textEncoder: checkpoints.textEncoder,
            tokenizer: checkpoints.tokenizer, referenceVideoFrames: referenceVideoFrames,
            log: log)
        timings.textConditioning = Date().timeIntervalSince(phase)
        // The encoder and the tower are out of scope by here. Their buffers are
        // not: MLX pools freed allocations, so without this the 51.5 GB
        // checkpoint — held twice, once by each — is still resident when the DiT
        // loads.
        Memory.clearCache()
        reportMemory("after text conditioning", log: log)
        try checkCancellation(cancellation, phase: .textConditioning,
                              detail: "after text conditioning")

        // ---- 2. conditions through the VAEs
        mark(.conditionEncoding, "encoding conditions")
        phase = Date()
        let visualLatents = try conditioningLatents.video != nil
            ? ConditionEncoder.loadLatents(conditioningLatents.video)
            : ConditionEncoder.encodeVisual(request: request, videoVAE: checkpoints.videoVAE,
                                            width: width, height: height,
                                            frameCount: geometry.frameCount,
                                            referenceVideoFrames: referenceVideoFrames, log: log)
        let audioLatents = try conditioningLatents.audio != nil
            ? ConditionEncoder.loadLatents(conditioningLatents.audio)
            : ConditionEncoder.encodeAudio(request: request, audioVAE: checkpoints.audioVAE,
                                           log: log)
        let conditions = try ConditionEncoder.assemble(
            request: request, geometry: geometry,
            visualLatents: visualLatents, audioLatents: audioLatents,
            recordedNoise: try ConditionEncoder.loadLatents(request.conditioningNoise), log: log)
        timings.conditionEncoding = Date().timeIntervalSince(phase)
        Memory.clearCache()
        reportMemory("after conditioning and VAE encodes", log: log)

        try checkCancellation(cancellation, phase: .conditionEncoding,
                              detail: "before the DiT was loaded")

        // ---- 3. sampling
        mark(.sampling, "loading the DiT", completed: 0, total: request.steps)
        phase = Date()
        let weights = try H3Weights(url: checkpoints.dit)
        let model = try H3Transformer(weights: weights,
                                      computeDType: PipelineRuntime.diagnosticComputeDType(log: log),
                                      backend: selection.backend)
        timings.modelLoad = Date().timeIntervalSince(phase)
        reportMemory("after the DiT load", log: log)

        // A distilled schedule is resolved in `RenderEngine.execute`, before the
        // request is staged, so that the receipt and the estimate see the count
        // that will actually run. Belt and braces: a caller reaching this
        // directly still gets it, because the sampler must never guess.
        var request = request
        // A debugging escape: run a distilled checkpoint as if it were the base
        // model, to separate a weight-conversion fault from a sampler fault.
        let ignoreDistilled =
            ProcessInfo.processInfo.environment["H3_IGNORE_DISTILLED"] == "1"
        if ignoreDistilled { request.distilledSteps = nil }
        if !ignoreDistilled, request.distilledSteps == nil,
           let steps = (try? CheckpointIdentity.identify(url: checkpoints.dit))?.distilledSteps {
            request.distilledSteps = steps
            request.steps = steps.count
        }

        phase = Date()
        let sampled = try SamplingLoop.run(
            model: model, request: request, geometry: geometry,
            conditioning: conditioning, conditions: conditions,
            cancellation: cancellation,
            onStep: { done, total in
                mark(.sampling, "step \(done) of \(total)", completed: done, total: total)
            },
            log: log)
        timings.sampling = Date().timeIntervalSince(phase)
        try checkCancellation(cancellation, phase: .sampling,
                              detail: "before decoder checkpoints were loaded")

        // ---- 4. decode
        mark(.decoding, "decoding audio")
        phase = Date()
        let audioVAE = try MiniMaxH3AudioVAE(url: checkpoints.audioVAE)
        let waveform = audioVAE.decode(sampled.audio)              // [1, 2, L]
        eval(waveform)
        timings.audioDecode = Date().timeIntervalSince(phase)
        try checkCancellation(cancellation, phase: .decoding,
                              detail: "after audio decode")

        mark(.decoding, "decoding video")
        phase = Date()
        let videoVAE = try VideoVAE(url: checkpoints.videoVAE)
        let frames = videoVAE.decode(sampled.video)                // [1, 3, T, H, W]
        eval(frames)
        timings.videoDecode = Date().timeIntervalSince(phase)
        try checkCancellation(cancellation, phase: .decoding,
                              detail: "before output packing")

        // ---- 5. write
        mark(.writing, "packing pixels")
        let samples = deinterleave(waveform)
        if let audioURL = request.audioOutput {
            try MovieWriter.writeWAV(samples: samples, to: audioURL)
        }

        // Pack on the GPU: [T,H,W,3] float in [-1,1] -> [T,H,W,4] uint8 ARGB.
        // The per-pixel Swift loop this replaces was ~580M bounds-checked
        // iterations at 2K.
        phase = Date()
        let t = frames.dim(2), fh = frames.dim(3), fw = frames.dim(4)
        let rgb = clip((frames + 1.0) * 127.5, min: 0.0, max: 255.0)
            .transposed(0, 2, 3, 4, 1).reshaped([t, fh, fw, 3])
        let alpha = MLXArray.full([t, fh, fw, 1], values: MLXArray(255.0 as Float))
        let argb = concatenated([alpha, rgb], axis: -1).asType(.uint8)
        eval(argb)
        timings.pixelPack = Date().timeIntervalSince(phase)

        mark(.writing, "muxing")
        phase = Date()
        var muxedAudio = true
        do {
            try await MovieWriter.writeMovie(
                argb: argb, waveform: samples, to: request.videoOutput,
                fps: Double(H3Video.fps), sampleRate: H3Audio.sampleRate)
        } catch {
            // A mux failure must not cost a thirty-minute render. Fall back to
            // video-only; when a wav was requested it still carries the audio.
            log("warning: muxing audio failed (\(error)); retrying video-only."
                + (request.audioOutput.map { " The audio is still in \($0.lastPathComponent)." }
                   ?? ""))
            muxedAudio = false
            try await MovieWriter.writeMovie(
                argb: argb, waveform: samples, to: request.videoOutput,
                fps: Double(H3Video.fps), sampleRate: H3Audio.sampleRate,
                withAudio: false)
        }
        timings.mux = Date().timeIntervalSince(phase)

        if let summary = sampled.cacheSummary { log("  " + summary) }
        reportMemory("final", log: log)

        var result = RenderResult(video: request.videoOutput, audio: request.audioOutput,
                                  frameCount: t, width: fw, height: fh,
                                  seconds: Double(t) / Double(H3Video.fps),
                                  timings: timings, cacheSummary: sampled.cacheSummary,
                                  muxedAudio: muxedAudio)
        result.trace = sampled.trace
        result.attentionBackend = selection.identifier
        // Read here, at the end, because `Memory.peakMemory` is a high-water
        // mark for the process and the pipeline's `clearCache()` calls between
        // phases free buffers without lowering it. Sampling is the peak in every
        // configuration measured, but that is a finding to keep checking rather
        // than a fact to bake into where the reading is taken.
        result.mlxPeakBytes = UInt64(Memory.peakMemory)
        result.mlxActiveBytesAtEnd = UInt64(Memory.activeMemory)
        return result
    }

    /// `[1, 2, L]` to two channel arrays.
    private static func deinterleave(_ waveform: MLXArray) -> [[Float]] {
        let length = waveform.dim(2)
        let flat = waveform.reshaped([waveform.dim(0) * waveform.dim(1), length])
            .asType(.float32).asArray(Float.self)
        return (0 ..< 2).map { Array(flat[($0 * length) ..< (($0 + 1) * length)]) }
    }

    private static func checkCancellation(_ cancellation: RenderCancellation?,
                                          phase: RenderProgress.Phase,
                                          detail: String) throws {
        if Task.isCancelled || cancellation?.isCancelled == true {
            throw RenderCancelled(phase: phase, detail: detail)
        }
    }

    /// **Read the two numbers this prints as different things, because they
    /// are.** `MLX.GPU.peakMemory` counts what MLX *allocated*. It does not
    /// count the checkpoints, which `loadArrays` memory-maps — so a render
    /// holding 117.8 GB of mapped DiT and text-encoder pages reports a 53.1 GB
    /// peak, and both figures are honest. Jetsam counts resident pages, mapped
    /// or not, which is why its number was nearly four times this one when a
    /// render was killed at 196.2 GB on a 275 GB machine.
    ///
    /// A render that looks comfortable here can still be the largest process on
    /// the box. That kill was a crowding — a second renderer — not a leak and
    /// not an oversized payload.
    private static func reportMemory(_ stage: String, log: (String) -> Void) {
        let g = 1e9
        log(String(format: "  memory: %@ — active %.1f GB, cache %.1f GB, peak %.1f GB",
                   stage, Double(Memory.activeMemory) / g,
                   Double(Memory.cacheMemory) / g, Double(Memory.peakMemory) / g))
    }
}
