// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXRandom
import H3Foundation
import H3Modules

/// Phase 3: the sampler loop, and the only phase whose peak binds the machine.
///
/// Split from the pipeline because it is the one part worth reading on its own:
/// everything before it prepares inputs and everything after it consumes
/// outputs, but this is where the 66 GB of weights meets tens of gigabytes of
/// activations, twenty times over.
enum SamplingLoop {

    struct Output {
        let video: MLXArray
        let audio: MLXArray
        let cacheSummary: String?
        /// Every step's measurement, decision and wall clock. Present whether or
        /// not the cache ran — a dense control has no decisions to record but
        /// still has the step times a cached arm has to be compared against,
        /// and a control that produced no machine-readable record would have to
        /// be re-run by hand every time something wanted to check it.
        let trace: SamplingTrace
    }

    /// - Parameter onStep: called after each completed step, and the only place
    ///   a cancellation can be observed cheaply. A step is minutes at production
    ///   shape, so checking inside one would mean checking inside the block
    ///   loop, and abandoning a half-finished forward leaves nothing useful.
    static func run(model: H3Transformer,
                    request: RenderRequest,
                    geometry: LatentGeometry,
                    conditioning: TextConditioning,
                    conditions: EncodedConditions,
                    cancellation: RenderCancellation?,
                    onStep: (Int, Int) -> Void = { _, _ in },
                    log: (String) -> Void = { _ in }) throws -> Output {

        if request.cfgScale > 1.0 && ANELinearBackend.isEnabled {
            log("  CFG overlap: GPU attention on one branch beside engine linears on the other")
        }

        MLXRandom.seed(request.seed)
        let noiseVideo = MLXRandom.normal([1, 24, geometry.latentT,
                                           geometry.latentH, geometry.latentW])
        let noiseAudio = MLXRandom.normal([1, 32, 2, geometry.audioT])

        // A distilled checkpoint carries its own denoising steps; the flow
        // schedule does not apply to it. See `DistilledSchedule`.
        let schedule = FlowSchedule(shift: H3Shift.video)
        let distilled = request.distilledSteps.map { DistilledSchedule(timesteps: $0) }
        let sigmas = distilled?.sigmas ?? schedule.sigmas(steps: request.steps)
        // The loop is bound to the schedule, not to `request.steps`: a
        // distilled checkpoint decides its own count, and indexing a four-entry
        // schedule with a twenty-step request runs off the end.
        let steps = sigmas.count - 1
        if let distilled {
            log("  distilled schedule: \(steps) forwards at "
                + distilled.sigmas.dropLast().map { String(format: "%.3f", $0) }
                    .joined(separator: ", "))
        }

        // One cache per conditioning stream. Under CFG the two forwards see
        // different conditioning, so their block-0 residuals are not comparable
        // and a shared cache would reuse across that gap.
        // The reference DMD loop has no cross-step cache, and a distilled
        // ladder gives it nothing to work with anyway: consecutive DMD steps
        // differ by ~0.64 against the 0.1 reuse threshold, so the probe never
        // fires. Disabled outright rather than left inert -- on a 4-step
        // ladder one silent reuse would skip a quarter of all denoising.
        let cache = distilled == nil && request.cacheThreshold > 0
            ? H3StepCache(threshold: request.cacheThreshold,
                          maxConsecutiveSkips: RenderRequest.cacheMaxSkips,
                          branch: .conditional)
            : nil
        let negativeCache = (distilled == nil && request.cacheThreshold > 0 && request.cfgScale > 1.0)
            ? H3StepCache(threshold: request.cacheThreshold,
                          maxConsecutiveSkips: RenderRequest.cacheMaxSkips,
                          branch: .unconditional)
            : nil

        if let cache {
            // Announced on every run, not assumed. An approximation that is on
            // by default is one people forget is running when they compare two
            // renders.
            log(String(format: "  cross-step cache: ON at threshold %.3f — an APPROXIMATION. "
                       + "Measured at 0.10: 1.9x faster for 16%% less high-frequency detail. "
                       + "Set the threshold to 0 for a faithful render.", cache.threshold))
            log("    probe \(cache.perStreamProbe ? "per-stream" : "whole-sequence"), at most "
                + "\(cache.maxConsecutiveSkips) consecutive reuses, first and last steps "
                + "always full")
        } else {
            log("  cross-step cache: off — every step runs the full stack")
        }
        defer {
            cache?.release()
            negativeCache?.release()
        }

        let renderState = try model.prepareRender(context: conditioning.context,
                                              geometry: geometry,
                                              keyframes: conditions.keyframes,
                                              refs: conditions.refs)
        let negativeRenderState = try conditioning.negative.map {
            try model.prepareRender(context: $0, geometry: geometry,
                                keyframes: conditions.keyframes, refs: conditions.refs)
        }

        var currentVideo = noiseVideo
        var currentAudio = noiseAudio
        let videoSampler = Sampler()
        let audioSampler = Sampler()

        var stepSeconds: [Double] = []
        for i in 0 ..< steps {
            let stepBegan = Date()
            if cancellation?.isCancelled == true {
                throw RenderCancelled(phase: .sampling,
                                      detail: "after \(i) of \(steps) step(s)")
            }
            let sigma = Float(sigmas[i])
            let sigmaNext = Float(sigmas[i + 1])
            let prevSigma = i > 0 ? Float(sigmas[i - 1]) : nil

            var taps = H3Transformer.Taps()
            let (videoVelocity, audioVelocity) = try model.guidedVelocity(
                videoLatent: currentVideo, audioLatent: currentAudio,
                context: conditioning.context, negative: conditioning.negative,
                scale: Float(request.cfgScale), sigmaVideo: Double(sigma),
                geometry: geometry, textTags: conditioning.tags,
                negativeTextTags: conditioning.negativeTags,
                keyframes: conditions.keyframes, refs: conditions.refs,
                condVideo: conditions.videoRows, condAudio: conditions.audioRows,
                renderState: renderState, negativeRenderState: negativeRenderState,
                stepCache: cache, negativeStepCache: negativeCache,
                stepIndex: i, stepCount: steps, taps: &taps,
                // The reference scheduler consumes the RAW per-stream output;
                // the slope scaling exists for the ODE path, which integrates
                // audio in video-sigma space. Scaling in bf16 and dividing it
                // back out here was a multiply-round-divide the reference
                // never performs -- ~0.4% multiplicative noise on every audio
                // velocity, absent from every pipeline whose audio is clean.
                scaleAudioVelocity: distilled == nil)

            let videoDenoised = currentVideo - videoVelocity * MLXArray(sigma)
            let audioDenoised = currentAudio - audioVelocity * MLXArray(sigma)

            if distilled != nil {
                // **A distilled checkpoint is not integrated, it is iterated.**
                // DMD2's multi-step inference alternates denoising with noise
                // *injection*, following consistency models: each step predicts
                // x0 from scratch and is re-noised to the next level with fresh
                // Gaussian noise. It is not a trajectory through an ODE.
                //
                // Running these four timesteps through `res_multistep` — a
                // second-order integrator that also carries state between steps
                // — produces a coherent-looking render that is incoherent to
                // watch, which is how this was found.
                //
                // For flow matching, `x_sigma = (1 - sigma) * x0 + sigma * eps`,
                // so re-noising to `sigmaNext` is that identity applied to the
                // prediction.
                // **The audio stream is not at the video's sigma.** It runs on
                // shift 3 where video runs on 12, and the model returns its
                // velocity already scaled by `d(sigma_a)/d(sigma_v)` so that
                // integrating in video-sigma space is correct.
                //
                // The ODE path is immune to both facts: it only uses `denoised`
                // to recover `(x - denoised)/sigma`, and the scaling cancels.
                // Re-noising does not cancel it. Using the video sigma and the
                // scaled velocity here treats a quantity that is not x0 as if
                // it were, and re-noises it to the wrong level -- which comes
                // out as garbled audio over a perfectly good picture.
                let sigmaAudio = SigmaMap.shift(sigma, from: Float(H3Shift.video),
                                                to: Float(H3Shift.audio))
                let audioX0 = currentAudio - audioVelocity * MLXArray(sigmaAudio)

                // **The reference DMD step is deterministic, and it carries the
                // noise.** FastVideo's published mode (`dmd_stochastic_renoise`
                // defaults false) advances each stream with its own scheduler:
                //
                //     x0   = x + sigma * raw          (per-stream sigma, raw output)
                //     next = (s_next/s) * x + (1 - s_next/s) * x0
                //
                // which is algebraically `(1 - s_next) * x0 + s_next * eps` with
                // the SAME eps the latent already carries -- DDIM-style, not a
                // re-noise. Three faithful-looking alternatives all produced
                // robotic audio: fresh noise per stream (upstream's optional
                // branch), fresh noise at the video sigma, and velocity Euler in
                // video-sigma space -- the last overshoots the audio ladder by
                // 9% on the first rung and 150% on the final one, because the
                // slope at a rung's start is not the secant across a giant step.
                // Video tolerates all of them; audio is the stream that hears
                // the difference. At sigmaNext == 0 the ratio is 0 and the step
                // reduces to x0 exactly, so the last rung needs no branch.
                let ratioVideo = sigmaNext / sigma
                currentVideo = MLXArray(ratioVideo) * currentVideo
                    + MLXArray(1 - ratioVideo) * videoDenoised
                let sigmaAudioNext = SigmaMap.shift(sigmaNext, from: Float(H3Shift.video),
                                                    to: Float(H3Shift.audio))
                let ratioAudio = sigmaAudio > 0 ? sigmaAudioNext / sigmaAudio : 0
                currentAudio = MLXArray(ratioAudio) * currentAudio
                    + MLXArray(1 - ratioAudio) * audioX0
            } else {
                currentVideo = videoSampler.step(x: currentVideo, denoised: videoDenoised,
                                                 sigma: sigma, sigmaNext: sigmaNext,
                                                 prevSigma: prevSigma)
                currentAudio = audioSampler.step(x: currentAudio, denoised: audioDenoised,
                                                 sigma: sigma, sigmaNext: sigmaNext,
                                                 prevSigma: prevSigma)
            }
            // The clock stops **after** this eval, and that placement is the
            // whole reason the timings mean anything. MLX is lazy: the sampler
            // step above builds a graph and returns, so a clock stopped before
            // the eval measures graph construction and hands the arithmetic to
            // whichever later line happens to force it. Under a cache that is
            // not a small error — a skipped step builds almost no graph, so the
            // configuration doing less work would appear to have moved its cost
            // somewhere else rather than saved it.
            eval(currentVideo, currentAudio)
            stepSeconds.append(Date().timeIntervalSince(stepBegan))
            onStep(i + 1, steps)
        }

        // Both branches' traces, kept as separate rows rather than merged. Under
        // CFG the two forwards hold separate caches and can disagree about
        // every step; a merged view would average that away.
        var trace = SamplingTrace(steps: (cache?.trace.steps ?? []) + (negativeCache?.trace.steps ?? []),
                                  stepSeconds: stepSeconds)
        trace.steps.sort { ($0.step, $0.branch.rawValue) < ($1.step, $1.branch.rawValue) }

        return Output(video: currentVideo, audio: currentAudio,
                      cacheSummary: cache?.summary, trace: trace)
    }
}
