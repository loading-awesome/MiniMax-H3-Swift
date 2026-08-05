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

        MLXRandom.seed(request.seed)
        let noiseVideo = MLXRandom.normal([1, 24, geometry.latentT,
                                           geometry.latentH, geometry.latentW])
        let noiseAudio = MLXRandom.normal([1, 32, 2, geometry.audioT])

        let schedule = FlowSchedule(shift: H3Shift.video)
        let sigmas = schedule.sigmas(steps: request.steps)

        // One cache per conditioning stream. Under CFG the two forwards see
        // different conditioning, so their block-0 residuals are not comparable
        // and a shared cache would reuse across that gap.
        let cache = request.cacheThreshold > 0
            ? H3StepCache(threshold: request.cacheThreshold,
                          maxConsecutiveSkips: request.cacheMaxSkips,
                          perStreamProbe: !request.cacheWholeSequenceProbe)
            : nil
        let negativeCache = (request.cacheThreshold > 0 && request.cfgScale > 1.0)
            ? H3StepCache(threshold: request.cacheThreshold,
                          maxConsecutiveSkips: request.cacheMaxSkips,
                          perStreamProbe: !request.cacheWholeSequenceProbe)
            : nil

        if let cache {
            // Announced on every run, not assumed. An approximation that is on
            // by default is one people forget is running when they compare two
            // renders.
            log(String(format: "  cross-step cache: ON at threshold %.3f — an APPROXIMATION. "
                       + "Measured at 0.10: 1.93x faster for 16%% less high-frequency detail. "
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

        for i in 0 ..< request.steps {
            if cancellation?.isCancelled == true {
                throw RenderCancelled(phase: .sampling,
                                      detail: "after \(i) of \(request.steps) step(s)")
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
                stepIndex: i, stepCount: request.steps, taps: &taps)

            let videoDenoised = currentVideo - videoVelocity * MLXArray(sigma)
            let audioDenoised = currentAudio - audioVelocity * MLXArray(sigma)

            currentVideo = videoSampler.step(x: currentVideo, denoised: videoDenoised,
                                             sigma: sigma, sigmaNext: sigmaNext,
                                             prevSigma: prevSigma)
            currentAudio = audioSampler.step(x: currentAudio, denoised: audioDenoised,
                                             sigma: sigma, sigmaNext: sigmaNext,
                                             prevSigma: prevSigma)
            eval(currentVideo, currentAudio)
            onStep(i + 1, request.steps)
        }

        return Output(video: currentVideo, audio: currentAudio, cacheSummary: cache?.summary)
    }
}
