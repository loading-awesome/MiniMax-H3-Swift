import Foundation
import MLX
import H3Foundation

/// Cross-step residual reuse — "first block cache".
///
/// The single biggest acceleration available for this model, and it is not a
/// kernel. NVIDIA's own H3 breakdown attributes the bulk of their 3.95x to it
/// (1.534x -> 3.95x of the total), with fused AdaLN second and sparse attention
/// a further 1.25x. It costs one extra tensor subtraction per step.
///
/// ## The idea
///
/// A diffusion step's job is to produce a *change* to the latent, and across
/// adjacent steps on a back-loaded schedule that change is often nearly the same
/// change. Block 0 is a cheap probe for it: if block 0's residual this step
/// closely matches block 0's residual last step, the other 49 blocks are very
/// likely to produce nearly the same total residual they produced last step —
/// so reuse it and skip them.
///
///     h1 = block0(h)                     always run
///     r  = h1 - h                        the probe
///     if relative_change(r, r_prev) < threshold:
///         h_out = h + total_residual_prev     49 blocks skipped
///     else:
///         h_out = blocks[1...](h1)
///         total_residual_prev = h_out - h
///
/// **What is cached is the total residual, not the output.** Caching the output
/// would pin the render to a stale latent; caching the change lets the reused
/// delta apply to wherever the trajectory has actually got to.
///
/// ## Three ways to get this wrong
///
/// **One cache per conditioning stream.** CFG runs two forwards per step with
/// different conditioning, and their residuals are not comparable. Sharing one
/// cache between them compares a conditional residual against an unconditional
/// one and reuses across the gap — the numbers stay finite and the render goes
/// subtly wrong. This type is deliberately not shared: the sampler holds one per
/// branch.
///
/// **Never skip twice in a row without a ceiling.** Reuse is self-reinforcing:
/// a skipped step does not update the probe, so the same comparison recurs and
/// the sampler can coast to the end on one stale delta. `maxConsecutiveSkips`
/// bounds it.
///
/// **Never skip the first or last steps.** The first has no history. The last
/// determines the output that is actually decoded, and a reused delta there
/// lands directly in the pixels.
public final class H3StepCache {

    /// Relative L1 change in block 0's residual below which the rest of the
    /// stack is skipped.
    ///
    /// Higher is faster and less faithful. There is no published value for this
    /// model — it has to be swept and judged on rendered output, which for H3
    /// means judging the **audio as well as the video**: the two streams come
    /// out of one forward pass and there is no reason for them to degrade at the
    /// same rate.
    public let threshold: Double

    /// Upper bound on consecutive reuses.
    public let maxConsecutiveSkips: Int

    /// Steps at the start and end of the schedule that always run in full.
    public let warmupSteps: Int
    public let cooldownSteps: Int

    private var previousProbe: MLXArray?
    private var cachedTotalResidual: MLXArray?
    private var consecutiveSkips = 0

    public private(set) var stepsRun = 0
    public private(set) var stepsSkipped = 0
    /// The measured relative change at each step, for tuning the threshold
    /// against something other than intuition.
    public private(set) var observedChange: [Double] = []
    /// The audio stream's own relative change, recorded separately so a sweep
    /// can see whether the two streams actually move together.
    public private(set) var observedAudioChange: [Double] = []
    /// The whole-sequence change — what every other cache implementation
    /// measures. Recorded so the two can be compared directly rather than
    /// argued about.
    public private(set) var observedVideoChange: [Double] = []

    /// When false, the probe is the mean over the **whole packed sequence** —
    /// what every published cache for this model does. Kept as an option
    /// purely so the two can be compared under identical conditions; a claim
    /// that splitting the probe matters is worth nothing without the run that
    /// does not split it.
    public let perStreamProbe: Bool

    public init(threshold: Double, maxConsecutiveSkips: Int = 3,
                warmupSteps: Int = 1, cooldownSteps: Int = 1,
                perStreamProbe: Bool = true) {
        self.perStreamProbe = perStreamProbe
        self.threshold = threshold
        self.maxConsecutiveSkips = maxConsecutiveSkips
        self.warmupSteps = warmupSteps
        self.cooldownSteps = cooldownSteps
    }

    /// What the block loop should do this step.
    public enum Decision {
        /// Run every remaining block, then call `record(totalResidual:)`.
        case runFull
        /// Skip them; add this to the stack's input instead.
        case reuse(MLXArray)
    }

    /// Decides after block 0 has run.
    ///
    /// - Parameters:
    ///   - probe: block 0's residual, `h_after_block0 - h_in`.
    ///   - audioRange: rows of the packed sequence carrying the target audio.
    ///   - step: 0-based sampler step.
    ///   - totalSteps: how many steps this render has.
    public func decide(probe: MLXArray, audioRange: Range<Int>?,
                       step: Int, totalSteps: Int) -> Decision {
        defer { previousProbe = probe }

        /// Relative L1 change over a slice. Mean absolute rather than RMS
        /// because the probe is dominated by a few large rows, and RMS would let
        /// a big change in a small region hide behind a quiet majority.
        func relativeChange(_ a: MLXArray, _ b: MLXArray) -> Double {
            let denom = MLX.mean(MLX.abs(b)).item(Float.self)
            guard denom > 0 else { return .infinity }
            return Double(MLX.mean(MLX.abs(a - b)).item(Float.self) / denom)
        }

        // **Measured per stream, and gated on the worse of the two.**
        //
        // This is the whole reason to write another cache rather than use one.
        // At 864x480x124 the packed sequence is 95.1% video rows and **2.6%
        // audio rows**, so a probe averaged over the whole sequence is a
        // video-only probe wearing a disguise: a change that destroys the
        // soundtrack moves it by almost nothing, and the step gets skipped.
        //
        // That is not a hypothetical. Every published cache for this model
        // degrades audio — EasyCache, Spectrum and sparse-attention step
        // skipping alike — and the tool author's summary is "all cache methods
        // currently warp audio". NVIDIA hit the same thing from the other
        // direction and record a configuration where "the picture scored best
        // of its set while its dialogue fell apart".
        //
        // Splitting the probe costs one extra mean over 414 rows and makes the
        // audio stream's needs visible to the decision. Whether it is *enough*
        // is a measurement, not a claim — `speech_check.py` gives WER on the
        // rendered waveform, which is the only honest way to settle it.
        var change = Double.infinity
        var audioChange = Double.infinity
        if let prev = previousProbe, prev.shape == probe.shape {
            if perStreamProbe, let audioRange, audioRange.upperBound <= probe.dim(0),
               !audioRange.isEmpty {
                let videoLike = relativeChange(probe, prev)
                audioChange = relativeChange(probe[audioRange], prev[audioRange])
                observedVideoChange.append(videoLike)
                change = Swift.max(videoLike, audioChange)
            } else {
                change = relativeChange(probe, prev)
            }
        }
        observedChange.append(change)
        observedAudioChange.append(audioChange)

        let inWarmup = step < warmupSteps
        let inCooldown = step >= totalSteps - cooldownSteps
        let capped = consecutiveSkips >= maxConsecutiveSkips

        if !inWarmup, !inCooldown, !capped, change < threshold,
           let cached = cachedTotalResidual, cached.shape == probe.shape {
            consecutiveSkips += 1
            stepsSkipped += 1
            return .reuse(cached)
        }
        consecutiveSkips = 0
        stepsRun += 1
        return .runFull
    }

    /// Records the full stack's total residual after a `runFull` step.
    public func record(totalResidual: MLXArray) {
        cachedTotalResidual = totalResidual
    }

    /// Frees the cached tensors. The residual is the size of the packed hidden
    /// state — at 20k tokens and hidden 5376 that is ~215 MB in bf16, held for
    /// the whole render and doubled under CFG.
    public func release() {
        previousProbe = nil
        cachedTotalResidual = nil
    }

    public var summary: String {
        let total = stepsRun + stepsSkipped
        guard total > 0 else { return "step cache: unused" }
        let pct = 100.0 * Double(stepsSkipped) / Double(total)
        func median(_ xs: [Double]) -> Double {
            let f = xs.filter { $0.isFinite }.sorted()
            return f.isEmpty ? 0 : f[f.count / 2]
        }
        return String(format: "step cache [%@]: %d/%d steps skipped (%.0f%%), threshold %.3f, "
                      + "median change %.3f = max(whole-sequence %.3f, audio-only %.3f)",
                      (perStreamProbe ? "per-stream" : "whole-sequence") as NSString,
                      stepsSkipped, total, pct, threshold, median(observedChange),
                      median(observedVideoChange), median(observedAudioChange))
    }
}
