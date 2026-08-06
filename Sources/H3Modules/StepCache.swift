// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

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
/// **Never skip too many in a row without a ceiling — and be clear about what
/// the ceiling protects.** This used to claim a skipped step leaves the probe
/// unrefreshed so the same comparison recurs. It does not: block 0 runs on
/// every step, skipped or not, and `previousProbe` is updated unconditionally.
/// The comparison is fresh every time.
///
/// What ages is the **cached total residual**, which only a full step
/// refreshes. So `maxConsecutiveSkips` bounds how many steps of trajectory one
/// delta gets applied across — residual age — and that, not a recurring
/// comparison, is the quantity to reason about when output starts warping.
/// The distinction matters because the two suggest different fixes.
///
/// **Never skip the first or last steps.** The first has no history. The last
/// determines the output that is actually decoded, and a reused delta there
/// lands directly in the pixels.
package final class H3StepCache {

    /// Relative L1 change in block 0's residual below which the rest of the
    /// stack is skipped.
    ///
    /// Higher is faster and less faithful. There is no published value for this
    /// model — it has to be swept and judged on rendered output, which for H3
    /// means judging the **audio as well as the video**: the two streams come
    /// out of one forward pass and there is no reason for them to degrade at the
    /// same rate.
    package let threshold: Double

    /// Upper bound on consecutive reuses — equivalently, the maximum age in
    /// steps of the residual being re-applied.
    ///
    /// **Raising it does not simply buy that many more skipped steps.** The
    /// counter resets at every refusal, so changing the cap moves the refresh
    /// points rather than removing them. Replayed against the measured deltas,
    /// caps 3 and 4 both yield nine reuses and identical wall clock; the
    /// refreshes merely relocate from steps 7/11/15 to 8/13. See
    /// `StepCachePolicyReplayTests`.
    ///
    /// **It counts steps, and a step is not a fixed amount of trajectory.**
    /// This matters the moment anyone renders at other than the default 20
    /// steps, and cuts in two directions at once:
    ///
    ///  * A finer schedule moves the latent less per step, so *n* steps of
    ///    residual age covers less trajectory. In that sense a fixed cap gets
    ///    **more** conservative as step count rises, not less.
    ///  * But smaller per-step deltas also fall below the threshold more
    ///    often, so more steps qualify for reuse and the cap becomes the
    ///    binding constraint far more of the time. The **fraction** of the
    ///    schedule that gets skipped rises.
    ///
    /// Which effect dominates is not something to reason out from here — the
    /// deltas at 40 steps have to be measured and replayed, exactly as the
    /// 20-step ones were. Recorded because the intuition "more steps means the
    /// cache coasts further" is only half right, and the half that is wrong
    /// points the other way.
    package let maxConsecutiveSkips: Int

    /// Steps at the start and end of the schedule that always run in full.
    package let warmupSteps: Int
    package let cooldownSteps: Int

    private var previousProbe: MLXArray?
    private var cachedTotalResidual: MLXArray?
    private var consecutiveSkips = 0

    package private(set) var stepsRun = 0
    package private(set) var stepsSkipped = 0

    /// Every step's measurement and decision, in order.
    ///
    /// **Three deltas are recorded and only two are consulted.** The decision
    /// still gates on `max(wholeSequence, audio)`, exactly as before; the
    /// target-video rows are measured alongside and used by nothing. That is
    /// deliberate — an instrument that changes the thing it measures is worth
    /// nothing as a control, so the video figure is gathered first and given a
    /// vote only once there is a baseline to judge the change against.
    package private(set) var trace = SamplingTrace()

    /// When false, the probe is the mean over the **whole packed sequence** —
    /// what every published cache for this model does. Kept as an option
    /// purely so the two can be compared under identical conditions; a claim
    /// that splitting the probe matters is worth nothing without the run that
    /// does not split it.
    package let perStreamProbe: Bool

    /// Which CFG forward this cache serves. Held here rather than passed per
    /// call because a cache instance belongs to exactly one branch for its whole
    /// life — that is the invariant the type exists to enforce — and a
    /// per-call argument would be one more place for the two to be crossed.
    package let branch: StepTrace.Branch

    package init(threshold: Double, maxConsecutiveSkips: Int = 3,
                warmupSteps: Int = 1, cooldownSteps: Int = 1,
                perStreamProbe: Bool = true,
                branch: StepTrace.Branch = .conditional) {
        self.branch = branch
        self.perStreamProbe = perStreamProbe
        self.threshold = threshold
        self.maxConsecutiveSkips = maxConsecutiveSkips
        self.warmupSteps = warmupSteps
        self.cooldownSteps = cooldownSteps
    }

    /// What the block loop should do this step.
    package enum Decision {
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
    ///   - videoRange: rows carrying the target video. Measured and recorded,
    ///     not yet voted on — see `trace`.
    ///   - step: 0-based sampler step.
    ///   - totalSteps: how many steps this render has.
    ///   - sigma: the video sigma being integrated, for the trace.
    ///   - branch: which CFG forward this is.
    package func decide(probe: MLXArray, audioRange: Range<Int>?,
                       videoRange: Range<Int>? = nil,
                       step: Int, totalSteps: Int,
                       sigma: Double = .nan) -> Decision {
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
        var wholeSequenceChange = Double.infinity
        var audioChange = Double.infinity
        var videoChange = Double.infinity
        if let prev = previousProbe, prev.shape == probe.shape {
            wholeSequenceChange = relativeChange(probe, prev)
            func slice(_ r: Range<Int>?) -> Double {
                guard let r, !r.isEmpty, r.upperBound <= probe.dim(0) else { return .infinity }
                return relativeChange(probe[r], prev[r])
            }
            audioChange = slice(audioRange)
            videoChange = slice(videoRange)
        }

        // Every rule lives in H3Foundation.StepCachePolicy, which has no MLX
        // dependency and is therefore tested in microseconds without a GPU.
        // This method's job is the measurement; the policy's job is the choice.
        let policy = StepCachePolicy(threshold: threshold,
                                     maxConsecutiveSkips: maxConsecutiveSkips,
                                     warmupSteps: warmupSteps,
                                     cooldownSteps: cooldownSteps,
                                     perStream: perStreamProbe)
        let cached = cachedTotalResidual
        let verdict = policy.explain(
            wholeSequenceChange: wholeSequenceChange,
            // Only the per-stream arm hands the audio a vote; the whole-sequence
            // arm has to be genuinely blind to it for the A/B to mean anything.
            audioChange: perStreamProbe ? audioChange : nil,
            // Advisory in every arm — video is measured and does not vote.
            videoChange: videoChange,
            step: step, totalSteps: totalSteps,
            consecutiveSkips: consecutiveSkips,
            haveCachedResidual: cached != nil && cached?.shape == probe.shape)

        trace.steps.append(StepTrace(
            step: step, branch: self.branch, sigma: sigma,
            wholeSequenceChange: wholeSequenceChange, videoChange: videoChange,
            audioChange: audioChange, decision: verdict.decision, reason: verdict.reason,
            constraints: verdict.constraints, advisory: verdict.advisory,
            consecutiveSkipsBefore: consecutiveSkips))

        if verdict.decision == .reuse, let cached {
            consecutiveSkips += 1
            stepsSkipped += 1
            return .reuse(cached)
        }
        consecutiveSkips = 0
        stepsRun += 1
        return .runFull
    }

    /// Records the full stack's total residual after a `runFull` step.
    package func record(totalResidual: MLXArray) {
        cachedTotalResidual = totalResidual
    }

    /// Frees the cached tensors. The residual is the size of the packed hidden
    /// state — at 20k tokens and hidden 5376 that is ~215 MB in bf16, held for
    /// the whole render and doubled under CFG.
    package func release() {
        previousProbe = nil
        cachedTotalResidual = nil
    }

    package var summary: String {
        trace.summary(threshold: threshold, perStreamProbe: perStreamProbe)
    }
}
