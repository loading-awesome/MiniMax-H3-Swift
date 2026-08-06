// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Whether to reuse the previous step's residual, decided from two scalars.
///
/// **Deliberately free of MLX.** The measurement of how much block 0's residual
/// moved needs tensors; deciding what to do about it does not. Separating them
/// puts every rule that can fail silently — the warm-up, the cool-down, the
/// consecutive cap, which stream gets to veto — in a target that tests in
/// microseconds on any machine, with no GPU and no checkpoint.
///
/// The rules themselves are each a way the cache goes wrong in a way no shape
/// check would catch:
///
/// **Never reuse on the first step.** There is no history to compare against.
///
/// **Never reuse on the last step.** Its residual lands directly in the decoded
/// pixels, with no later step to correct it.
///
/// **Bound consecutive reuse.** A skipped step does not update the probe, so
/// the same comparison recurs and the sampler can coast to the end on one stale
/// delta. Unbounded reuse is not a slow degradation; it is a cliff.
///
/// **Gate on the worse stream, not the average.** At 864x480x124 the packed
/// sequence is 95.1% video and 2.6% audio, and the audio residual was measured
/// moving 32% more per step than the whole-sequence average. A probe averaged
/// over everything is a video-only decision wearing a disguise.
package struct StepCachePolicy: Sendable, Equatable {

    package let threshold: Double
    package let maxConsecutiveSkips: Int
    package let warmupSteps: Int
    package let cooldownSteps: Int
    /// When false, the whole-sequence change alone decides — what every other
    /// published cache for this model does. Kept so the two can be compared
    /// under identical conditions rather than argued about.
    package let perStream: Bool

    package init(threshold: Double, maxConsecutiveSkips: Int = 3,
                warmupSteps: Int = 1, cooldownSteps: Int = 1,
                perStream: Bool = true) {
        self.threshold = threshold
        self.maxConsecutiveSkips = maxConsecutiveSkips
        self.warmupSteps = warmupSteps
        self.cooldownSteps = cooldownSteps
        self.perStream = perStream
    }

    package enum Decision: Sendable, Equatable {
        case runFull
        case reuse
    }

    /// Why a step ran in full.
    ///
    /// A cache that reports "14 of 20 steps skipped" has told you almost
    /// nothing about *why* the other six ran. Six steps refused because the
    /// deltas were genuinely large is a cache working; six refused because the
    /// consecutive cap kept firing is a threshold set too loose with a safety
    /// rail absorbing the consequences — and the two look identical in the
    /// skip count. Sweeping a threshold against a number that cannot tell them
    /// apart is how you tune into the rail and call it a result.
    package enum Reason: String, Sendable, Codable, Equatable, CaseIterable {
        /// No full step has been recorded yet, so there is nothing to reuse.
        case noHistory
        /// Inside the opening steps that always run in full.
        case warmup
        /// Inside the closing steps that always run in full.
        case cooldown
        /// The consecutive-reuse ceiling fired.
        case consecutiveCap
        /// The measured change was infinite or NaN.
        case nonFinite
        /// The whole-sequence change was the binding constraint.
        case videoAboveThreshold
        /// The audio rows were the binding constraint — the veto that only a
        /// per-stream probe can cast.
        case audioAboveThreshold
        /// Both streams were quiet. The only reason that yields `.reuse`.
        case belowThreshold
    }

    package struct Verdict: Sendable, Equatable {
        package let decision: Decision
        package let reason: Reason
        /// The value actually compared against `threshold`, after the
        /// per-stream maximum. Infinite when no comparison was possible.
        package let effectiveChange: Double
    }

    /// The decision and the reason for it. ``decide(wholeSequenceChange:audioChange:step:totalSteps:consecutiveSkips:haveCachedResidual:)``
    /// is this with the reason discarded, so the two can never disagree.
    package func explain(wholeSequenceChange: Double, audioChange: Double?,
                        step: Int, totalSteps: Int,
                        consecutiveSkips: Int, haveCachedResidual: Bool) -> Verdict {
        func verdict(_ d: Decision, _ r: Reason, _ change: Double = .infinity) -> Verdict {
            Verdict(decision: d, reason: r, effectiveChange: change)
        }
        guard haveCachedResidual else { return verdict(.runFull, .noHistory) }
        guard step >= warmupSteps else { return verdict(.runFull, .warmup) }
        guard step < totalSteps - cooldownSteps else { return verdict(.runFull, .cooldown) }
        guard consecutiveSkips < maxConsecutiveSkips else {
            return verdict(.runFull, .consecutiveCap)
        }

        let effective: Double
        if perStream, let audioChange {
            effective = Swift.max(wholeSequenceChange, audioChange)
        } else {
            effective = wholeSequenceChange
        }
        guard effective.isFinite else { return verdict(.runFull, .nonFinite) }
        guard effective < threshold else {
            // Which stream bound it. Under `perStream` the audio can be the
            // larger of the two, and that distinction is the entire argument
            // for splitting the probe — recording it makes the argument
            // checkable against a sweep rather than a claim.
            let audioBound = perStream && (audioChange ?? -.infinity) >= wholeSequenceChange
            return verdict(.runFull, audioBound ? .audioAboveThreshold : .videoAboveThreshold,
                           effective)
        }
        return verdict(.reuse, .belowThreshold, effective)
    }

    /// - Parameters:
    ///   - wholeSequenceChange: relative L1 change over every row.
    ///   - audioChange: the same over the target-audio rows, or nil when the
    ///     layout has none.
    ///   - consecutiveSkips: how many reuses immediately preceded this step.
    ///   - haveCachedResidual: false before the first full step has been recorded.
    package func decide(wholeSequenceChange: Double, audioChange: Double?,
                       step: Int, totalSteps: Int,
                       consecutiveSkips: Int, haveCachedResidual: Bool) -> Decision {
        explain(wholeSequenceChange: wholeSequenceChange, audioChange: audioChange,
                step: step, totalSteps: totalSteps, consecutiveSkips: consecutiveSkips,
                haveCachedResidual: haveCachedResidual).decision
    }
}
