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
/// **Bound consecutive reuse — because the cached residual ages, not because
/// the probe does.** This said the opposite until it was checked against the
/// code: block 0 runs on every step including a skipped one, and
/// `previousProbe` is updated unconditionally, so the comparison is against a
/// fresh probe every time. What goes stale is the **total stack residual**,
/// which is only refreshed by a full step. The cap therefore bounds *residual
/// age*: how many steps of trajectory the reused delta is being applied
/// across. That is the quantity to reason about when a render starts warping,
/// and it is not the same quantity the threshold measures.
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
    package enum Reason: String, Sendable, Equatable, CaseIterable {
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
        /// The whole-sequence change was above threshold.
        ///
        /// Spelled `videoAboveThreshold` in records written before 2026-08-06,
        /// which was a misnomer: this probe measures every packed row, and
        /// video is 95.1% of them but not all of them. The legacy spelling
        /// still decodes.
        case wholeSequenceAboveThreshold
        /// The audio rows were above threshold — the veto only a per-stream
        /// probe can cast.
        ///
        /// **Its presence does not mean audio was decisive.** At steps 16-18
        /// of the measured control both probes are above threshold and the
        /// whole-sequence one would have forced a full step on its own; audio
        /// appears in the headline merely because it is the larger number. The
        /// step where audio is genuinely the only objection is 15 — whole
        /// 0.087, audio 0.108 — and under a cap of 3 that is invisible,
        /// because the cap fires first. Read `constraints`, not `reason`, to
        /// tell these apart.
        case audioAboveThreshold
        /// The target-video rows were above threshold.
        ///
        /// **Advisory only** — video does not vote. Recorded because it
        /// disagrees with the whole-sequence probe exactly once in the measured
        /// control, at step 4: whole 0.096 admits reuse, video 0.101 would
        /// refuse it. That single crossing is the entire empirical case for
        /// ever giving this probe a vote, and it costs a step rather than
        /// saving one.
        ///
        /// Spelled `videoRowsAboveThreshold` on the wire, **not**
        /// `videoAboveThreshold`, which is taken: that string means the
        /// whole-sequence probe in every record written before the rename. A
        /// shared spelling would have made this case decode as a different one,
        /// so it could be written but never read back.
        case videoAboveThreshold = "videoRowsAboveThreshold"
        /// Both streams were quiet. The only reason that yields `.reuse`.
        case belowThreshold
    }

    package struct Verdict: Sendable, Equatable {
        package let decision: Decision
        /// The headline — the first constraint in priority order. Convenient,
        /// and not sufficient: see `constraints`.
        package let reason: Reason
        /// **Every constraint that independently forced a full step**, not
        /// just the one that got reported.
        ///
        /// The headline alone is actively misleading for tuning. Under a cap
        /// of 3 the measured control refuses step 15 with `consecutiveCap`,
        /// and the audio probe — which also objected, and objected alone —
        /// never appears. Relax the cap and audio becomes the thing standing
        /// between the cache and the late high-change region. A sweep reading
        /// only the headline would conclude the audio probe was doing nothing
        /// and remove it, immediately before the change that makes it
        /// load-bearing.
        package let constraints: [Reason]
        /// Constraints that would have forced a full step had that probe been
        /// given a vote. Currently only the video rows.
        package let advisory: [Reason]
        /// The value actually compared against `threshold`, after the
        /// per-stream maximum. Infinite when no comparison was possible.
        package let effectiveChange: Double
    }

    /// The decision and the reason for it. ``decide(wholeSequenceChange:audioChange:step:totalSteps:consecutiveSkips:haveCachedResidual:)``
    /// is this with the reason discarded, so the two can never disagree.
    /// - Parameter videoChange: the target-video rows. **Advisory** — it does
    ///   not vote, it is recorded so that the case for giving it one can be
    ///   made from data rather than from the fact that it exists.
    package func explain(wholeSequenceChange: Double, audioChange: Double?,
                        videoChange: Double? = nil,
                        step: Int, totalSteps: Int,
                        consecutiveSkips: Int, haveCachedResidual: Bool) -> Verdict {

        // Every constraint is evaluated, not short-circuited. Short-circuiting
        // is why a sweep could not see that the audio probe was the sole
        // objection at step 15 — the cap was checked first and returned.
        var constraints: [Reason] = []
        if !haveCachedResidual { constraints.append(.noHistory) }
        if step < warmupSteps { constraints.append(.warmup) }
        if step >= totalSteps - cooldownSteps { constraints.append(.cooldown) }
        if consecutiveSkips >= maxConsecutiveSkips { constraints.append(.consecutiveCap) }

        let effective = (perStream && audioChange != nil)
            ? Swift.max(wholeSequenceChange, audioChange!)
            : wholeSequenceChange
        if !effective.isFinite {
            constraints.append(.nonFinite)
        } else {
            if wholeSequenceChange >= threshold {
                constraints.append(.wholeSequenceAboveThreshold)
            }
            if perStream, let a = audioChange, a.isFinite, a >= threshold {
                constraints.append(.audioAboveThreshold)
            }
        }

        // Advisory: what a probe that does not vote would have said.
        var advisory: [Reason] = []
        if let v = videoChange, v.isFinite, v >= threshold {
            advisory.append(.videoAboveThreshold)
        }

        guard constraints.isEmpty else {
            // The headline keeps its old meaning — the largest of the two
            // thresholded streams when a threshold is what bound it, otherwise
            // the first structural rule in priority order.
            let structural = constraints.first {
                $0 == .noHistory || $0 == .warmup || $0 == .cooldown
                    || $0 == .consecutiveCap || $0 == .nonFinite
            }
            let headline: Reason
            if let structural {
                headline = structural
            } else if constraints.contains(.audioAboveThreshold),
                      (audioChange ?? -.infinity) >= wholeSequenceChange {
                headline = .audioAboveThreshold
            } else {
                headline = .wholeSequenceAboveThreshold
            }
            return Verdict(decision: .runFull, reason: headline, constraints: constraints,
                           advisory: advisory, effectiveChange: effective)
        }
        return Verdict(decision: .reuse, reason: .belowThreshold, constraints: [],
                       advisory: advisory, effectiveChange: effective)
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

extension StepCachePolicy.Reason: Codable {
    /// Hand-written so the records already on disk keep decoding.
    ///
    /// `videoAboveThreshold` meant the whole-sequence probe until 2026-08-06,
    /// and now means the target-video rows. Those are different quantities, so
    /// the legacy spelling maps to `wholeSequenceAboveThreshold` — which is
    /// what those records actually measured. Encoding is unambiguous; only the
    /// read side has to know about the rename.
    ///
    /// The mapping is resolved in favour of the archive because the only
    /// records carrying the old spelling are the eight controls in docs/bench,
    /// and they all predate the split. A record written after it uses the new
    /// spelling for the whole-sequence probe, so there is no collision.
    package init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        if raw == "videoAboveThreshold" {
            self = .wholeSequenceAboveThreshold
            return
        }
        guard let value = Self(rawValue: raw) else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "unknown reason \(raw)"))
        }
        self = value
    }

    package func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(rawValue)
    }
}
