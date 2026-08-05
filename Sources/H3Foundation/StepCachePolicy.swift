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
public struct StepCachePolicy: Sendable, Equatable {

    public let threshold: Double
    public let maxConsecutiveSkips: Int
    public let warmupSteps: Int
    public let cooldownSteps: Int
    /// When false, the whole-sequence change alone decides — what every other
    /// published cache for this model does. Kept so the two can be compared
    /// under identical conditions rather than argued about.
    public let perStream: Bool

    public init(threshold: Double, maxConsecutiveSkips: Int = 3,
                warmupSteps: Int = 1, cooldownSteps: Int = 1,
                perStream: Bool = true) {
        self.threshold = threshold
        self.maxConsecutiveSkips = maxConsecutiveSkips
        self.warmupSteps = warmupSteps
        self.cooldownSteps = cooldownSteps
        self.perStream = perStream
    }

    public enum Decision: Sendable, Equatable {
        case runFull
        case reuse
    }

    /// - Parameters:
    ///   - wholeSequenceChange: relative L1 change over every row.
    ///   - audioChange: the same over the target-audio rows, or nil when the
    ///     layout has none.
    ///   - consecutiveSkips: how many reuses immediately preceded this step.
    ///   - haveCachedResidual: false before the first full step has been recorded.
    public func decide(wholeSequenceChange: Double, audioChange: Double?,
                       step: Int, totalSteps: Int,
                       consecutiveSkips: Int, haveCachedResidual: Bool) -> Decision {
        guard haveCachedResidual else { return .runFull }
        guard step >= warmupSteps else { return .runFull }
        guard step < totalSteps - cooldownSteps else { return .runFull }
        guard consecutiveSkips < maxConsecutiveSkips else { return .runFull }

        let effective: Double
        if perStream, let audioChange {
            effective = Swift.max(wholeSequenceChange, audioChange)
        } else {
            effective = wholeSequenceChange
        }
        guard effective.isFinite else { return .runFull }
        return effective < threshold ? .reuse : .runFull
    }
}
