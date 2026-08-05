import Foundation

/// Every way this library can refuse, as a value the caller can switch on.
///
/// **The reason this type exists is that the experimental tree crashed
/// instead.** `PackedLayout` called `preconditionFailure` on a keyframe anchor
/// that was neither the first frame nor the last; `VisionPreprocess` called it
/// on an off-grid image. Both are reachable from a caller passing ordinary
/// wrong input, and in a CLI they print a stack trace, which is survivable. In
/// a library linked into somebody's application they take the whole process
/// down. Nothing a caller can provoke may trap.
///
/// Three properties every case here keeps, because they are what made the
/// render policy's refusals useful rather than annoying:
///
///  * **the rule** — what was violated, in one clause;
///  * **the measurement** — the number that establishes it, never a bare
///    "invalid" or "unsupported";
///  * **the remedy** — what to do instead, concretely enough to act on.
///
/// A message that has all three lets somebody fix their call without reading
/// the source. A message with only the first sends them to the source.
public enum H3Error: Error, CustomStringConvertible, Sendable {

    // MARK: geometry and request shape

    /// A frame count that does not lie on the 17k+5 lattice, or lies outside
    /// the trained range.
    case frameCount(requested: Int, aligned: Int, trained: ClosedRange<Int>)

    /// A keyframe anchor at an index the reference has no `cond_t` for.
    case keyframeIndex(index: Int, frameCount: Int)

    /// A pixel dimension that is not on the VAE's 16x grid times the DiT's 2x
    /// patch.
    case dimensionOffGrid(width: Int, height: Int, multiple: Int)

    /// Anchors and references in one payload. The reference raises on this and
    /// so do we.
    case conflictingConditioning(String)

    /// A count limit from the reference node: 9 images, 3 videos, 3 audio.
    case tooManyReferences(kind: String, got: Int, limit: Int)

    // MARK: checkpoints

    /// A checkpoint whose vendor could not be established from `__metadata__`.
    ///
    /// This is a refusal and not a guess on purpose: the two published bf16
    /// conversions store `attn.qkv_proj.weight` in different layouts with the
    /// same shape, dtype, mean and standard deviation, and loading one as the
    /// other yields cos 0.029 — uncorrelated output that still looks like a
    /// subtle numerical bug. See FRAGILE_CONTRACTS.md #9.
    case unidentifiedCheckpoint(url: URL, detail: String)

    /// A checkpoint that is the wrong partition for the requested mode.
    case wrongPartition(needed: String, loaded: String, mode: String)

    /// A checkpoint the configuration names but which is not on disk.
    case checkpointMissing(role: String, path: String)

    /// A checkpoint whose weights are an approximation of the released ones,
    /// used without `allowApproximateWeights`.
    case approximateWeightsNotPermitted(variant: String, detail: String)

    // MARK: files and media

    case unreadable(path: String, reason: String)
    case noTrack(path: String, kind: String)

    /// Media handed in at a size that would need the unported LANCZOS resize.
    case mediaOffCanvas(path: String, size: String, remedy: String)

    // MARK: capacity

    /// The machine cannot hold this configuration.
    case insufficientMemory(needGB: Double, availableGB: Double, detail: String)

    /// A feature that exists in the reference and is not implemented here.
    case notImplemented(feature: String, detail: String)

    /// A render the policy rejects, carrying every reason at once so the caller
    /// fixes all of them rather than discovering them one run at a time.
    case policyViolations([PolicyViolation])

    public var description: String {
        switch self {
        case let .frameCount(requested, aligned, trained):
            return "\(requested) frames aligns to \(aligned), outside the trained range "
                 + "\(trained.lowerBound)-\(trained.upperBound). Frame counts snap up onto a "
                 + "17k+5 lattice, so 4 s becomes 107 — the only value in the whole 4-15 s "
                 + "range that lands under the floor. Ask for 5 s or more."
        case let .keyframeIndex(index, frameCount):
            return "keyframe anchor at frame \(index) is neither the first (0) nor the last "
                 + "(\(frameCount - 1)). The reference defines cond_t for those two positions "
                 + "only; a middle index has no defined value and would pin the anchor to the "
                 + "wrong instant with every shape still correct."
        case let .dimensionOffGrid(width, height, multiple):
            return "\(width)x\(height) is not a multiple of \(multiple) on both axes. The VAE "
                 + "downsamples by 16 and the DiT patchifies by 2, so an off-grid size produces "
                 + "a latent the packed layout cannot describe."
        case let .conflictingConditioning(detail):
            return "conflicting conditioning: \(detail)"
        case let .tooManyReferences(kind, got, limit):
            return "\(got) \(kind) references; the reference node accepts at most \(limit)."
        case let .unidentifiedCheckpoint(url, detail):
            return "cannot identify \(url.lastPathComponent): \(detail). Refusing rather than "
                 + "guessing — the published bf16 conversions store fused attention weights in "
                 + "different layouts with identical shape, dtype and statistics, and loading "
                 + "one as the other gives cos 0.029 against the reference."
        case let .wrongPartition(needed, loaded, mode):
            return "\(mode) needs the \(needed) partition; the loaded checkpoint is \(loaded). "
                 + "The two partitions share an architecture and differ in weights, so this "
                 + "would render without error and without the training the mode relies on."
        case let .checkpointMissing(role, path):
            return "no \(role) checkpoint at \(path). Run `h3 doctor` to see every path the "
                 + "configuration resolves to and which are missing."
        case let .approximateWeightsNotPermitted(variant, detail):
            return "\(variant) is an approximation of the released weights (\(detail)). Set "
                 + "policy.allow_approximate_weights = true to use it, and do not compare its "
                 + "output to the gated configuration."
        case let .unreadable(path, reason):
            return "cannot read \(path): \(reason)"
        case let .noTrack(path, kind):
            return "\(path) has no \(kind) track"
        case let .mediaOffCanvas(path, size, remedy):
            return "\(path) is \(size), which is off the canvas grid. \(remedy)"
        case let .insufficientMemory(need, available, detail):
            return String(format: "needs about %.0f GB, this machine has %.0f GB available. %@",
                          need, available, detail)
        case let .notImplemented(feature, detail):
            return "\(feature) is not implemented: \(detail)"
        case let .policyViolations(vs):
            return "the render policy rejects this configuration:\n"
                 + vs.map { "  - \($0.rule): \($0.reason)\n    remedy: \($0.remedy)" }
                     .joined(separator: "\n")
        }
    }
}

/// One reason a configuration is worse than what has been verified.
///
/// Carried in a list rather than thrown one at a time so a caller fixing a bad
/// request sees every problem in one pass.
public struct PolicyViolation: Sendable, Equatable {
    public let rule: String
    public let reason: String
    public let remedy: String
    public init(rule: String, reason: String, remedy: String) {
        self.rule = rule
        self.reason = reason
        self.remedy = remedy
    }
}
