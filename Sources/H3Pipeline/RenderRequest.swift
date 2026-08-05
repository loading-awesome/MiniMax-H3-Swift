import Foundation
import H3Foundation
import H3Catalog
import H3Recipes

/// Everything a render needs, as a value the caller can build, inspect, store
/// and validate without loading a single byte of checkpoint.
///
/// **This is deliberately not the CLI's argument struct.** In the experimental
/// tree the request *was* the command: `Generate` carried 40 `@Option`s, a
/// `validate()` bound to ArgumentParser's error type, and a 550-line `run()`
/// that did everything from tokenising to muxing. Nothing about a render could
/// be tested, reused from another program, or serialised, because all of it was
/// welded to a process invocation.
///
/// Here the request is a plain value. `validate()` throws `H3Error`, so the
/// same refusal reaches a CLI user as text and a library caller as a value they
/// can switch on.
public struct RenderRequest: Sendable {

    /// Named, auditable numerical behavior. The default is the configuration
    /// whose implementation parity was measured; faster profiles are explicit
    /// approximations and are recorded in every render receipt.
    public enum QualityProfile: String, Codable, Sendable, CaseIterable {
        case faithful
        case balanced
        case fast
        case custom

        public var cacheThreshold: Double {
            switch self {
            case .faithful: 0
            case .balanced: 0.10
            case .fast: 0.15
            case .custom: 0
            }
        }

        public var isApproximate: Bool {
            self == .balanced || self == .fast
        }
    }

    // MARK: what to render

    public var prompt: String
    /// Negation belongs here rather than in the prompt: it is applied through
    /// guidance, not through the text, so it does nothing at `cfgScale` 1.
    public var negativePrompt: String?
    public var seconds: Int
    public var steps: Int
    public var seed: UInt64

    // MARK: shape

    /// Explicit pixel dimensions, which override `recipe` and `aspectRatio`.
    ///
    /// The tiers cannot express 864x480, and that is the shape every golden and
    /// every measured tolerance uses — without this there is no way to render at
    /// the verified shape at all.
    public var width: Int?
    public var height: Int?
    public var recipe: H3RecipeID?
    public var aspectRatio: AspectRatio
    public var resolution: ResolutionTier

    // MARK: conditioning

    public var firstFrame: URL?
    public var lastFrame: URL?
    public var referenceImages: [URL]
    public var referenceVideos: [URL]
    /// Index-matched to `referenceVideos`; nil leaves that video silent.
    ///
    /// The pairing is not cosmetic. A paired soundtrack's `<Audio j>` label is
    /// emitted immediately *before* its `<Video k>`, so the audio ordinal can
    /// precede the video ordinal, and the DiT block becomes `video_audio` with
    /// the audio rows ahead of the video rows inside one block. Standalone audio
    /// does neither.
    public var referenceVideoSoundtracks: [URL?]
    public var referenceAudio: [URL]

    // MARK: numerics

    public var cfgScale: Double
    public var qualityProfile: QualityProfile
    /// Cross-step residual reuse. 0 disables it and renders faithfully.
    ///
    /// 0.10 is the measured knee, swept at 864x480x124x20 against an uncached
    /// control of the same prompt and seed: 1.93x faster for 16% less
    /// high-frequency detail. 0.15 buys 2.60x and costs 28%; 0.25 buys 2.93x and
    /// costs 44%. Faithful output is the default; selecting this optimization
    /// is an explicit quality decision.
    public var cacheThreshold: Double
    public var cacheMaxSkips: Int
    /// Probe the whole packed sequence rather than per stream — what every other
    /// published cache for this model does, kept so the two can be compared.
    public var cacheWholeSequenceProbe: Bool
    /// Recorded conditioning noise from `emit_cond_noise.py`. Without it the
    /// rows are augmented from MLX's PRNG, which cannot reproduce the
    /// reference's `torch.Generator("cpu")` bytes, and the render falls outside
    /// the parity contract.
    public var conditioningNoise: URL?
    public var attentionBackend: String

    // MARK: policy

    /// Run a configuration the policy rejects, having been told why.
    ///
    /// These refusals were warnings once. The warning was printed, ignored by
    /// its own author, and cost an evening chasing a decoder bug that did not
    /// exist.
    public var allowSuboptimal: Bool

    /// Existing final artifacts are protected unless replacement is explicit.
    public var overwriteOutput: Bool

    // MARK: output

    public var videoOutput: URL
    public var audioOutput: URL?

    public init(prompt: String,
                videoOutput: URL,
                audioOutput: URL? = nil,
                negativePrompt: String? = nil,
                seconds: Int = 5,
                steps: Int = 20,
                seed: UInt64 = 0,
                width: Int? = nil,
                height: Int? = nil,
                recipe: H3RecipeID? = nil,
                aspectRatio: AspectRatio = .r16x9,
                resolution: ResolutionTier = .p768,
                firstFrame: URL? = nil,
                lastFrame: URL? = nil,
                referenceImages: [URL] = [],
                referenceVideos: [URL] = [],
                referenceVideoSoundtracks: [URL?] = [],
                referenceAudio: [URL] = [],
                cfgScale: Double = 1.0,
                qualityProfile: QualityProfile = .faithful,
                cacheThreshold: Double? = nil,
                cacheMaxSkips: Int = 3,
                cacheWholeSequenceProbe: Bool = false,
                conditioningNoise: URL? = nil,
                attentionBackend: String = "auto",
                allowSuboptimal: Bool = false,
                overwriteOutput: Bool = false) {
        self.prompt = prompt
        self.videoOutput = videoOutput
        self.audioOutput = audioOutput
        self.negativePrompt = negativePrompt
        self.seconds = seconds
        self.steps = steps
        self.seed = seed
        self.width = width
        self.height = height
        self.recipe = recipe
        self.aspectRatio = aspectRatio
        self.resolution = resolution
        self.firstFrame = firstFrame
        self.lastFrame = lastFrame
        self.referenceImages = referenceImages
        self.referenceVideos = referenceVideos
        self.referenceVideoSoundtracks = referenceVideoSoundtracks
        self.referenceAudio = referenceAudio
        self.cfgScale = cfgScale
        self.qualityProfile = cacheThreshold == nil ? qualityProfile : .custom
        self.cacheThreshold = cacheThreshold ?? qualityProfile.cacheThreshold
        self.cacheMaxSkips = cacheMaxSkips
        self.cacheWholeSequenceProbe = cacheWholeSequenceProbe
        self.conditioningNoise = conditioningNoise
        self.attentionBackend = attentionBackend
        self.allowSuboptimal = allowSuboptimal
        self.overwriteOutput = overwriteOutput
    }

    // MARK: derived

    public enum ResolutionTier: String, Sendable, CaseIterable {
        /// The model's own low-latency tier, and the default.
        case p768 = "768p"
        /// **Not a base render size.** 2K is the output of In-Context
        /// Regeneration, which takes a base render and upscales it — a second
        /// stage that is neither released nor ported. Sampling 2560x1440
        /// directly is a shape the DiT was never trained on, at 133,200 packed
        /// tokens: an estimated 14.6 hours for 20 steps at this machine's
        /// measured throughput.
        case k2 = "2k"
    }

    public enum AspectRatio: String, Sendable, CaseIterable {
        case r16x9 = "16:9", r9x16 = "9:16", r21x9 = "21:9"
        case r4x3 = "4:3", r3x4 = "3:4", r1x1 = "1:1"
    }

    /// Which mode this request is, and therefore which DiT partition it needs.
    ///
    /// **Derived, never a flag the caller can get wrong.** The two partitions
    /// share an architecture and differ in weights, so the wrong one renders
    /// without error and without the training the mode relies on — which is
    /// exactly what happened to every reference render on 2026-08-05, because
    /// the runner hard-coded one path. `CheckpointIdentity.validate(forMode:)`
    /// refuses the mismatch at load time, using this.
    public var mode: RenderMode {
        if !referenceImages.isEmpty || !referenceVideos.isEmpty || !referenceAudio.isEmpty {
            return .reference
        }
        if firstFrame != nil || lastFrame != nil { return .firstLastFrame }
        return .textToVideo
    }

    public var modeDescription: String {
        if firstFrame != nil || lastFrame != nil { return "first/last-frame interpolation" }
        if !referenceVideos.isEmpty { return "video reference" }
        if !referenceAudio.isEmpty { return "audio reference" }
        if !referenceImages.isEmpty { return "image reference" }
        return "text to video and audio"
    }

    public var usesApproximateSampling: Bool { cacheThreshold > 0 }

    /// Pixel dimensions, from whichever source was given.
    public func dimensions() throws -> (width: Int, height: Int) {
        if let id = recipe {
            guard let rec = H3RecipeRegistry.all[id] else {
                throw H3Error.invalidRequest(
                    rule: "unknown recipe", detail: "no recipe registered as '\(id.rawValue)'",
                    remedy: "run `h3 recipes` for the list, with a status column.")
            }
            return (rec.targetWidth, rec.targetHeight)
        }
        if let w = width, let h = height { return (w, h) }
        let is2k = resolution == .k2
        switch aspectRatio {
        case .r16x9: return is2k ? (2560, 1440) : (1344, 768)
        case .r9x16: return is2k ? (1440, 2560) : (768, 1344)
        case .r21x9: return is2k ? (3360, 1440) : (1792, 768)
        case .r4x3:  return is2k ? (1920, 1440) : (1024, 768)
        case .r3x4:  return is2k ? (1440, 1920) : (768, 1024)
        case .r1x1:  let e = is2k ? 1440 : 768; return (e, e)
        }
    }

    /// Soundtracks in **block order**: each video's paired soundtrack at that
    /// video's own position, then the standalone clips.
    ///
    /// The DiT walks the conditioning audio as one flat stream in segment order,
    /// and a `video_audio` block puts its audio rows *ahead* of its video rows
    /// inside the block. So a paired soundtrack is consumed before any
    /// standalone clip, whatever order the caller supplied them in.
    package var orderedAudio: [URL] {
        (0 ..< referenceVideos.count).compactMap { soundtrack(for: $0) } + referenceAudio
    }

    package func soundtrack(for videoIndex: Int) -> URL? {
        guard videoIndex < referenceVideoSoundtracks.count else { return nil }
        return referenceVideoSoundtracks[videoIndex]
    }

    // MARK: validation

    /// Every problem with this request, thrown as one error rather than one per
    /// run. Cheap — no file is opened and no model is loaded.
    public func validate() throws {
        if (width == nil) != (height == nil) {
            throw H3Error.invalidRequest(
                rule: "incomplete dimensions", detail: "width and height come as a pair",
                remedy: "give both, or neither and use a recipe or an aspect ratio.")
        }
        if let w = width, let h = height {
            guard w % 32 == 0, h % 32 == 0 else {
                throw H3Error.dimensionOffGrid(width: w, height: h, multiple: 32)
            }
            if let id = recipe, let rec = H3RecipeRegistry.all[id],
               (w, h) != (rec.targetWidth, rec.targetHeight) {
                throw H3Error.invalidRequest(
                    rule: "conflicting shape",
                    detail: "recipe \(id.rawValue) is \(rec.targetWidth)x\(rec.targetHeight); "
                          + "width and height ask for \(w)x\(h)",
                    remedy: "give one or the other.")
            }
        }
        if resolution == .k2 && !allowSuboptimal {
            throw H3Error.notImplemented(
                feature: "2K as a base render size",
                detail: "2K is produced by In-Context Regeneration from a base render; the "
                      + "DiT does not sample it directly, and that upscaler is neither "
                      + "released nor ported. Sampling 2560x1440 directly is 133,200 packed "
                      + "tokens — an estimated 14.6 hours for 20 steps. Render at 768p.")
        }
        guard (4 ... 15).contains(seconds) else {
            throw H3Error.invalidRequest(
                rule: "duration out of range", detail: "\(seconds) s; the model covers 4-15 s",
                remedy: "ask for 5 s or more — 4 s snaps to 107 frames, under the trained floor.")
        }
        guard prompt.count <= 7000 else {
            throw H3Error.invalidRequest(
                rule: "prompt too long", detail: "\(prompt.count) characters against a 7,000 limit",
                remedy: "shorten it; the encoder truncates rather than refusing, which would "
                      + "silently drop the end of your description.")
        }
        guard steps > 0 else {
            throw H3Error.invalidRequest(rule: "no steps", detail: "steps must be positive",
                                         remedy: "20 is the gated value; 50 is the practical ceiling.")
        }
        guard referenceImages.count <= 9 else {
            throw H3Error.tooManyReferences(kind: "image", got: referenceImages.count, limit: 9)
        }
        guard referenceVideos.count <= 3 else {
            throw H3Error.tooManyReferences(kind: "video", got: referenceVideos.count, limit: 3)
        }
        guard referenceAudio.count <= 3 else {
            throw H3Error.tooManyReferences(kind: "audio", got: referenceAudio.count, limit: 3)
        }

        let hasAnchors = firstFrame != nil || lastFrame != nil
        let hasReferences = !referenceImages.isEmpty || !referenceVideos.isEmpty
            || !referenceAudio.isEmpty || !referenceVideoSoundtracks.isEmpty
        if hasAnchors && hasReferences {
            throw H3Error.conflictingConditioning(
                "frame anchors and general references cannot share one payload. They are "
                + "served by different DiT partitions — anchors by FL2VA, references by "
                + "Ref2VA — so there is no checkpoint that could honour both.")
        }

        // A soundtrack belongs to a video by index. More soundtracks than videos
        // would silently drop the tail, and each dropped one moves every later
        // `<Audio j>` ordinal.
        guard referenceVideoSoundtracks.count <= referenceVideos.count else {
            throw H3Error.invalidRequest(
                rule: "unmatched soundtracks",
                detail: "\(referenceVideoSoundtracks.count) soundtrack(s) for "
                      + "\(referenceVideos.count) reference video(s)",
                remedy: "soundtracks are index-matched to videos; pass nil for a silent one.")
        }

        if negativePrompt != nil && cfgScale <= 1.0 {
            throw H3Error.invalidRequest(
                rule: "negation without guidance",
                detail: "a negative prompt does nothing at a CFG scale of \(cfgScale)",
                remedy: "negation is applied through guidance, not through the prompt text. "
                      + "Raise cfgScale above 1, or drop the negative prompt.")
        }
        guard cacheThreshold >= 0 else {
            throw H3Error.invalidRequest(
                rule: "negative cache threshold", detail: "\(cacheThreshold)",
                remedy: "0 disables the cache; 0.10 is the measured knee.")
        }
    }

    /// The policy's verdict on this request's cost and shape.
    ///
    /// Separate from `validate()` because it is advisory when `allowSuboptimal`
    /// is set, and because it needs the geometry.
    package func policyViolations(geometry: LatentGeometry) -> [PolicyViolation] {
        let (w, h) = (try? dimensions()) ?? (0, 0)
        return H3RenderPolicy.check(width: w, height: h, frameCount: geometry.frameCount,
                                    steps: steps,
                                    tokens: geometry.videoTokens + geometry.audioTokens)
            .map { PolicyViolation(rule: $0.rule, reason: $0.reason, remedy: $0.remedy) }
    }
}
