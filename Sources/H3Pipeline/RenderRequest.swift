// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

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

    /// Named, auditable numerical behavior.
    ///
    /// **The default is `balanced`, and it is an approximation.** That is a
    /// deliberate reversal. It was `faithful` on the argument that a default
    /// should be the configuration whose parity was measured — which is a good
    /// argument, and it lost to the measurement: the cross-step cache is 1.79x
    /// on this machine for 16% less fine detail, and nobody rendering a
    /// twenty-minute clip wants to discover afterwards that a flag would have
    /// made it eleven.
    ///
    /// What made the reversal safe is that the honesty machinery is separate
    /// from the default. `isApproximate` is true here, so the banner says so on
    /// every run and the render receipt records it. Somebody comparing two
    /// renders months later can still see which was which — which was the real
    /// concern, and it never depended on `faithful` being the default.
    ///
    /// `--quality faithful` is one flag away and remains exact.
    /// **There was a `fast` profile at threshold 0.15 and it has been removed.**
    ///
    /// It was added in a commit about packaging and a README, it was referenced
    /// nowhere but its own threshold table, it had no tests, and nobody ever
    /// watched a render made with it. The threshold sweep in
    /// `docs/ACCELERATION.md` puts 0.15 at **28% less high-frequency detail**
    /// against `balanced`'s 16% — so it was a named, recommended-looking option
    /// that degraded output by nearly twice the default, on no evidence.
    ///
    /// Raising `cacheMaxSkips` from 3 to 5 made that worse without anyone
    /// noticing: at the new cap, 0.15 skips more steps than the 13 of 20 the
    /// sweep measured, so its quality had drifted further from the figure being
    /// published for it.
    ///
    /// Nothing is lost. `--cache-threshold 0.15` still does exactly the same
    /// thing, and reports the profile as `custom` — which is what an
    /// unvalidated setting should be called.
    public enum QualityProfile: String, Codable, Sendable, CaseIterable {
        case faithful
        case balanced

        public var cacheThreshold: Double {
            switch self {
            case .faithful: 0
            case .balanced: 0.10
            }
        }

        /// Whether this *profile* approximates. **Not the same question as
        /// whether a render does** — `cacheThreshold` can be overridden past
        /// the profile, so the render's truth is
        /// ``RenderRequest/usesApproximateSampling``, which reads the threshold
        /// actually in force. Anything reporting to a user must use that one.
        public var isApproximate: Bool { self == .balanced }
    }

    /// A still image anchored at one frame of the timeline.
    public struct Keyframe: Sendable, Equatable, Codable {
        public var image: URL
        /// A **pixel** frame index, 0-based, into the *aligned* timeline —
        /// ``RenderRequest/alignedFrameCount``, not `seconds * fps`. The two
        /// differ whenever the requested duration is off the 17k+5 lattice,
        /// which is most of the time.
        public var frame: Int

        public init(image: URL, frame: Int) {
            self.image = image
            self.frame = frame
        }

        /// `path@frame`, the CLI's spelling.
        ///
        /// Split at the **last** `@`, so a path may contain one — `@` is legal
        /// in a filename and splitting at the first would truncate the path and
        /// then fail to parse a frame out of the rest, reporting the wrong
        /// problem. It lives here rather than in the command because the command
        /// has no test target, and a parser nobody can test is a parser nobody
        /// has checked.
        public init(spec: String) throws {
            guard let at = spec.lastIndex(of: "@") else {
                throw H3Error.invalidRequest(
                    rule: "malformed keyframe", detail: "'\(spec)' has no @",
                    remedy: "spell an anchor as path@frame, e.g. shot.png@48.")
            }
            let path = String(spec[spec.startIndex ..< at])
            let index = String(spec[spec.index(after: at)...])
            guard !path.isEmpty, let frame = Int(index) else {
                throw H3Error.invalidRequest(
                    rule: "malformed keyframe",
                    detail: path.isEmpty ? "'\(spec)' has no path before its @"
                                         : "'\(index)' is not a frame number",
                    remedy: "spell an anchor as path@frame, e.g. shot.png@48. The frame is a "
                          + "0-based index into the aligned timeline.")
            }
            self.init(image: URL(fileURLWithPath: path), frame: frame)
        }
    }

    // MARK: what to render

    public var prompt: String
    /// Negation belongs here rather than in the prompt: it is applied through
    /// guidance, not through the text, so it does nothing at `cfgScale` 1.
    public var negativePrompt: String?
    public var seconds: Int
    public var steps: Int

    /// A distilled checkpoint's own denoising steps, out of 1000, taken from
    /// its metadata. Nil for the base model, which follows the flow schedule
    /// for `steps`.
    ///
    /// This travels on the request because the checkpoint is the only thing
    /// that knows it: a four-forward distill is trained to jump between
    /// specific noise levels, so its schedule is a property of the weights and
    /// not a number the caller may choose.
    public var distilledSteps: [Int]? = nil
    public var seed: UInt64

    // MARK: shape

    /// Explicit pixel dimensions, which override `recipe` and `aspectRatio`.
    ///
    /// **This used to be the only route to the verified shape.** The tiers
    /// cannot express 864x480 — the shape every golden and every measured
    /// tolerance uses — so before the megapixel ladder there was no way to name
    /// it. `--recipe h3_16x9_0p4mp` is now that name, and this stays for sizes
    /// off the ladder rather than for the one size that most needed it.
    public var width: Int?
    public var height: Int?
    public var recipe: H3RecipeID?
    public var aspectRatio: AspectRatio
    public var resolution: ResolutionTier

    // MARK: conditioning

    public var firstFrame: URL?
    public var lastFrame: URL?
    /// Still images anchored at chosen frames, beyond the two ends.
    ///
    /// **`firstFrame` and `lastFrame` are the same mechanism** — sugar for
    /// anchors at 0 and `alignedFrameCount - 1`. They are kept because they are
    /// the two positions the model was trained with, and because naming the end
    /// is safer than computing it (see `validate()`).
    ///
    /// An anchor away from the ends is positionally exact and behaviourally
    /// untested: fl2va saw conditioning at the ends during training, so a middle
    /// anchor is out of distribution. It pins rows to the right instant; the
    /// model is not promised to land on them.
    public var keyframes: [Keyframe]
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
    /// costs 44%. This is what `balanced` selects, and `balanced` is the
    /// default — turning it *off* is now the explicit decision, and
    /// `--quality faithful` is how.
    public var cacheThreshold: Double
    /// How many steps in a row may reuse the cached residual — equivalently,
    /// the maximum age of the residual being re-applied.
    ///
    /// **5, raised from 3 on 2026-08-06 after a measured sweep.** It buys
    /// 8.4–9.5% end-to-end at the default 20 steps and **24.5% at 40**, where
    /// the finer schedule puts more steps under the threshold and the cap
    /// becomes what governs the render: the skipped fraction goes from 45% to
    /// 68% at cap 3 and 75% at cap 5.
    ///
    /// **Raising it does not buy that many more skipped steps.** The counter
    /// resets at every refusal, so a larger cap relocates the refresh points
    /// rather than deleting them — caps 3 and 4 do identical work, refreshes
    /// merely moving from steps 7/11/15 to 8/13. 5 is where one extra reuse
    /// actually appears. See `StepCachePolicyReplayTests`.
    ///
    /// The reuse structure is the same on a calm beach, fast motion with a
    /// tracking camera, close-up dialogue and a second seed — 9 reuses at cap 3
    /// and 10 at cap 5, refused at the same steps every time. The delta curve
    /// is driven by the flow schedule, not by scene content.
    ///
    /// **Accepted on viewing, not on a metric**, and that is not a shortcut.
    /// Every cap setting changes the sampling trajectory, so every setting
    /// renders a *different scene* rather than a degraded version of the same
    /// one — measured at structural dissimilarity 0.017 to 0.265 between arms.
    /// A detail ratio across that gap compares content. The distribution
    /// fallback does not rescue it either: detail varies 25–28% between seeds,
    /// so resolving a 2% effect would need on the order of a hundred renders.
    /// How many steps in a row may reuse the cached residual.
    ///
    /// **A constant, not a setting.** It was a knob while it was being swept —
    /// 3, 4, 5 and 6 rendered and watched — and 5 is the answer. See
    /// `docs/PERF_ROADMAP.md` §6C.
    public static let cacheMaxSkips = 5
    /// Probe the whole packed sequence rather than per stream — what every other
    /// published cache for this model does, kept so the two can be compared.
    /// The whole-sequence probe lost its A/B against the per-stream one and
    /// the flag that selected it is gone. Audio is the sole objection at one
    /// late step of every render measured, which a whole-sequence probe cannot
    /// see — it is 2.6% of the rows.
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
                keyframes: [Keyframe] = [],
                referenceImages: [URL] = [],
                referenceVideos: [URL] = [],
                referenceVideoSoundtracks: [URL?] = [],
                referenceAudio: [URL] = [],
                cfgScale: Double = 1.0,
                qualityProfile: QualityProfile = .balanced,
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
        self.keyframes = keyframes
        self.referenceImages = referenceImages
        self.referenceVideos = referenceVideos
        self.referenceVideoSoundtracks = referenceVideoSoundtracks
        self.referenceAudio = referenceAudio
        self.cfgScale = cfgScale
        self.qualityProfile = qualityProfile
        self.cacheThreshold = qualityProfile.cacheThreshold
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
        if hasAnchors { return .firstLastFrame }
        return .textToVideo
    }

    public var modeDescription: String {
        if hasAnchors {
            let n = anchorCount
            return n <= 2 && keyframes.isEmpty
                ? "first/last-frame interpolation"
                : "keyframe interpolation, \(n) anchors"
        }
        if !referenceVideos.isEmpty { return "video reference" }
        if !referenceAudio.isEmpty { return "audio reference" }
        if !referenceImages.isEmpty { return "image reference" }
        return "text to video and audio"
    }

    /// Whether this request carries any visual anchor, of either spelling.
    /// Anchors select the fl2va partition, so this is what `mode` turns on.
    public var hasAnchors: Bool {
        firstFrame != nil || lastFrame != nil || !keyframes.isEmpty
    }

    public var anchorCount: Int {
        (firstFrame == nil ? 0 : 1) + (lastFrame == nil ? 0 : 1) + keyframes.count
    }

    /// The frame count the render will actually have.
    ///
    /// **Anchor indices are indices into this, not into `seconds * fps`.** The
    /// requested duration is snapped *up* onto the 17k+5 lattice before anything
    /// downstream sees it, so 5 s at 24 fps is 124 frames and the last one is
    /// 123 — not the 119 the arithmetic a caller is likely to do produces.
    public var alignedFrameCount: Int {
        LatentGeometry.alignFrameCount(seconds * H3Video.fps)
    }

    /// Every visual anchor, in the one order all three consumers must agree on:
    /// **ascending by frame**.
    ///
    /// The packed layout, the VAE encode and the `<Picture N>` presentation each
    /// walk this list positionally and independently. If they disagreed, the
    /// k-th cond segment would carry the k-th image at some other image's
    /// instant — every shape valid, every anchor wrong. So there is one list and
    /// they all call it.
    ///
    /// The sort is stable in the index a caller supplied, which matters only if
    /// two anchors share a frame — and `validate()` refuses that.
    package func resolvedKeyframes(frameCount: Int) -> [Keyframe] {
        var all: [Keyframe] = []
        if let f = firstFrame { all.append(Keyframe(image: f, frame: 0)) }
        if let l = lastFrame { all.append(Keyframe(image: l, frame: frameCount - 1)) }
        all += keyframes
        return all.enumerated()
            .sorted { ($0.element.frame, $0.offset) < ($1.element.frame, $1.offset) }
            .map(\.element)
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
        // The 768p landscape and portrait sizes here are the ladder's 0.98 MP
        // rung, so `--aspect-ratio` and `--recipe` cannot disagree about what
        // 768p means. The other four ratios have no ladder and only this.
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

        try validateAnchors()

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

    /// The anchor half of `validate()`.
    ///
    /// Every rule here guards a mistake that would otherwise render — the packed
    /// layout only refuses an index off the timeline, and everything else it
    /// accepts, correctly, at whatever instant it was given.
    private func validateAnchors() throws {
        let frames = alignedFrameCount
        let last = frames - 1

        // A cap, and it is not arbitrary: one anchor per second across the
        // longest render the model covers, plus the closing one. 15 s is 362
        // frames, so 0, 24, ... 360 is 16. Each anchor costs its rows twice —
        // one cond segment in the packed sequence *and* one `<Picture N>` block
        // in the text span — which is ~26% more sequence at this density and
        // roughly 1.6x the attention work.
        guard anchorCount <= 16 else {
            throw H3Error.tooManyReferences(kind: "keyframe anchor", got: anchorCount, limit: 16)
        }

        for kf in keyframes {
            guard kf.frame >= 0, kf.frame < frames else {
                throw H3Error.keyframeIndex(index: kf.frame, frameCount: frames)
            }
            // The trap the layout used to catch by refusing every middle index.
            // `seconds * fps - 1` is the "last frame" a caller computes, and it
            // is short of the real end whenever the duration was off-lattice —
            // an anchor placed a fraction of a second early, with every shape
            // still correct. It costs one legal index to refuse it here, and the
            // message names both ways to say what was meant.
            if kf.frame == seconds * H3Video.fps - 1 && kf.frame != last {
                throw H3Error.invalidRequest(
                    rule: "anchor on the unaligned end",
                    detail: "frame \(kf.frame) is `seconds * fps - 1`, but this render is "
                          + "\(frames) frames — the duration snapped up onto the 17k+5 lattice, "
                          + "so the last frame is \(last)",
                    remedy: "use frame \(last), or `lastFrame`, which resolves the end for you. "
                          + "If you did mean \(kf.frame), it is \(last - kf.frame) frame(s) "
                          + "early — ask for it as a fraction of \(frames).")
            }
        }

        var seen: [Int: Int] = [:]
        for kf in resolvedKeyframes(frameCount: frames) {
            seen[kf.frame, default: 0] += 1
            guard seen[kf.frame] == 1 else {
                let named = kf.frame == 0 ? " (which is where `firstFrame` sits)"
                    : kf.frame == last ? " (which is where `lastFrame` sits)" : ""
                throw H3Error.invalidRequest(
                    rule: "duplicate anchor",
                    detail: "two anchors claim frame \(kf.frame)\(named)",
                    remedy: "one image per frame; the layout would stack both cond segments at "
                          + "the same instant and the later one would not obviously win.")
            }
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
                                    tokens: geometry.videoTokens + geometry.audioTokens,
                                    distilledSteps: distilledSteps)
            .map { PolicyViolation(rule: $0.rule, reason: $0.reason, remedy: $0.remedy) }
    }
}
