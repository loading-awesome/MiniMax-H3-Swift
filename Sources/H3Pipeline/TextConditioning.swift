import Foundation
import MLX
import H3Foundation
import H3Modules

/// The conditioning the DiT is given, and the modality tag of every text row.
/// Not `Sendable`: MLXArray is not, and these live entirely within one render
/// on one thread.
struct TextConditioning {
    let context: MLXArray
    let tags: [Int]
    let negative: MLXArray?
    let negativeTags: [Int]?
    let tokenCount: Int
}

/// Phase 1: prompt and presented media become conditioning.
///
/// **This phase holds the 51.5 GB text encoder and, when references are
/// present, the vision tower on top of it — and it must be finished and
/// released before the DiT is loaded.** That ordering is a memory contract, not
/// a style choice: hold both and a 275 GB machine is carrying 118 GB of mapped
/// checkpoint before a single activation is allocated.
enum ConditioningEncoder {

    /// - Parameter referenceVideoFrames: decoded once by the caller and used
    ///   twice — Qwen sees a 2 fps subsample of these exact frames and the video
    ///   VAE encodes all of them. Decoding per consumer would let the two halves
    ///   drift apart while every shape stayed valid.
    static func encode(request: RenderRequest,
                       textEncoder textEncoderURL: URL,
                       tokenizer tokenizerURL: URL,
                       referenceVideoFrames: [MLXArray],
                       log: (String) -> Void = { _ in }) throws -> TextConditioning {

        let encoder = try TextEncoder(url: textEncoderURL)
        let tokenizer = try Qwen2Tokenizer(directory: tokenizerURL)

        // Keyframes and reference images are presented to the conditioning
        // encoder as `<Picture N>` **as well as** being VAE-encoded into the
        // packed sequence. The reference does both from the same image, so
        // dropping this half leaves the DiT conditioned on a prompt that never
        // mentions the picture it is being asked to start from.
        let presented = [request.firstFrame, request.lastFrame].compactMap { $0 }
            + request.referenceImages
        let hasReferences = !presented.isEmpty || !request.referenceVideos.isEmpty
            || !request.referenceAudio.isEmpty

        let positive: (cond: MLXArray, count: Int, tags: [Int])
        if !hasReferences {
            let e = encoder.encode(request.prompt, tokenizer: tokenizer)
            positive = (e.cond, e.ids.count, e.tags)
        } else {
            positive = try presentedConditioning(
                request: request, presented: presented,
                referenceVideoFrames: referenceVideoFrames,
                textEncoderURL: textEncoderURL, encoder: encoder,
                tokenizer: tokenizer, log: log)
        }
        log("  prompt -> \(positive.count) tokens")

        var negative: MLXArray?
        var negativeTags: [Int]?
        if let text = request.negativePrompt {
            let n = encoder.encode(text, tokenizer: tokenizer)
            negative = n.cond
            negativeTags = n.tags
            log("  negative -> \(n.ids.count) tokens")
            // Past roughly 30% the negative starves the sampler of variation.
            // Raising the guidance scale is the lever; lengthening the negative
            // prompt is not.
            let ratio = Double(n.ids.count) / Double(max(positive.count, 1))
            if ratio > 0.5 {
                log(String(format: "warning: the negative prompt is %.0f%% of the positive by "
                           + "tokens. Past roughly 30%% it starves the sampler of variation; "
                           + "raise the guidance scale instead.", ratio * 100))
            }
        }
        eval(positive.cond)

        return TextConditioning(context: positive.cond, tags: positive.tags,
                                negative: negative, negativeTags: negativeTags,
                                tokenCount: positive.count)
    }

    /// The reference-carrying path, where **order is the contract**.
    ///
    /// Images, then videos with each paired soundtrack's `<Audio j>` immediately
    /// before its `<Video k>`, then standalone audio. Ordinals are 1-based per
    /// type, so an image, an audio and a video are `<Picture 1>`, `<Audio 1>`,
    /// `<Video 1>` — not 1, 2, 3. Every one of those mistakes keeps all
    /// downstream shapes correct while sliding the content, which is why the
    /// ref2va check gates `embeds` and `position_ids` before it gates
    /// `text_cond`.
    private static func presentedConditioning(
        request: RenderRequest, presented: [URL],
        referenceVideoFrames: [MLXArray], textEncoderURL: URL,
        encoder: TextEncoder, tokenizer: Qwen2Tokenizer,
        log: (String) -> Void
    ) throws -> (cond: MLXArray, count: Int, tags: [Int]) {

        log("Presenting \(presented.count) image(s), \(request.referenceVideos.count) "
            + "video(s), \(request.referenceAudio.count) audio clip(s) to the encoder...")
        let tower = try VisionTower(weights: try MLX.loadArrays(url: textEncoderURL))

        /// One vision block from pixels already on the grid.
        func block(_ pixels: MLXArray, framePair: Bool) throws
            -> (H3Presentation.VisionBlock, VisionGrid) {
            let (gw, gh, grid) = VisionPreprocess.grid(width: pixels.dim(2),
                                                       height: pixels.dim(1))
            guard gw == pixels.dim(2) && gh == pixels.dim(1) else {
                throw H3Error.mediaOffCanvas(
                    path: "presented media", size: "\(pixels.dim(2))x\(pixels.dim(1))",
                    remedy: "the tower's own rounding would resize this to \(gw)x\(gh), and "
                          + "that resize is not ported. Supply pixels on a multiple of 32.")
            }
            let patches = framePair
                ? VisionPreprocess.patches(framePair: pixels, grid: grid)
                : VisionPreprocess.patches(image: pixels, grid: grid)
            let out = tower(patches: patches, grid: grid)
            return (H3Presentation.VisionBlock(merged: out.merged, deepstack: out.deepstack),
                    grid)
        }

        var items: [H3Presentation.RefItem] = []
        for url in presented {
            let size = try MediaLoad.imageSize(at: url.path)
            let (tw, th, _) = VisionPreprocess.grid(width: size.width, height: size.height)
            let (b, grid) = try block(
                try MediaLoad.imageHWC(at: url.path, width: tw, height: th), framePair: false)
            items.append(.image(b, grid))
            log("    \(url.lastPathComponent): \(tw)x\(th) -> \(b.merged.dim(0)) vision tokens")
        }

        for (i, frames) in referenceVideoFrames.enumerated() {
            // The soundtrack's label goes in FIRST. It is a label and nothing
            // else — audio never reaches Qwen — so dropping it because "there is
            // no embedding" shifts every position after it while leaving every
            // shape valid.
            if request.soundtrack(for: i) != nil { items.append(.audio) }

            // Qwen sees the clip at 2 fps, in frame pairs, each pair stamped
            // with the MIDPOINT of its two frames to exactly one decimal. The
            // text is tokenised, so "<0.2 seconds>" and "<0.25 seconds>" are
            // different lengths.
            let step = H3Video.fps / 2
            var idx = Array(stride(from: 0, to: frames.dim(0), by: step))
            var stamps = (0 ..< idx.count).map { Double($0) / 2.0 }
            if idx.count % 2 == 1 {
                idx.append(idx.last!)                       // repeat-pad
                stamps.append(stamps.last!)
            }
            var pairs: [(block: H3Presentation.VisionBlock, grid: VisionGrid)] = []
            for p in stride(from: 0, to: idx.count, by: 2) {
                let pair = concatenated([frames[idx[p]].expandedDimensions(axis: 0),
                                         frames[idx[p + 1]].expandedDimensions(axis: 0)],
                                        axis: 0)
                let (b, grid) = try block(pair, framePair: true)
                pairs.append((b, grid))
                eval(b.merged)
            }
            items.append(.video(pairs: pairs, timestamps: stamps))
            log("    \(request.referenceVideos[i].lastPathComponent): \(frames.dim(0)) frames "
                + "-> \(pairs.count) frame pair(s)")
        }

        for url in request.referenceAudio {
            items.append(.audio)
            log("    \(url.lastPathComponent): <Audio> label only — audio never enters Qwen")
        }

        let a = H3Presentation.assemble(prompt: request.prompt, items: items,
                                        tokenizer: tokenizer, encoder: encoder)
        var taps = TextEncoder.Taps()
        let cond = encoder(embeds: a.embeds, computeDType: .float32,
                           positionIds: a.positionIds,
                           visualSpans: a.spans.map { (start: $0.start, count: $0.size) },
                           deepstack: a.deepstack, taps: &taps)
        return (cond, a.embeds.dim(1), a.tags)
    }
}
