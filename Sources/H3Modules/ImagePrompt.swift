import Foundation
import MLX
import H3Foundation

/// The H3 conditioning **presentation** — how a prompt and its images are laid
/// out before the language stack sees them.
///
/// It is deliberately **not chat-templated**. Raw prompt and label text, no
/// special tokens, with vision blocks spliced in
/// (`comfy/text_encoders/minimax.py`):
///
///     t2va    <prompt>
///     fl2va   "<Picture 1>: " <vision> ["<Picture 2>: " <vision>] <prompt>
///     ref2va  per condition in request order, 1-based ordinals per type
///
/// A vision block is `<|vision_start|>`, the tower's merged embeddings, then
/// `<|vision_end|>`. The labels are ordinary text — `<Picture 1>: ` tokenises
/// like any other string, it is not a special token.
///
/// **fl2va sends a keyframe through both paths.** The same image is VAE-encoded
/// into a `cond` segment of the packed sequence *and* presented here as
/// `<Picture 1>`. They are not alternatives.
package enum H3Presentation {
    package static let visionStart = 151_652
    package static let visionEnd = 151_653

    /// One image already through the vision tower.
    package struct VisionBlock {
        package let merged: MLXArray            // [n, 5120]
        package let deepstack: [MLXArray]       // 3 x [n, 5120]
        package init(merged: MLXArray, deepstack: [MLXArray]) {
            self.merged = merged
            self.deepstack = deepstack
        }
        package var tokens: Int { merged.dim(0) }
    }

    /// Where an image's embeddings sit in the assembled sequence.
    package struct Span: Sendable, Equatable {
        package let start: Int, size: Int
        package var end: Int { start + size }
    }

    package struct Assembled {
        /// `[1, S, hidden]` — token embeddings with vision blocks spliced in.
        package let embeds: MLXArray
        package let spans: [Span]
        /// `[3, S]` t/h/w rows, or nil when there is no image (plain RoPE then).
        package let positionIds: MLXArray?
        /// `[S]` — true at vision-embedding positions only.
        package let visualMask: MLXArray?
        /// Three tensors, each `[totalVisionTokens, hidden]`, concatenated over
        /// images in order.
        package let deepstack: [MLXArray]
        /// `minimax_token_tags`: 1 for text, **0 for the whole vision block**
        /// including the flanking start/end tokens.
        package let tags: [Int]
    }

    /// One `ref2va` reference, already through whatever encoding it needs.
    ///
    /// The three cases are not symmetric, and the asymmetry is the contract:
    ///
    ///  * `image` contributes a label and one vision block.
    ///  * `audio` contributes **a label and nothing else** — audio never enters
    ///    Qwen. Its latents ride in the DiT payload instead. Dropping the label
    ///    because "there is no embedding" shifts every position after it.
    ///  * `video` contributes a label and then one vision block per **frame
    ///    pair**, each preceded by its own `<T.T seconds>` timestamp text.
    package enum RefItem {
        case image(VisionBlock, VisionGrid)
        case audio
        case video(pairs: [(block: VisionBlock, grid: VisionGrid)], timestamps: [Double])
    }

    /// Builds the presentation. `blocks` must be in the same order as the
    /// `<Picture N>` labels, which is request order.
    package static func assemble(prompt: String, blocks: [VisionBlock],
                                tokenizer: Qwen2Tokenizer, encoder: TextEncoder,
                                gridPerImage: [VisionGrid]) -> Assembled {
        precondition(blocks.count == gridPerImage.count,
                     "\(blocks.count) vision block(s) but \(gridPerImage.count) grid(s)")
        return assemble(prompt: prompt,
                        items: zip(blocks, gridPerImage).map { .image($0, $1) },
                        tokenizer: tokenizer, encoder: encoder)
    }

    /// The `ref2va` presentation: references in request order, then the prompt.
    ///
    /// Ordinals are **1-based per type** and counted over the whole item list,
    /// so an image, an audio and a video are `<Picture 1>`, `<Audio 1>` and
    /// `<Video 1>` — not 1, 2, 3. The caller owns the ordering; the node emits
    /// images, then videos (each paired soundtrack's `<Audio j>` immediately
    /// before its `<Video k>`), then standalone audio.
    package static func assemble(prompt: String, items: [RefItem],
                                tokenizer: Qwen2Tokenizer, encoder: TextEncoder) -> Assembled {
        var pieces: [MLXArray] = []
        var spans: [Span] = []
        var grids: [VisionGrid] = []
        var visionBlocks: [VisionBlock] = []
        var cursor = 0

        func appendText(_ s: String) {
            let ids = tokenizer.encode(s)
            guard !ids.isEmpty else { return }
            pieces.append(encoder.embed(ids: ids)[0])
            cursor += ids.count
        }
        func appendIds(_ ids: [Int]) {
            pieces.append(encoder.embed(ids: ids)[0])
            cursor += ids.count
        }
        func appendVision(_ block: VisionBlock, _ grid: VisionGrid) {
            appendIds([visionStart])
            spans.append(Span(start: cursor, size: block.tokens))
            grids.append(grid)
            visionBlocks.append(block)
            pieces.append(block.merged.asType(.float32))
            cursor += block.tokens
            appendIds([visionEnd])
        }

        var nImage = 0, nAudio = 0, nVideo = 0
        for item in items {
            switch item {
            case let .image(block, grid):
                nImage += 1
                appendText("<Picture \(nImage)>: ")
                appendVision(block, grid)
            case .audio:
                nAudio += 1
                appendText("<Audio \(nAudio)>: ")
            case let .video(pairs, timestamps):
                nVideo += 1
                appendText("<Video \(nVideo)>: ")
                precondition(timestamps.count == pairs.count * 2,
                             "\(pairs.count) frame pair(s) need \(pairs.count * 2) "
                             + "timestamps, got \(timestamps.count)")
                for (i, pair) in pairs.enumerated() {
                    // The block's stamp is the MIDPOINT of its two frames, and
                    // it is formatted with exactly one decimal — "<0.2 seconds>",
                    // not "<0.25 seconds>". Both halves matter: the text is
                    // tokenised, so a different string is a different length.
                    let mid = (timestamps[2 * i] + timestamps[2 * i + 1]) / 2.0
                    appendText(String(format: "<%.1f seconds>", mid))
                    appendVision(pair.block, pair.grid)
                }
            }
        }
        appendText(prompt)
        if cursor == 0 { appendIds([Qwen2Tokenizer.padToken]) }
        let gridPerImage = grids
        let blocks = visionBlocks

        let embeds = concatenated(pieces, axis: 0).expandedDimensions(axis: 0)
        let s = embeds.dim(1)

        // Tags widen by one on each side: the reference tags the *whole* vision
        // block as video modality, start and end tokens included.
        var tags = [Int](repeating: 1, count: s)
        for span in spans {
            for i in max(0, span.start - 1) ..< min(s, span.end + 1) { tags[i] = 0 }
        }

        guard !spans.isEmpty else {
            return Assembled(embeds: embeds, spans: [], positionIds: nil,
                             visualMask: nil, deepstack: [], tags: tags)
        }

        var mask = [Int32](repeating: 0, count: s)
        for span in spans { for i in span.start ..< span.end { mask[i] = 1 } }

        let stackCount = blocks.first?.deepstack.count ?? 0
        let deepstack = (0 ..< stackCount).map { k in
            concatenated(blocks.map { $0.deepstack[k].asType(.float32) }, axis: 0)
        }

        return Assembled(embeds: embeds, spans: spans,
                         positionIds: positionIds(spans: spans, grids: gridPerImage,
                                                  sequenceLength: s),
                         visualMask: MLXArray(mask), deepstack: deepstack, tags: tags)
    }

    /// `qwen2vl_mrope_position_ids` — three rows of positions, t/h/w.
    ///
    /// Text runs sequentially. An image span pins **t** to a single value, and
    /// gives **h**/**w** the row and column of each token inside the *merged*
    /// grid (so `grid.h / 2`, not `grid.h`). Text after the image resumes at
    /// `start + max(grid)/2` — the image advances the clock by its merged extent,
    /// not by its token count, which is why `offset` goes negative.
    static func positionIds(spans: [Span], grids: [VisionGrid],
                            sequenceLength s: Int) -> MLXArray {
        var rows = [[Float]](repeating: [Float](repeating: 0, count: s), count: 3)
        var offset = 0
        var wroteHead = false

        for (span, grid) in zip(spans, grids) {
            if !wroteHead {
                for i in 0 ..< span.start { for r in 0 ..< 3 { rows[r][i] = Float(i) } }
                wroteHead = true
            }
            let lenMax = max(grid.t, max(grid.h, grid.w)) / 2
            let startNext = lenMax + span.start

            // text after the image
            for (k, i) in (span.end ..< s).enumerated() {
                for r in 0 ..< 3 { rows[r][i] = Float(startNext + offset + k) }
            }
            // t: constant across the span
            for i in span.start ..< span.end { rows[0][i] = Float(span.start + offset) }
            // h: row index, each repeated across the merged width.
            //
            // The repeat count is `ceil(size / mh)`, which is the merged width —
            // NOT `mh`. The two agree on a square image and disagree on every
            // other, which is exactly the kind of thing a 224x224 fixture
            // cannot see.
            let mh = grid.h / 2
            let repeatH = (span.size + mh - 1) / mh
            for k in 0 ..< span.size {
                rows[1][span.start + k] = Float(span.start + offset + k / repeatH)
            }
            // w: column index, cycling across the merged width
            let mw = grid.w / 2
            for k in 0 ..< span.size {
                rows[2][span.start + k] = Float(span.start + offset + k % mw)
            }
            offset += lenMax - span.size
        }
        return MLXArray(rows.flatMap { $0 }, [3, s])
    }
}
