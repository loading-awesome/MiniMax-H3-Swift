// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// The modality tag an AdaLN row carries. Fixed by the reference's `seg_tag`
/// table in `comfy/ldm/minimax/model.py`: **video 0, text 1, audio 2**.
package enum ModalityTag: Int, Sendable {
    case video = 0
    case text = 1
    case audio = 2
}

/// A kind of span in the packed sequence.
///
/// The kind decides three separate things — which embedding stream fills the
/// rows, which modality tag the AdaLN row uses, and which timestep the span
/// runs on. They do not coincide: `cond` rows are video-tagged but sit at a
/// near-1.0 timestep of their own.
package enum SegmentKind: String, Sendable, Equatable {
    case text
    case cond
    case refImage = "ref_img"
    case refAudio = "ref_audio"
    case audio
    case video

    package var modality: ModalityTag {
        switch self {
        case .text: .text
        case .audio, .refAudio: .audio
        case .video, .cond, .refImage: .video
        }
    }

    /// Whether the span's rows come from the video patch projection.
    package var isVideoStream: Bool { self == .video || self == .cond || self == .refImage }
}

package struct PackedSegment: Sendable, Equatable {
    package let start: Int
    package let stop: Int
    package let kind: SegmentKind
    package init(start: Int, stop: Int, kind: SegmentKind) {
        self.start = start
        self.stop = stop
        self.kind = kind
    }
    package var count: Int { stop - start }
}

/// RoPE position coordinates.
///
/// Positions are **not** integer token indices. The spatial axes are
/// area-normalized so that a 480x864 render and a 720x720 render of the same
/// pixel count land on comparable coordinates, and the temporal axis advances in
/// rescaled frame spans rather than by one per token. Everything here is
/// computed in Double because the reference builds `position_ids` in float64
/// and only narrows to fp32 inside `rope_freqs`.
package enum PositionGrid {
    /// Frames represented by each video latent token, cycling with period 5.
    /// The first latent frame covers 1 source frame, the rest cover 4.
    package static let framesPerToken = [1, 4, 4, 4, 4]
    /// Temporal spans are scaled by 5/3 before being accumulated.
    package static let frameRescale = 5.0 / 3.0

    /// `linspace((1-ratio)/2, (1+ratio)/2, dim/patch, endpoint=False) * 32`
    /// where `ratio = dim / sqrt(h*w)`.
    package static func axis(dim: Int, patch: Int, sqrtArea: Double) throws -> [Double] {
        let ratio = Double(dim) / sqrtArea
        let n = dim / patch
        guard n > 0 else {
            throw H3Error.dimensionOffGrid(width: dim, height: dim, multiple: patch)
        }
        let step = ratio / Double(n)
        let origin = (1.0 - ratio) / 2.0
        return (0 ..< n).map { (Double($0) * step + origin) * 32.0 }
    }

    /// One latent frame's `(h, w)` coordinates, row-major over the 2x2-patch
    /// grid, plus the w axis on its own (the audio grid pins to its extremes).
    package static func frameGrid(h: Int, w: Int, patch: Int = 2)
        throws -> (rows: [(h: Double, w: Double)], wAxis: [Double]) {
        let area = (Double(h) * Double(w)).squareRoot()
        let hAxis = try axis(dim: h, patch: patch, sqrtArea: area)
        let wAxis = try axis(dim: w, patch: patch, sqrtArea: area)
        var rows: [(h: Double, w: Double)] = []
        rows.reserveCapacity(hAxis.count * wAxis.count)
        for hv in hAxis { for wv in wAxis { rows.append((hv, wv)) } }
        return (rows, wAxis)
    }

    /// `origin + exclusive_cumsum(spans)`, one t coordinate per latent frame.
    package static func videoTGrid(_ n: Int, origin: Double) -> [Double] {
        var out = [Double](repeating: 0, count: n)
        var acc = origin
        for k in 0 ..< n {
            out[k] = acc
            acc += frameRescale * Double(framesPerToken[k % framesPerToken.count])
        }
        return out
    }

    /// Total temporal extent of `n` video latent frames — what a cursor advances
    /// by when a reference video block precedes the target.
    package static func videoTSpan(_ n: Int) -> Double {
        (0 ..< n).reduce(0.0) { $0 + frameRescale * Double(framesPerToken[$1 % framesPerToken.count]) }
    }

    /// The t coordinate of a keyframe anchored at pixel frame `p`.
    ///
    /// Latent token `k` spans `frameRescale * framesPerToken[k % 5]` and covers
    /// `framesPerToken[k % 5]` pixel frames, so cumulative time at pixel frame
    /// `p` is exactly `frameRescale * p`. That identity is what makes an anchor
    /// away from the ends *defined* rather than guessed: substituting
    /// `p = frameCount - 1` reproduces the reference's own last-frame
    /// expression, because `sum(framesPerToken over latentT) == frameCount` on
    /// the 17k+5 lattice.
    ///
    ///     textTokens + frameRescale * (frameCount - 1)
    ///       == textTokens + frameRescale * frameCount - frameRescale
    ///       == textTokens + videoTSpan(latentT) - frameRescale
    ///
    /// **The two endpoints keep the reference's expressions rather than the
    /// general form.** They are mathematically identical, but the reference
    /// accumulates `latentT` float additions where the general form does one
    /// multiply, and the two disagree in the last bits (~7e-15 at F = 362).
    /// Reusing them means every existing first/last render produces
    /// byte-identical positions, so widening this cannot move an L1 verdict.
    ///
    /// `p` is a **pixel** frame index on the aligned lattice, not a latent one,
    /// and the caller is responsible for having aligned it. Range is not checked
    /// here; `PackedLayout` refuses out-of-range anchors where it knows the
    /// frame count.
    package static func condT(textTokens: Int, latentT: Int, frameCount: Int,
                              pixelIndex p: Int) -> Double {
        if p == 0 { return Double(textTokens) }
        if p == frameCount - 1 { return Double(textTokens) + videoTSpan(latentT) - frameRescale }
        return Double(textTokens) + frameRescale * Double(p)
    }
}

package struct KeyframeConfig: Sendable, Equatable {
    package let resolvedFrameIndex: Int
    package init(resolvedFrameIndex: Int) {
        self.resolvedFrameIndex = resolvedFrameIndex
    }
}

package enum ReferenceKind: String, Sendable, Equatable {
    case image
    case audio
    case video
    case videoAudio = "video_audio"
}

package struct ReferenceBlock: Sendable, Equatable {
    package let kind: ReferenceKind
    package let latentH: Int
    package let latentW: Int
    package let latentT: Int
    package let refAudioT: Int
    
    package init(kind: ReferenceKind, latentH: Int = 0, latentW: Int = 0, latentT: Int = 0, refAudioT: Int = 0) {
        self.kind = kind
        self.latentH = latentH
        self.latentW = latentW
        self.latentT = latentT
        self.refAudioT = refAudioT
    }
}

/// The packed sequence a DiT block actually sees.
///
/// **Order is text | cond/refs | audio | video** — target audio then target video,
/// always the last two segments. The reference builds it that way.
///
/// **There is no batch dimension inside the stack.** Blocks operate on
/// `[S, hidden]`, and slices come from the segment table.
package struct PackedLayout: Sendable, Equatable {
    package let textTokens: Int
    package let audioTokens: Int
    package let videoTokens: Int
    package let segments: [PackedSegment]
    /// Row-major `[S, 3]` of `(t, h, w)`, in float64 as the reference builds it.
    package let positionIds: [Double]

    /// - Throws: `H3Error.keyframeIndex` for an anchor off the timeline. This
    ///   was a `preconditionFailure`, which in a library takes the host
    ///   application down over a value the caller is entitled to get wrong.
    package init(textTokens: Int, geometry: LatentGeometry,
                keyframes: [KeyframeConfig] = [], refs: [ReferenceBlock] = []) throws {
        self.textTokens = textTokens
        self.audioTokens = geometry.audioTokens
        self.videoTokens = geometry.videoTokens

        let patch = geometry.config.patchSize[1]
        let (frame, wAxis) = try PositionGrid.frameGrid(h: geometry.latentH,
                                                    w: geometry.latentW, patch: patch)
        let wLow = wAxis.first ?? 0, wHigh = wAxis.last ?? 0
        var pos = [Double]()
        var segments: [PackedSegment] = []
        var row = 0

        // 1. Text segment
        segments.append(PackedSegment(start: row, stop: row + textTokens, kind: .text))
        for i in 0 ..< textTokens { pos.append(contentsOf: [Double(i), 0, 0]) }
        row += textTokens
        var cursor = Double(textTokens)

        // 2. Keyframes (fl2va)
        for kf in keyframes {
            // The reference raises on any anchor that is not an end, but only
            // because it never derives the general position — the two branches
            // it does define are one expression evaluated at its endpoints, and
            // `condT` carries the derivation. What stays refused is an index off
            // the timeline entirely, which has no coordinate at all.
            //
            // Anchors away from the ends are *positionally* exact and
            // *behaviourally* untested: fl2va was trained with conditioning at
            // the ends, so a middle anchor is out of distribution. It pins the
            // rows to the right instant; it does not promise the model lands on
            // them.
            guard kf.resolvedFrameIndex >= 0, kf.resolvedFrameIndex < geometry.frameCount else {
                throw H3Error.keyframeIndex(index: kf.resolvedFrameIndex,
                                            frameCount: geometry.frameCount)
            }
            let condT = PositionGrid.condT(textTokens: textTokens, latentT: geometry.latentT,
                                           frameCount: geometry.frameCount,
                                           pixelIndex: kf.resolvedFrameIndex)
            segments.append(PackedSegment(start: row, stop: row + frame.count, kind: .cond))
            for r in frame { pos.append(contentsOf: [condT, r.h, r.w]) }
            row += frame.count
        }

        // 3. References (ref2va / ref_img / ref_audio / video)
        for blk in refs {
            switch blk.kind {
            case .image:
                let (rFrame, _) = try PositionGrid.frameGrid(h: blk.latentH, w: blk.latentW, patch: patch)
                let n = rFrame.count
                segments.append(PackedSegment(start: row, stop: row + n, kind: .refImage))
                for r in rFrame { pos.append(contentsOf: [cursor, r.h, r.w]) }
                row += n
                cursor += 1.0

            case .audio:
                let rt = blk.refAudioT
                if rt > 0 {
                    segments.append(PackedSegment(start: row, stop: row + rt * 2, kind: .refAudio))
                    for w in [wLow, wHigh] {
                        for i in 0 ..< rt { pos.append(contentsOf: [cursor + Double(i), 0, w]) }
                    }
                    row += rt * 2
                }
                cursor += Double(rt)

            case .video, .videoAudio:
                let rt = blk.refAudioT
                let vt = blk.latentT
                let (rFrame, rWAxis) = try PositionGrid.frameGrid(h: blk.latentH, w: blk.latentW, patch: patch)
                let rWLow = rWAxis.first ?? 0, rWHigh = rWAxis.last ?? 0
                if rt > 0 {
                    segments.append(PackedSegment(start: row, stop: row + rt * 2, kind: .refAudio))
                    for w in [rWLow, rWHigh] {
                        for i in 0 ..< rt { pos.append(contentsOf: [cursor + Double(i), 0, w]) }
                    }
                    row += rt * 2
                }
                let n = vt * rFrame.count
                segments.append(PackedSegment(start: row, stop: row + n, kind: .refImage))
                let tGrid = PositionGrid.videoTGrid(vt, origin: cursor)
                for t in tGrid {
                    for r in rFrame { pos.append(contentsOf: [t, r.h, r.w]) }
                }
                row += n
                cursor += max(Double(rt), PositionGrid.videoTSpan(vt))
            }
        }

        // 4. Target audio (always second to last)
        segments.append(PackedSegment(start: row, stop: row + audioTokens, kind: .audio))
        for w in [wLow, wHigh] {
            for i in 0 ..< geometry.audioT { pos.append(contentsOf: [cursor + Double(i), 0, w]) }
        }
        row += audioTokens

        // 5. Target video (always last)
        segments.append(PackedSegment(start: row, stop: row + videoTokens, kind: .video))
        for t in PositionGrid.videoTGrid(geometry.latentT, origin: cursor) {
            for r in frame { pos.append(contentsOf: [t, r.h, r.w]) }
        }
        row += videoTokens

        self.positionIds = pos
        self.segments = segments
    }

    package var totalTokens: Int { positionIds.count / 3 }

    /// Half-open ranges into the packed sequence, in packing order.
    package var textRange: Range<Int> {
        let seg = segments.first { $0.kind == .text }!
        return seg.start ..< seg.stop
    }
    package var audioRange: Range<Int> {
        let seg = segments.first { $0.kind == .audio }!
        return seg.start ..< seg.stop
    }
    package var videoRange: Range<Int> {
        let seg = segments.first { $0.kind == .video }!
        return seg.start ..< seg.stop
    }

    /// [S, hidden] — the shape every block tap is recorded at.
    package func blockShape(config: H3Config) -> [Int] { [totalTokens, config.hiddenSize] }
}
