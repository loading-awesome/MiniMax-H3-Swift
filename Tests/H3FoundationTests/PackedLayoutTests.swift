// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
@testable import H3Foundation

/// The packed layout's refusals, which used to be traps.
///
/// A `preconditionFailure` in a library takes the host application down, and
/// both of the ones replaced here are reachable from ordinary wrong input: a
/// keyframe index a caller chose, and a dimension a caller asked for.
@Suite("packed layout refusals")
struct PackedLayoutTests {

    static let geometry = LatentGeometry(width: 864, height: 480, length: 5 * H3Video.fps)
    /// 480/16/2 * 864/16/2 — one cond segment is this many rows.
    static let rowsPerFrame = 405

    /// The t coordinate of every `cond` segment, in packing order.
    static func condTimes(_ layout: PackedLayout) -> [Double] {
        layout.segments.filter { $0.kind == .cond }.map { layout.positionIds[$0.start * 3] }
    }

    @Test("an anchor off the timeline is refused, not pinned to the wrong instant")
    func offTimelineAnchorRefused() {
        // What is refused is now an index with no coordinate at all, rather than
        // any index away from the ends. Accepting one and clamping it would pin
        // the anchor to the wrong instant with every shape still correct — the
        // failure mode this codebase is built against.
        #expect(throws: H3Error.self) {
            _ = try PackedLayout(textTokens: 52, geometry: Self.geometry,
                                 keyframes: [KeyframeConfig(resolvedFrameIndex: -1)])
        }
        #expect(throws: H3Error.self) {
            _ = try PackedLayout(textTokens: 52, geometry: Self.geometry,
                                 keyframes: [KeyframeConfig(resolvedFrameIndex:
                                                              Self.geometry.frameCount)])
        }
    }

    @Test("the two anchors the reference defines keep their exact coordinates")
    func endpointsAreUnchanged() throws {
        // The general form is one multiply where the reference accumulates
        // latentT float additions, and they disagree in the last bits. If this
        // ever drifts, every first/last golden in the tree moves with it, so it
        // is asserted as exact equality rather than a tolerance.
        let g = Self.geometry
        let first = try PackedLayout(textTokens: 52, geometry: g,
                                     keyframes: [KeyframeConfig(resolvedFrameIndex: 0)])
        #expect(Self.condTimes(first) == [52.0])

        // The last anchor sits on the *aligned* frame count, not the requested
        // duration: 5 s of 24 fps is 120 frames, and the request snaps up onto
        // the 17k+5 lattice to 124 before anything else sees it.
        #expect(g.frameCount == 124)
        let last = try PackedLayout(
            textTokens: 52, geometry: g,
            keyframes: [KeyframeConfig(resolvedFrameIndex: g.frameCount - 1)])
        let want = 52.0 + PositionGrid.videoTSpan(g.latentT) - PositionGrid.frameRescale
        #expect(Self.condTimes(last) == [want])
    }

    @Test("the lattice identity the general anchor position rests on")
    func spanSumEqualsFrameCount() {
        // cond_t = textTokens + frameRescale * p is only the same expression as
        // the reference's last-frame branch because the per-token frame spans
        // sum to exactly the frame count. That holds on the 17k+5 lattice and
        // nowhere else, so it is checked rather than assumed.
        for frames in [5, 22, 39, 124, 175, 362] {
            #expect(LatentGeometry.alignFrameCount(frames) == frames,
                    "\(frames) is not on the lattice; the case is mis-stated")
            let g = LatentGeometry(width: 864, height: 480, length: frames)
            let covered = (0 ..< g.latentT).reduce(0) {
                $0 + PositionGrid.framesPerToken[$1 % PositionGrid.framesPerToken.count]
            }
            #expect(covered == frames)
            // ...and therefore the two expressions agree to float noise.
            let general = 52.0 + PositionGrid.frameRescale * Double(frames - 1)
            let reference = 52.0 + PositionGrid.videoTSpan(g.latentT) - PositionGrid.frameRescale
            #expect(abs(general - reference) < 1e-9)
        }
    }

    @Test("an anchor away from the ends lands strictly between them")
    func middleAnchorIsOrdered() throws {
        let g = Self.geometry
        let layout = try PackedLayout(textTokens: 52, geometry: g, keyframes: [
            KeyframeConfig(resolvedFrameIndex: 0),
            KeyframeConfig(resolvedFrameIndex: 60),
            KeyframeConfig(resolvedFrameIndex: g.frameCount - 1),
        ])
        let t = Self.condTimes(layout)
        #expect(t.count == 3)
        #expect(t[0] < t[1] && t[1] < t[2])
        #expect(abs(t[1] - (52.0 + PositionGrid.frameRescale * 60.0)) < 1e-9)

        // Each anchor is its own cond segment of a full frame's rows, and the
        // target video still starts where it did — anchors sit inside the text
        // cursor's span and must not advance it.
        #expect(layout.segments.filter { $0.kind == .cond }
                    .allSatisfy { $0.count == Self.rowsPerFrame })
        let bare = try PackedLayout(textTokens: 52, geometry: g)
        #expect(layout.positionIds[layout.videoRange.lowerBound * 3]
                == bare.positionIds[bare.videoRange.lowerBound * 3])
    }

    @Test("anchors one per second are evenly spaced on the timeline")
    func oneAnchorPerSecondIsUniform() throws {
        // The case this widening exists for: an anchor every 24 frames. Spacing
        // must be constant, because the per-token span cycles with period 5 and
        // 24 is not a multiple of 5 — an implementation that walked tokens
        // instead of pixels would produce a wobble here and nowhere else.
        let g = LatentGeometry(width: 864, height: 480, length: 362)
        #expect(g.frameCount == 362)
        let anchors = stride(from: 0, to: g.frameCount, by: H3Video.fps).map {
            KeyframeConfig(resolvedFrameIndex: $0)
        }
        #expect(anchors.count == 16)
        let t = Self.condTimes(try PackedLayout(textTokens: 52, geometry: g, keyframes: anchors))
        let gaps = zip(t.dropFirst(), t).map(-)
        let want = PositionGrid.frameRescale * Double(H3Video.fps)
        #expect(gaps.allSatisfy { abs($0 - want) < 1e-9 })
    }

    @Test("an axis smaller than the patch is refused")
    func axisSmallerThanPatch() {
        #expect(throws: H3Error.self) {
            _ = try PositionGrid.axis(dim: 1, patch: 2, sqrtArea: 10)
        }
        #expect(throws: Never.self) {
            _ = try PositionGrid.axis(dim: 2, patch: 2, sqrtArea: 10)
        }
    }

    @Test("the segment order is text, conditions, audio, video")
    func segmentOrder() throws {
        // Target audio then target video, always the last two. The reference
        // builds it that way, and every downstream slice comes from this table.
        let layout = try PackedLayout(
            textTokens: 52, geometry: Self.geometry,
            refs: [ReferenceBlock(kind: .image, latentH: 30, latentW: 54)])
        let kinds = layout.segments.map(\.kind)
        #expect(kinds.first == .text)
        #expect(kinds.suffix(2) == [.audio, .video])
        #expect(kinds.contains(.refImage))
    }
}
