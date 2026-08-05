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

    @Test("a keyframe anchor in the middle is refused, not pinned to the wrong instant")
    func middleKeyframeRefused() {
        // The reference defines cond_t for the first and last frames only.
        // Accepting a middle index and quietly treating it as the last frame
        // would pin the anchor to the wrong instant with every shape still
        // correct — the failure mode this codebase is built against.
        #expect(throws: H3Error.self) {
            _ = try PackedLayout(textTokens: 52, geometry: Self.geometry,
                                 keyframes: [KeyframeConfig(resolvedFrameIndex: 60)])
        }
    }

    @Test("the two anchors the reference does define are accepted")
    func firstAndLastAccepted() throws {
        let first = try PackedLayout(textTokens: 52, geometry: Self.geometry,
                                     keyframes: [KeyframeConfig(resolvedFrameIndex: 0)])
        #expect(first.segments.contains { $0.kind == .cond })

        // The last anchor sits on the *aligned* frame count, not the requested
        // duration: 5 s of 24 fps is 120 frames, and the request snaps up onto
        // the 17k+5 lattice to 124 before anything else sees it.
        #expect(Self.geometry.frameCount == 124)
        let last = try PackedLayout(
            textTokens: 52, geometry: Self.geometry,
            keyframes: [KeyframeConfig(resolvedFrameIndex: Self.geometry.frameCount - 1)])
        #expect(last.segments.contains { $0.kind == .cond })

        // 119 is the anchor you get by forgetting the alignment, and it is
        // refused rather than silently accepted a frame early.
        #expect(throws: H3Error.self) {
            _ = try PackedLayout(textTokens: 52, geometry: Self.geometry,
                                 keyframes: [KeyframeConfig(resolvedFrameIndex: 119)])
        }
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
