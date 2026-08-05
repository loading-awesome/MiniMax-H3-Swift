import Testing
import Foundation
import H3Foundation
import H3Catalog
import H3Pipeline

/// The rules a render must satisfy, none of which needs a checkpoint to check.
///
/// In the experimental tree all of this lived inside an ArgumentParser
/// `validate()`, so the only way to exercise it was to spawn a process — and
/// consequently none of it was exercised at all.
@Suite("render request")
struct RenderRequestTests {

    static func base(_ mutate: (inout RenderRequest) -> Void = { _ in }) -> RenderRequest {
        var r = RenderRequest(prompt: "a cat", videoOutput: URL(fileURLWithPath: "/tmp/o.mp4"))
        mutate(&r)
        return r
    }

    static func url(_ s: String) -> URL { URL(fileURLWithPath: s) }

    @Test("the partition follows from the mode, never from a flag")
    func modeSelectsPartition() {
        // The bug this pins cost every reference render on 2026-08-05: the
        // runner hard-coded one checkpoint path, so image and video references
        // ran against FL2VA — weights never trained to consume them — and
        // produced plausible output with no error anywhere.
        #expect(Self.base().mode == .textToVideo)
        #expect(Self.base { $0.firstFrame = Self.url("/a.png") }.mode == .firstLastFrame)
        #expect(Self.base { $0.lastFrame = Self.url("/a.png") }.mode == .firstLastFrame)
        #expect(Self.base { $0.referenceImages = [Self.url("/a.png")] }.mode == .reference)
        #expect(Self.base { $0.referenceVideos = [Self.url("/a.mp4")] }.mode == .reference)
        #expect(Self.base { $0.referenceAudio = [Self.url("/a.wav")] }.mode == .reference)

        #expect(Self.base().mode.requiredPartition == .fl2va)
        #expect(Self.base { $0.firstFrame = Self.url("/a.png") }
                    .mode.requiredPartition == .fl2va)
        #expect(Self.base { $0.referenceImages = [Self.url("/a.png")] }
                    .mode.requiredPartition == .ref2va)
    }

    @Test("anchors and references cannot share a payload")
    func anchorsExcludeReferences() {
        // Not a style rule: they are served by different partitions, so no
        // single checkpoint could honour both.
        let r = Self.base {
            $0.firstFrame = Self.url("/a.png")
            $0.referenceImages = [Self.url("/b.png")]
        }
        #expect(throws: H3Error.self) { try r.validate() }
    }

    @Test("soundtracks are index-matched to reference videos")
    func soundtrackCount() {
        // More soundtracks than videos would silently drop the tail, and every
        // dropped one moves every later `<Audio j>` ordinal — which slides the
        // conditioning while leaving every shape valid.
        let r = Self.base {
            $0.referenceVideos = [Self.url("/v0.mp4")]
            $0.referenceVideoSoundtracks = [Self.url("/a0.wav"), Self.url("/a1.wav")]
        }
        #expect(throws: H3Error.self) { try r.validate() }
    }

    @Test("paired soundtracks are consumed before standalone clips")
    func audioOrdering() {
        // The DiT walks the conditioning audio as one flat stream in segment
        // order, and a video_audio block puts its audio rows *ahead* of its
        // video rows inside the block. So the order is: each video's own
        // soundtrack at that video's position, then the standalone clips —
        // whatever order the caller passed them in.
        let r = Self.base {
            $0.referenceVideos = [Self.url("/v0.mp4"), Self.url("/v1.mp4"),
                                  Self.url("/v2.mp4")]
            // v1 is deliberately silent, so the tail must not shuffle up.
            $0.referenceVideoSoundtracks = [Self.url("/pair0.wav"), nil,
                                            Self.url("/pair2.wav")]
            $0.referenceAudio = [Self.url("/solo.wav")]
        }
        #expect(r.orderedAudio.map(\.lastPathComponent)
                == ["pair0.wav", "pair2.wav", "solo.wav"])
        #expect(r.soundtrack(for: 1) == nil)
        // Past the end of the array is "no soundtrack", not a crash.
        #expect(r.soundtrack(for: 99) == nil)
    }

    @Test("off-grid dimensions are refused with the reason")
    func dimensionsOnGrid() {
        // The VAE downsamples by 16 and the DiT patchifies by 2, so an off-grid
        // size produces a latent the packed layout cannot describe.
        #expect(throws: H3Error.self) {
            try Self.base { $0.width = 862; $0.height = 480 }.validate()
        }
        #expect(throws: Never.self) {
            try Self.base { $0.width = 864; $0.height = 480 }.validate()
        }
        // Half a pair is not a shape.
        #expect(throws: H3Error.self) { try Self.base { $0.width = 864 }.validate() }
    }

    @Test("a negative prompt without guidance is refused rather than ignored")
    func negationNeedsGuidance() {
        // Silently ignoring it is the failure worth avoiding: the render looks
        // like it honoured the negation and did not.
        #expect(throws: H3Error.self) {
            try Self.base { $0.negativePrompt = "blurry" }.validate()
        }
        #expect(throws: Never.self) {
            try Self.base { $0.negativePrompt = "blurry"; $0.cfgScale = 5 }.validate()
        }
    }

    @Test("2K is refused as a base render size")
    func twoKIsAnUpscaleTarget() {
        #expect(throws: H3Error.self) {
            try Self.base { $0.resolution = .k2 }.validate()
        }
        // Deliberate experiments stay possible, and are logged.
        #expect(throws: Never.self) {
            try Self.base { $0.resolution = .k2; $0.allowSuboptimal = true }.validate()
        }
    }

    @Test("reference counts stop at the reference node's own limits")
    func referenceLimits() {
        #expect(throws: H3Error.self) {
            try Self.base { $0.referenceImages = (0 ..< 10).map { Self.url("/i\($0).png") } }
                .validate()
        }
        #expect(throws: H3Error.self) {
            try Self.base { $0.referenceVideos = (0 ..< 4).map { Self.url("/v\($0).mp4") } }
                .validate()
        }
    }

    @Test("four seconds validates and the policy still refuses it")
    func fourSecondsLandsUnderTheFloor() {
        // 4 s is inside the model's stated 4-15 s range and is the one value in
        // it that snaps to 107 frames — under the 124-frame trained floor. The
        // request is well-formed; the policy is what refuses, and it needs the
        // geometry to say so.
        let r = Self.base { $0.seconds = 4; $0.width = 864; $0.height = 480 }
        #expect(throws: Never.self) { try r.validate() }
        let geometry = LatentGeometry(width: 864, height: 480, length: 4 * H3Video.fps)
        #expect(geometry.frameCount == 107)
        #expect(!r.policyViolations(geometry: geometry).isEmpty)

        let five = Self.base { $0.seconds = 5; $0.width = 864; $0.height = 480 }
        let fiveGeometry = LatentGeometry(width: 864, height: 480, length: 5 * H3Video.fps)
        #expect(fiveGeometry.frameCount == 124)
        #expect(five.policyViolations(geometry: fiveGeometry).isEmpty)
    }

    @Test("dimensions come from whichever source was given, in the stated order")
    func dimensionPrecedence() throws {
        #expect(try Self.base { $0.width = 864; $0.height = 480 }.dimensions() == (864, 480))
        #expect(try Self.base().dimensions() == (1344, 768))
        #expect(try Self.base { $0.aspectRatio = .r9x16 }.dimensions() == (768, 1344))
        #expect(try Self.base { $0.aspectRatio = .r1x1 }.dimensions() == (768, 768))
    }
}
