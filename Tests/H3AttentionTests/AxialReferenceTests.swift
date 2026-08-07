// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import H3Foundation
@testable import H3Attention

/// The axial oracle, before any Metal exists.
///
/// Phase 6E is gated on this suite. If the deterministic topology cannot
/// reproduce dense when it selects everything, or if the streaming recurrence
/// disagrees with the masked definition, there is nothing to build a kernel on
/// and the correct outcome is to stop.
@Suite("axial reference", .serialized)
struct AxialReferenceTests {

    /// A small packed sequence with the production *shape* of the problem:
    /// a prefix, then a video tail on a frames × height × width lattice.
    static func fixture(prefix: Int = 7, frames: Int = 5, h: Int = 3, w: Int = 4,
                        heads: Int = 2, d: Int = 16, seed: UInt64 = 4)
        -> (q: MLXArray, k: MLXArray, v: MLXArray, t: AxialTopology, scale: Float) {
        MLXRandom.seed(seed)
        let video = frames * h * w
        let s = prefix + video
        let t = AxialTopology(sequenceLength: s, videoSpan: prefix ..< s,
                              frames: frames, height: h, width: w,
                              landmarkFrames: [0, frames - 1])!
        return (MLXRandom.normal([heads, s, d]), MLXRandom.normal([heads, s, d]),
                MLXRandom.normal([heads, s, d]), t, 1.0 / Float(d).squareRoot())
    }

    static func relRMS(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = a.asType(.float32) - b.asType(.float32)
        return MLX.sqrt(MLX.mean(d * d)).item(Float.self)
             / MLX.sqrt(MLX.mean(b.asType(.float32) * b.asType(.float32))).item(Float.self)
    }

    @Test("selecting every key reproduces dense exactly")
    func fullSelectionIsDense() {
        // **The first test written, and the one that catches the trap.** An
        // implementation that runs a separate attention per axis and combines
        // them cannot pass this: each slice carries its own softmax denominator,
        // so the sum is not attention over the union and does not collapse to
        // dense even when the union is everything.
        let f = Self.fixture()
        // Every frame a landmark: the topology now admits the whole sequence.
        let dense = AxialTopology(sequenceLength: f.t.videoSpan.upperBound,
                                  videoSpan: f.t.videoSpan, frames: f.t.frames,
                                  height: f.t.height, width: f.t.width,
                                  landmarkFrames: Array(0 ..< f.t.frames))!
        let got = AxialReference.attend(queries: f.q, keys: f.k, values: f.v,
                                        scale: f.scale, topology: dense)
        let want = MLXFast.scaledDotProductAttention(
            queries: f.q.expandedDimensions(axis: 0).asType(.float32),
            keys: f.k.expandedDimensions(axis: 0).asType(.float32),
            values: f.v.expandedDimensions(axis: 0).asType(.float32),
            scale: f.scale, mask: nil).squeezed(axis: 0)
        #expect(Self.relRMS(got, want) < 1e-6)
    }

    @Test("the streaming recurrence agrees with the masked definition")
    func streamingMatchesMasked() {
        // The masked form is exact by construction; the streaming form is what
        // a kernel must implement, and its failure mode is silent. Several key
        // tile sizes, including ones that do not divide the sequence, because
        // a ragged final tile is where an online softmax goes wrong.
        let f = Self.fixture()
        let want = AxialReference.attend(queries: f.q, keys: f.k, values: f.v,
                                         scale: f.scale, topology: f.t)
        for tile in [1, 7, 16, 64, 1_000] {
            let got = AxialReference.attendStreamingKeys(
                queries: f.q, keys: f.k, values: f.v, scale: f.scale,
                topology: f.t, keyTile: tile)
            #expect(Self.relRMS(got, want) < 2e-6, "key tile \(tile)")
        }
    }

    @Test("query chunking changes nothing")
    func queryChunkingIsExact() {
        // Chunking over queries is exact by construction — each query's softmax
        // is complete inside its chunk — but only if the chunk boundary does
        // not leak. A chunk size that splits a latent frame is the case worth
        // checking, since the topology is defined per frame.
        let f = Self.fixture()
        let base = AxialReference.attend(queries: f.q, keys: f.k, values: f.v,
                                         scale: f.scale, topology: f.t,
                                         queryChunk: 4_096)
        for chunk in [1, 5, 13, 32] {
            let got = AxialReference.attend(queries: f.q, keys: f.k, values: f.v,
                                            scale: f.scale, topology: f.t,
                                            queryChunk: chunk)
            #expect(Self.relRMS(got, base) < 1e-6, "query chunk \(chunk)")
        }
    }

    @Test("the bulk mask matches the pairwise definition")
    func bulkMaskMatchesThePairwiseDefinition() {
        // `AxialTopology.allows` is the definition; `maskRows` is a vectorised
        // reconstruction of it. A bulk construction that is wrong in one corner
        // — the first frame, the last landmark, a prefix row — is exactly what
        // a per-pair check catches and a spot check does not.
        let f = Self.fixture()
        let s = f.t.videoSpan.upperBound
        let mask = AxialReference.maskRows(0 ..< s, topology: f.t)
        MLX.eval(mask)
        let flat = mask.asArray(Float.self)
        var mismatches = 0
        for q in 0 ..< s {
            for k in 0 ..< s where (flat[q * s + k] == 0) != f.t.allows(query: q, key: k) {
                mismatches += 1
            }
        }
        #expect(mismatches == 0)
    }

    @Test("a video query sees its frame, its column, the landmarks and the prefix — and nothing else")
    func topologyIsWhatItClaims() {
        let f = Self.fixture(prefix: 7, frames: 5, h: 3, w: 4)
        let t = f.t
        // Frame 2, position (y=1, x=2) -> token 7 + 2*12 + 1*4 + 2 = 37.
        let q: Int = 7 + 2 * 12 + 1 * 4 + 2
        let loc = t.locate(q)!
        #expect(loc.frame == 2 && loc.y == 1 && loc.x == 2)
        let sameFrame = 7 + 2 * 12 + 0
        let sameColumn = 7 + 4 * 12 + 1 * 4 + 2
        let inLandmark0 = 7 + 0 * 12 + 5
        let inLandmark4 = 7 + 4 * 12 + 5
        let excluded = 7 + 1 * 12 + 0 * 4 + 0
        #expect(t.allows(query: q, key: 3))            // prefix
        #expect(t.allows(query: q, key: sameFrame))
        #expect(t.allows(query: q, key: sameColumn))
        #expect(t.allows(query: q, key: inLandmark0))
        #expect(t.allows(query: q, key: inLandmark4))
        // Frame 1 is not a landmark, not the query's frame, and this position
        // differs — so it must be excluded. If nothing is excluded the topology
        // is dense and every measurement below is meaningless.
        #expect(!t.allows(query: q, key: excluded))
        // And a prefix query sees everything.
        #expect((0 ..< t.videoSpan.upperBound).allSatisfy { t.allows(query: 0, key: $0) })
    }

    @Test("density is counted, not estimated")
    func densityMatchesTheMask() {
        // Adding the three axes would double-count their overlap and overstate
        // the density — the direction that makes the method look worse than it
        // is, and the one a sanity check is least likely to question.
        let f = Self.fixture(prefix: 7, frames: 6, h: 3, w: 4)
        let s = f.t.videoSpan.upperBound
        let mask = AxialReference.maskRows(0 ..< s, topology: f.t)
        MLX.eval(mask)
        let counted = Double(MLX.sum((mask .== MLXArray(Float(0)))
            .asType(.int32)).item(Int32.self)) / Double(s * s)
        // Exact, not approximate: the closed form counts by frame, so it
        // agrees with the mask to floating-point noise.
        #expect(abs(counted - f.t.density) < 1e-9,
                "counted \(counted) against claimed \(f.t.density)")
    }

    @Test("a geometry that does not match the packed sequence is refused")
    func mismatchedGeometryRefused() {
        // The indices below are all derived from frames/height/width. If those
        // disagree with the actual video span, every frame boundary is wrong
        // and the topology silently attends the wrong tokens.
        let g = LatentGeometry(width: 864, height: 480, length: 124)
        #expect(AxialTopology(sequenceLength: 15_731,
                              videoSpan: 746 ..< 15_731, geometry: g) != nil)
        #expect(AxialTopology(sequenceLength: 15_731,
                              videoSpan: 700 ..< 15_731, geometry: g) == nil)
        #expect(AxialTopology(sequenceLength: 15_731,
                              videoSpan: 746 ..< 15_000, geometry: g) == nil)
    }

    @Test("landmark frames always include both ends and never repeat")
    func landmarkPlacement() {
        let g = LatentGeometry(width: 864, height: 480, length: 124)
        let t = AxialTopology(sequenceLength: 15_731, videoSpan: 746 ..< 15_731,
                              geometry: g, landmarks: 3)!
        #expect(t.frames == 37)
        #expect(t.landmarkFrames == [0, 18, 36])
        // Asking for fewer than two, or for more than there are frames, must
        // not produce a set missing an endpoint or containing a duplicate:
        // a repeated landmark narrows the set while looking like it has not.
        let two = AxialTopology(sequenceLength: 15_731, videoSpan: 746 ..< 15_731,
                                geometry: g, landmarks: 1)!
        #expect(two.landmarkFrames == [0, 36])
        let all = AxialTopology(sequenceLength: 15_731, videoSpan: 746 ..< 15_731,
                                geometry: g, landmarks: 999)!
        #expect(all.landmarkFrames.count == 37)
        #expect(Set(t.landmarkFrames).count == t.landmarkFrames.count)
    }
}
