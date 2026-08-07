// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import H3Foundation

/// Which keys a query is allowed to see — decided by position alone.
///
/// **The whole point is that no content is consulted.** Sol-Attn chose its
/// blocks from the scores themselves, which made the operator a function of the
/// latent and therefore different at every step: a block near the threshold
/// flipped in and out, the trajectory changed discontinuously, and a viewer saw
/// the subject pulse. This topology is a function of *index* only. The same
/// query attends the same keys at every step of every render at a given shape,
/// so whatever it costs in accuracy it cannot cost in stability.
///
/// That also makes it precomputable once per render rather than per call, and
/// checkable without a GPU — the whole of this type is integer arithmetic.
///
/// ## The topology
///
/// The packed sequence is `[text][cond][ref][audio][video]`, and everything
/// before the video tail is the **prefix**.
///
///  * **Prefix queries attend everything.** Text, conditioning, references and
///    the target audio are 746 rows of 15,731 at the production shape — under
///    5% of the sequence, and the rows carrying dialogue and lip-sync. Making
///    them dense costs almost nothing and removes the failure that every
///    published cache and sparse method for this model shares.
///  * **Every query keeps the whole prefix as keys.** A video token that cannot
///    see the text has lost the prompt.
///  * **Video queries attend three axes**, unioned:
///     - every token in the **same latent frame** — the spatial neighbourhood,
///       405 tokens at production shape;
///     - the **same spatial position across all frames** — the temporal
///       trajectory of that patch, 37 tokens;
///     - every token in a small set of **landmark frames**, which give a
///       global reference that neither axis provides.
///
/// At 864x480x124 that is roughly 2,400 of 15,731 keys per video query, about
/// 15% density, against a prefix that stays exact.
///
/// **Landmark frames are the part with a free parameter**, and it is fixed per
/// profile rather than tuned per render. The first and last latent frames are
/// always in: the first anchors what the clip is of, the last is where the
/// trajectory is going, and a video model that loses either drifts.
package struct AxialTopology: Sendable, Equatable {

    /// Rows before the video tail — text, conditioning, references, audio.
    package let prefix: Range<Int>
    package let videoSpan: Range<Int>
    /// Patchified video lattice. `frames * height * width == videoSpan.count`.
    package let frames: Int
    package let height: Int
    package let width: Int
    /// Frame indices every video query may attend in full, ascending and
    /// deduplicated.
    package let landmarkFrames: [Int]

    package var tokensPerFrame: Int { height * width }

    /// A lattice given directly, for tests and for callers that do not have a
    /// `LatentGeometry`. Still refuses a lattice that does not fill the span —
    /// the checked constructor below exists because that mismatch is silent.
    package init?(sequenceLength: Int, videoSpan: Range<Int>,
                 frames: Int, height: Int, width: Int, landmarkFrames: [Int]) {
        guard frames * height * width == videoSpan.count,
              videoSpan.upperBound == sequenceLength,
              videoSpan.lowerBound >= 0,
              landmarkFrames.allSatisfy({ (0 ..< frames).contains($0) }) else { return nil }
        self.prefix = 0 ..< videoSpan.lowerBound
        self.videoSpan = videoSpan
        self.frames = frames
        self.height = height
        self.width = width
        self.landmarkFrames = Set(landmarkFrames).sorted()
    }

    /// - Parameter landmarks: how many landmark frames to place, including the
    ///   first and last. Fewer than 2 is clamped to 2 — dropping either end is
    ///   not a configuration this topology offers.
    package init?(sequenceLength: Int, videoSpan: Range<Int>,
                 geometry: LatentGeometry, landmarks: Int = 3) {
        let h = geometry.latentH / geometry.config.patchSize[1]
        let w = geometry.latentW / geometry.config.patchSize[2]
        let f = geometry.latentT
        // Refuse rather than guess. A mismatch here means the caller's geometry
        // is not the geometry that produced the packed sequence, and every
        // index below would be quietly wrong.
        guard f * h * w == videoSpan.count,
              videoSpan.upperBound == sequenceLength,
              videoSpan.lowerBound >= 0 else { return nil }
        self.prefix = 0 ..< videoSpan.lowerBound
        self.videoSpan = videoSpan
        self.frames = f
        self.height = h
        self.width = w

        let n = Swift.max(2, Swift.min(landmarks, f))
        if n >= f {
            self.landmarkFrames = Array(0 ..< f)
        } else {
            // Evenly spaced, endpoints included. Deduplicated because at small
            // frame counts the spacing collapses and a repeated landmark would
            // silently narrow the set while looking like it had not.
            var set = Set<Int>()
            for i in 0 ..< n {
                set.insert(Int((Double(i) * Double(f - 1) / Double(n - 1)).rounded()))
            }
            self.landmarkFrames = set.sorted()
        }
    }

    /// The frame and spatial position of a video token.
    package func locate(_ token: Int) -> (frame: Int, y: Int, x: Int)? {
        guard videoSpan.contains(token) else { return nil }
        let v = token - videoSpan.lowerBound
        let per = tokensPerFrame
        return (v / per, (v % per) / width, v % width)
    }

    /// Whether `query` is allowed to attend `key`.
    ///
    /// Reference semantics, one pair at a time. The oracle builds masks in bulk
    /// and this is what those masks are checked against — a bulk construction
    /// that is wrong in a corner is exactly the bug a per-pair definition
    /// catches.
    package func allows(query: Int, key: Int) -> Bool {
        // Prefix queries see everything.
        if prefix.contains(query) { return true }
        // Every query keeps the whole prefix.
        if prefix.contains(key) { return true }
        guard let q = locate(query), let k = locate(key) else { return false }
        if q.frame == k.frame { return true }                 // same frame
        if q.y == k.y && q.x == k.x { return true }           // same position
        return landmarkFrames.contains(k.frame)               // landmark frame
    }

    /// Fraction of the full `[S, S]` attention this topology admits.
    ///
    /// Counted rather than estimated, because the three video axes overlap and
    /// an estimate that adds them would overstate the density — which is the
    /// direction that makes a method look worse than it is, and the direction a
    /// sanity check is least likely to catch.
    package var density: Double {
        let s = videoSpan.upperBound
        let p = prefix.count
        let per = tokensPerFrame
        let landmarks = Set(landmarkFrames)

        // Counted by frame, which makes the overlaps disappear instead of
        // needing inclusion-exclusion. For a query in frame `f0`, look at each
        // frame `g` of the lattice:
        //
        //   g == f0        every token — the same-frame axis
        //   g is landmark  every token — the landmark axis
        //   otherwise      exactly one token, the query's own spatial position
        //
        // The three axes overlap heavily and an earlier version added them and
        // subtracted an estimate of the overlap, which came out 4.2% wrong
        // against a counted mask. This form has nothing to estimate.
        var videoAllowed = 0
        for f0 in 0 ..< frames {
            let full = landmarks.contains(f0) ? landmarks.count : landmarks.count + 1
            let perQuery = per * full + (frames - full)
            videoAllowed += per * (p + perQuery)     // `per` queries in this frame
        }
        let total = Double(p) * Double(s) + Double(videoAllowed)
        return total / (Double(s) * Double(s))
    }
}
