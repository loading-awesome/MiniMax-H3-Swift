// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Latent shapes and the packed-sequence layout.
///
/// These are the first things a port must get right and the cheapest to verify:
/// every value here is checked against the CUDA reference in `H3CoreTests`.
/// A shape bug shows up at L1 as a SHAPE verdict rather than a value mismatch,
/// which is the easiest possible failure to diagnose — get these green first.
package struct LatentGeometry: Sendable, Equatable {
    package let frameCount: Int
    package let latentT: Int
    package let audioT: Int
    package let latentH: Int
    package let latentW: Int
    package let config: H3Config

    /// Snap a requested frame count up onto the 17k+5 lattice, minimum 5.
    package static func alignFrameCount(_ length: Int) -> Int {
        let n = max(H3Video.latticeOffset, length)
        let k = Int((Double(n - H3Video.latticeOffset) / Double(H3Video.lattice)).rounded(.up))
        return H3Video.latticeOffset + k * H3Video.lattice
    }

    /// `latentT = 5(F-5)/17 + 2`, `audioT = round(F/24 * 40)`.
    /// Verified against the reference at F = 5, 22, 39, 73, 124, 141, 362.
    package init(width: Int, height: Int, length: Int, config: H3Config = H3Config()) {
        let f = Self.alignFrameCount(length)
        self.frameCount = f
        self.latentT = 5 * (f - H3Video.latticeOffset) / H3Video.lattice + 2
        self.audioT = Int((Double(f) / Double(H3Video.fps) * Double(H3Audio.latentFPS)).rounded())
        self.latentH = height / H3Video.spatialDownsample
        self.latentW = width / H3Video.spatialDownsample
        self.config = config
    }

    /// `[B, 24, latentT, H/16, W/16]`
    package func videoLatentShape(batch: Int = 1) -> [Int] {
        [batch, config.videoLatentDim, latentT, latentH, latentW]
    }

    /// `[B, 32, 2, audioT]` — the 2 is not a batch or channel axis; it is part
    /// of the audio latent's own layout.
    package func audioLatentShape(batch: Int = 1) -> [Int] {
        [batch, config.audioLatentDim, 2, audioT]
    }

    /// Video tokens after patchify: `latentT * (H/2) * (W/2)`.
    package var videoTokens: Int {
        latentT * (latentH / config.patchSize[1]) * (latentW / config.patchSize[2])
    }

    /// Audio tokens: `2 * audioT`.
    package var audioTokens: Int { 2 * audioT }

    /// Whether the frame count sits inside the trained range. Renders outside
    /// it are numerically fine but must not be read as quality signals.
    package var inTrainedRange: Bool { H3Video.trainedFrameRange.contains(frameCount) }
}

// PackedLayout lives in PackedSequence.swift — it carries position ids and a
// segment table, which is more than "geometry".
