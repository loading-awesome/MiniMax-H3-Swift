// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// The flow-matching sigma schedule.
///
///     sigma(t) = shift * t / (1 + (shift - 1) * t),   t_i = 1 - i/steps
///
/// With `shift = 12` this is heavily back-loaded: 15 of 20 steps cover sigma
/// 1.0 -> 0.8, and the last step leaps 0.387 -> 0. **That is correct, not a
/// bug.** It matches the reference exactly at every step, and "fixing" it to a
/// linear schedule is a classic and expensive mistake.
package struct FlowSchedule: Sendable {
    package let shift: Double
    package init(shift: Double = H3Shift.video) { self.shift = shift }

    package func sigma(t: Double) -> Double {
        shift * t / (1.0 + (shift - 1.0) * t)
    }

    /// `steps + 1` values, from 1.0 down to exactly 0.0.
    package func sigmas(steps: Int) -> [Double] {
        precondition(steps > 0, "steps must be positive")
        return (0...steps).map { i in
            i == steps ? 0.0 : sigma(t: 1.0 - Double(i) / Double(steps))
        }
    }
}

/// A distilled model's own denoising steps, which are not a schedule at all.
///
/// A DMD-distilled checkpoint is trained to jump between specific noise levels
/// rather than to follow a curve, so its steps are a property of the weights.
/// FastVideo's FastH3 ships them in `fastvideo_inference.json` as diffusers
/// timesteps out of 1000 — `[999, 749, 500, 250]` for four transformer
/// forwards — which are already-shifted sigmas and are used directly.
///
/// Reading them as base-grid `t` instead would give `[1.0, 0.973, 0.923, 0.80]`
/// through `sigma(t)`, which never approaches zero and cannot be a four-step
/// denoise. That is the check if a distilled render comes out as noise.
package struct DistilledSchedule: Sendable {
    package let sigmas: [Double]

    /// - Parameter timesteps: the checkpoint's denoising steps, out of 1000.
    ///   These are points on the **unshifted** grid, so the stream's shift is
    ///   applied here exactly as `FlowSchedule` does for the base model.
    ///
    /// An earlier version used them as sigmas directly, on the reasoning that
    /// `[1.0, 0.973, 0.923, 0.80]` "never approaches zero and cannot be a
    /// four-step denoise". That reasoning was wrong, and this file says why one
    /// line up: with shift 12 the schedule *is* back-loaded, 15 of 20 steps
    /// covering 1.0 to 0.8 with the last leaping to zero. Staying high and then
    /// dropping is the normal shape here, not a broken one.
    package init(timesteps: [Int], shift: Double = H3Shift.video) {
        precondition(!timesteps.isEmpty, "a distilled schedule needs at least one step")
        let schedule = FlowSchedule(shift: shift)
        self.sigmas = timesteps.map { schedule.sigma(t: Double($0) / 1000.0) } + [0.0]
    }

    /// FastH3 4-step, from `fastvideo_inference.json`.
    package static let fastH3FourStep = DistilledSchedule(
        timesteps: [999, 749, 500, 250])

}

/// Mapping one stream's sigma onto another stream's shifted schedule.
///
/// The sampler only ever sees the **video** sigma. The audio stream runs on
/// shift 3 while video runs on shift 12, so the DiT maps the video sigma back
/// to the unshifted grid and re-applies the audio shift in closed form. The two
/// streams therefore denoise at genuinely different rates inside a single
/// forward pass — this is the reason there is a timestep row at all.
package enum SigmaMap {
    /// Invert `sigma = s*b/(1+(s-1)*b)` to the base grid, re-apply `to`.
    package static func shift(_ sigma: Float, from: Float, to: Float) -> Float {
        let base = sigma / (from + sigma * (1.0 - from))
        return to * base / (1.0 + (to - 1.0) * base)
    }

    /// `d(sigma_to)/d(sigma_from)` at the same base-grid point.
    ///
    /// The sampler integrates the flat ODE `dX/dsigma_v = (X - denoised)/sigma_v`
    /// for both streams. Scaling the audio velocity by this slope is what makes
    /// that shared ODE equal the audio stream's true ODE on its own schedule.
    /// Drop it and audio drifts out of step with video over the sampling run.
    package static func slope(_ sigma: Float, from: Float, to: Float) -> Float {
        let base = sigma / (from + sigma * (1.0 - from))
        let num = to * pow(1.0 + (from - 1.0) * base, 2)
        let den = from * pow(1.0 + (to - 1.0) * base, 2)
        return num / den
    }
}

/// Which AdaLN timestep row each stream uses for one forward pass.
///
/// The reference collects the distinct timesteps into a **sorted set** and
/// indexes AdaLN by position in that set. When video and audio land on the same
/// timestep — which happens at sigma 1.0, where both are exactly 0 — the set
/// collapses to one entry and both streams share row 0. The single-step goldens
/// only ever exercise that collapsed case, so a port that hard-codes two rows
/// passes L1 and then fails on any multi-step run.
///
/// Arithmetic is float32 on purpose: the reference derives these from an fp32
/// sigma tensor, and the deduplication is exact equality on those fp32 values.
package struct TimestepPlan: Sendable, Equatable {
    /// Distinct timestep values, ascending — the rows of `t_emb`.
    package let values: [Float]
    private let segRowMap: [SegmentKind: Int]
    /// `d(sigma_a)/d(sigma_v)`, which the audio velocity must be scaled by.
    package let audioSlope: Float
    /// The video sigma this plan was built from, retained rather than recovered.
    /// `values` holds timesteps, and several distinct sigmas map to the same
    /// deduplicated timestep set, so the plan cannot be run backwards to the
    /// sigma that produced it.
    package let sigmaVideo: Float

    package init(sigmaVideo: Double,
                segments: [PackedSegment] = [],
                visualCondNoiseAug: Float = 0.999,
                audioCondNoiseAug: Float = 1.0,
                shiftVideo: Double = H3Shift.video,
                shiftAudio: Double = H3Shift.audio) {
        let sv = max(Float(sigmaVideo), 1e-6)
        let fv = Float(shiftVideo), fa = Float(shiftAudio)
        let tv = 1.0 - sv
        let ta = 1.0 - SigmaMap.shift(sv, from: fv, to: fa)

        let actualSegments = segments.isEmpty ? [
            PackedSegment(start: 0, stop: 1, kind: .text),
            PackedSegment(start: 1, stop: 2, kind: .audio),
            PackedSegment(start: 2, stop: 3, kind: .video)
        ] : segments

        let hasVisCond = actualSegments.contains { $0.kind == .cond || $0.kind == .refImage }
        let hasAudCond = actualSegments.contains { $0.kind == .refAudio }

        let tCond = max(tv, visualCondNoiseAug)
        let tRefAudio = max(ta, audioCondNoiseAug)

        var uniqueSet = Set<Float>([tv, ta])
        if hasVisCond { uniqueSet.insert(tCond) }
        if hasAudCond { uniqueSet.insert(tRefAudio) }

        let sortedUnique = uniqueSet.sorted()
        self.values = sortedUnique

        var rowMap: [SegmentKind: Int] = [:]
        rowMap[.text] = sortedUnique.firstIndex(of: tv)!
        rowMap[.video] = sortedUnique.firstIndex(of: tv)!
        rowMap[.audio] = sortedUnique.firstIndex(of: ta)!
        rowMap[.cond] = sortedUnique.firstIndex(of: tCond) ?? sortedUnique.firstIndex(of: tv)!
        rowMap[.refImage] = sortedUnique.firstIndex(of: tCond) ?? sortedUnique.firstIndex(of: tv)!
        rowMap[.refAudio] = sortedUnique.firstIndex(of: tRefAudio) ?? sortedUnique.firstIndex(of: ta)!

        self.segRowMap = rowMap
        self.audioSlope = SigmaMap.slope(sv, from: fv, to: fa)
        self.sigmaVideo = sv
    }

    package func row(for kind: SegmentKind) -> Int {
        segRowMap[kind]!
    }

    package var videoRow: Int { row(for: .video) }
    package var audioRow: Int { row(for: .audio) }
}
