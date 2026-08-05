// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
@testable import H3Foundation

/// The cache's decision rules, each of which fails silently if broken.
///
/// No MLX, no GPU, no checkpoint — the whole suite is scalar arithmetic, which
/// is the point of separating the policy from the measurement.
@Suite("step cache policy")
struct StepCachePolicyTests {

    static let p = StepCachePolicy(threshold: 0.5)
    /// A change well under the threshold, so only the structural rules can
    /// force a full step.
    static let quiet = 0.01

    @Test("nothing is reused before a full step has been recorded")
    func needsHistory() {
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 5, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: false) == .runFull)
    }

    @Test("the first step always runs in full")
    func warmup() {
        // Step 0 has no previous residual to compare against.
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 0, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .runFull)
    }

    @Test("the last step always runs in full")
    func cooldown() {
        // Its residual lands directly in the decoded pixels.
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 19, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .runFull)
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 18, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .reuse)
    }

    @Test("consecutive reuse is capped")
    func consecutiveCap() {
        // A skipped step does not refresh the probe, so without a ceiling the
        // same comparison recurs and the sampler coasts on one stale delta.
        for n in 0 ..< 3 {
            #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                                  step: 5, totalSteps: 20, consecutiveSkips: n,
                                  haveCachedResidual: true) == .reuse)
        }
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 5, totalSteps: 20, consecutiveSkips: 3,
                              haveCachedResidual: true) == .runFull)
    }

    @Test("the per-stream probe lets the audio veto a skip")
    func audioVetoes() {
        // The packed sequence is 95% video and 2.6% audio. Video quiet, audio
        // moving: a per-stream policy must refuse.
        let perStream = StepCachePolicy(threshold: 0.5, perStream: true)
        #expect(perStream.decide(wholeSequenceChange: 0.01, audioChange: 0.9,
                                 step: 5, totalSteps: 20, consecutiveSkips: 0,
                                 haveCachedResidual: true) == .runFull)
        // and does not veto when the audio is also quiet
        #expect(perStream.decide(wholeSequenceChange: 0.01, audioChange: 0.02,
                                 step: 5, totalSteps: 20, consecutiveSkips: 0,
                                 haveCachedResidual: true) == .reuse)
    }

    @Test("the whole-sequence probe misses exactly what the per-stream probe catches")
    func wholeSequenceMisses() {
        // Same inputs, same threshold, only the probe differs — the A/B from
        // docs/ACCELERATION.md, asserted. Measured on a real render, the audio
        // residual moves 32% more per step than the whole-sequence average
        // (0.079 against 0.060), and the lip-sync margin fell from +0.718 to
        // +0.453 when this arm was used.
        let whole = StepCachePolicy(threshold: 0.5, perStream: false)
        #expect(whole.decide(wholeSequenceChange: 0.01, audioChange: 0.9,
                             step: 5, totalSteps: 20, consecutiveSkips: 0,
                             haveCachedResidual: true) == .reuse)
    }

    @Test("a non-finite change never reuses")
    func infiniteChangeRuns() {
        // The first comparison of a render has no predecessor and reports
        // infinity; that must not read as "below threshold".
        #expect(Self.p.decide(wholeSequenceChange: .infinity, audioChange: nil,
                              step: 5, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .runFull)
        #expect(Self.p.decide(wholeSequenceChange: .nan, audioChange: nil,
                              step: 5, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .runFull)
    }

    @Test("a threshold of zero disables reuse entirely")
    func zeroThresholdNeverReuses() {
        let off = StepCachePolicy(threshold: 0)
        #expect(off.decide(wholeSequenceChange: 0, audioChange: nil,
                           step: 5, totalSteps: 20, consecutiveSkips: 0,
                           haveCachedResidual: true) == .runFull)
    }
}
