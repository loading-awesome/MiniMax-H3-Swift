// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
@testable import H3Foundation

/// The measured control's deltas, replayed through the policy at other settings.
///
/// **Why this exists.** The first reading of the control trace said three steps
/// were being refused by the consecutive cap, and concluded that raising the cap
/// would remove three full steps and buy about 26%. That is wrong, and wrong in
/// a way no amount of staring at the trace reveals: the skip counter **resets at
/// every refusal**, so raising the cap does not delete refresh points, it moves
/// them. Replayed properly, caps 3 and 4 do exactly the same amount of work.
///
/// These are offline projections against fixed deltas. A different cap changes
/// the trajectory, which changes every subsequent delta, so a rendered arm
/// remains the only authority. What the projection is good for is deciding
/// which arms are worth an hour of GPU each — and it says cap 4 is a coherence
/// experiment rather than a speed one.
@Suite("step cache policy replay")
struct StepCachePolicyReplayTests {

    /// Median deltas over the three `control-cached` runs of 2026-08-06,
    /// 864x480x124, 20 steps, seed 7. Step 0 has no predecessor.
    /// `(whole, video, audio)`.
    static let measured: [Int: (whole: Double, video: Double, audio: Double)] = [
        1: (0.216, 0.237, 0.120), 2: (0.137, 0.147, 0.112), 3: (0.115, 0.124, 0.096),
        4: (0.096, 0.101, 0.079), 5: (0.082, 0.087, 0.072), 6: (0.071, 0.075, 0.065),
        7: (0.065, 0.067, 0.056), 8: (0.062, 0.065, 0.055), 9: (0.060, 0.063, 0.059),
        10: (0.059, 0.061, 0.060), 11: (0.062, 0.064, 0.065), 12: (0.067, 0.070, 0.067),
        13: (0.067, 0.070, 0.074), 14: (0.076, 0.078, 0.089), 15: (0.087, 0.090, 0.108),
        16: (0.106, 0.109, 0.120), 17: (0.128, 0.130, 0.157), 18: (0.159, 0.163, 0.196),
        19: (0.250, 0.255, 0.246)
    ]

    static let steps = 20
    static let fullStepSeconds = 58.84
    static let reusedStepSeconds = 1.25

    /// Runs the real policy over the measured deltas, threading the skip
    /// counter exactly as `H3StepCache` does.
    static func replay(cap: Int, threshold: Double = 0.10)
        -> (reused: [Int], refreshedByCap: [Int], audioOnly: [Int]) {
        let policy = StepCachePolicy(threshold: threshold, maxConsecutiveSkips: cap)
        var skips = 0
        var reused: [Int] = [], byCap: [Int] = [], audioOnly: [Int] = []
        for i in 0 ..< steps {
            let d = measured[i]
            let v = policy.explain(
                wholeSequenceChange: d?.whole ?? .infinity,
                audioChange: d?.audio ?? .infinity,
                videoChange: d?.video,
                step: i, totalSteps: steps, consecutiveSkips: skips,
                haveCachedResidual: i > 0)
            if v.decision == .reuse {
                reused.append(i); skips += 1
            } else {
                skips = 0
                if v.constraints.contains(.consecutiveCap) { byCap.append(i) }
                // The case the headline hides: audio is the sole objection.
                if v.constraints == [.audioAboveThreshold] { audioOnly.append(i) }
            }
        }
        return (reused, byCap, audioOnly)
    }

    static func samplingSeconds(reused: Int) -> Double {
        Double(reused) * reusedStepSeconds + Double(steps - reused) * fullStepSeconds
    }

    @Test("the replay reproduces the run it came from")
    func replayMatchesTheObservedRun() {
        // Nine reuses at 45% of twenty branch-steps, refreshed at 7, 11 and 15.
        // If this ever fails, the projections below are describing a policy
        // that no longer exists.
        let r = Self.replay(cap: 3)
        #expect(r.reused == [4, 5, 6, 8, 9, 10, 12, 13, 14])
        #expect(r.refreshedByCap == [7, 11, 15])
    }

    @Test("raising the cap to 4 moves the refresh points and buys nothing")
    func capFourIsNotASpeedExperiment() {
        // The correction that matters. The counter resets at every refusal, so
        // a larger cap relocates the refreshes rather than removing them:
        // 7/11/15 becomes 8/13, and step 15 is then refused by audio instead.
        // Same nine reuses, same wall clock.
        let three = Self.replay(cap: 3), four = Self.replay(cap: 4)
        #expect(four.reused.count == three.reused.count)
        #expect(four.refreshedByCap == [8, 13])
        #expect(Self.samplingSeconds(reused: four.reused.count)
                == Self.samplingSeconds(reused: three.reused.count))
    }

    @Test("caps 5 and 6 buy one step, and the same one")
    func capsFiveAndSix() {
        // ~9.6% of sampling, not the 26% first claimed. Five and six differ
        // only in where the single refresh lands, so running both is a test of
        // refresh placement, not of speed.
        for cap in [5, 6] {
            let r = Self.replay(cap: cap)
            #expect(r.reused.count == 10, "cap \(cap)")
            let gain = Self.samplingSeconds(reused: 9) / Self.samplingSeconds(reused: 10)
            #expect(abs(gain - 1.096) < 0.005)
        }
    }

    @Test("unbounded reuse is bounded by the audio probe, not by nothing")
    func unboundedStopsAtStepFifteen() {
        // Worth stating plainly: with no cap at all the cache does not run to
        // the end of the schedule. It reuses steps 4 through 14 and is then
        // stopped by audio at step 15, for eleven reuses and about 21%. The
        // audio probe is the only thing standing between the cache and the
        // late high-change region once the cap is relaxed — which is the
        // argument for keeping it, and it is invisible at cap 3.
        let r = Self.replay(cap: 99)
        #expect(r.reused == Array(4 ... 14))
        #expect(r.refreshedByCap.isEmpty)
        #expect(r.audioOnly == [15])
        let gain = Self.samplingSeconds(reused: 9) / Self.samplingSeconds(reused: 11)
        #expect(abs(gain - 1.212) < 0.005)
    }

    @Test("at cap 3 the audio probe never appears to do anything")
    func audioVetoIsMaskedByTheCap() {
        // The reason `constraints` had to be recorded. At cap 3 the sole-audio
        // objection at step 15 is masked, because the cap is also active there
        // and the headline reports one thing. A sweep reading only the headline
        // would conclude the audio probe was inert and drop it, immediately
        // before the change that makes it the binding constraint.
        //
        // It first appears at cap 6, not cap 5: at cap 5 the counter happens to
        // reach the ceiling at step 15 as well, so the cap and audio object
        // together there. Which cap exposes it is an accident of where the
        // refreshes land, and that is exactly the sort of thing worth having a
        // replay for rather than reasoning about.
        #expect(Self.replay(cap: 3).audioOnly.isEmpty)
        #expect(Self.replay(cap: 5).audioOnly.isEmpty)
        #expect(Self.replay(cap: 6).audioOnly == [15])
    }

    @Test("the video probe would disagree exactly once, and it costs a step")
    func videoAdvisoryDisagreesAtStepFour() {
        // "Video never disagrees with whole-sequence" was too strong. At step 4
        // whole is 0.096 and video is 0.101, so a video vote would refuse a step
        // the current policy reuses. One crossing in twenty steps, in the
        // direction of doing more work — which is the whole empirical case for
        // giving this probe a vote, and it is not a speed argument.
        let policy = StepCachePolicy(threshold: 0.10)
        var disagreements: [Int] = []
        for (i, d) in Self.measured.sorted(by: { $0.key < $1.key }) {
            let v = policy.explain(wholeSequenceChange: d.whole, audioChange: d.audio,
                                   videoChange: d.video, step: i, totalSteps: Self.steps,
                                   consecutiveSkips: 0, haveCachedResidual: true)
            if v.advisory.contains(.videoAboveThreshold),
               !v.constraints.contains(.wholeSequenceAboveThreshold) {
                disagreements.append(i)
            }
        }
        #expect(disagreements == [4])
    }

    @Test("late refusals are not audio-only, whatever the headline says")
    func lateRefusalsAreBothStreams() {
        // Steps 16-18 report `audioAboveThreshold` because audio is the larger
        // number, but the whole-sequence probe is over threshold there too and
        // would have forced a full step by itself. Reading the headline as
        // "audio vetoed" overstates what the per-stream probe contributes.
        let policy = StepCachePolicy(threshold: 0.10)
        for i in [16, 17, 18] {
            let d = Self.measured[i]!
            let v = policy.explain(wholeSequenceChange: d.whole, audioChange: d.audio,
                                   videoChange: d.video, step: i, totalSteps: Self.steps,
                                   consecutiveSkips: 0, haveCachedResidual: true)
            #expect(v.reason == .audioAboveThreshold)
            #expect(v.constraints.contains(.wholeSequenceAboveThreshold),
                    "step \(i) would have been refused without the audio probe")
        }
    }
}
