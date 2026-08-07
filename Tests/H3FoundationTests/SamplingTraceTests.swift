// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
@testable import H3Foundation

/// The evidence trail, checked without a GPU.
///
/// Every claim the later phases make rests on these numbers being computed the
/// way the doc comments say. A median that quietly included infinities, or a CSV
/// that wrote `inf` where a reader coerces it to zero, would not fail anything —
/// it would produce a plausible table with the wrong numbers in it.
@Suite("sampling trace")
struct SamplingTraceTests {

    private func step(_ i: Int, whole: Double = 0.05, video: Double = 0.05,
                      audio: Double = 0.05, decision: StepCachePolicy.Decision = .reuse,
                      reason: StepCachePolicy.Reason = .belowThreshold,
                      branch: StepTrace.Branch = .conditional) -> StepTrace {
        StepTrace(step: i, branch: branch, sigma: 1.0 - Double(i) / 20,
                  wholeSequenceChange: whole, videoChange: video, audioChange: audio,
                  decision: decision, reason: reason, consecutiveSkipsBefore: 0)
    }

    @Test("the median ignores the infinities every render starts with")
    func medianSkipsNonFinite() {
        // Step 0 has no predecessor, so its change is infinite by construction.
        // A mean would be infinite; a median that sorted infinities to the end
        // and indexed the middle would be wrong by however many there are.
        #expect(SamplingTrace.median([.infinity, 0.1, 0.2, 0.3]) == 0.2)
        #expect(SamplingTrace.median([.nan, 1.0]) == 1.0)
        #expect(SamplingTrace.median([]).isNaN)
        // Even count averages the middle pair rather than picking the upper.
        #expect(SamplingTrace.median([1.0, 2.0, 3.0, 4.0]) == 2.5)
    }

    @Test("the comparison figure survives a bimodal cached render")
    func meanNotMedianUnderACache() {
        // **The regression that cost a control sweep.** These are the real step
        // times from the first control run, rounded: twelve full steps at
        // ~59.9 s and eight reused at ~1.26 s, with nothing between them. The
        // median lands inside the full-step population and reports 59.9 —
        // indistinguishable from what dense would report — so a cache saving
        // 40% of the sampling time came out at a speed-up of about 1.00x.
        //
        // The mean is what wall clock is made of.
        let full = [Double](repeating: 59.9, count: 12)
        let reused = [Double](repeating: 1.26, count: 8)
        let trace = SamplingTrace(steps: [], stepSeconds: full + reused)
        #expect(abs(trace.medianStepSeconds - 59.9) < 1e-9)
        #expect(abs(trace.meanStepSeconds - 36.44) < 0.01)
        // The gap between them is the whole point: if these two ever agree on a
        // cached render, the cache is not skipping anything.
        #expect(trace.medianStepSeconds > trace.meanStepSeconds * 1.5)
    }

    @Test("full and reused step costs are reported separately")
    func perPopulationStepCosts() {
        // A kernel optimisation makes full steps cheaper; a looser threshold
        // skips more of them. Both move the mean, and only the first is a
        // speed-up that composes with anything else.
        let trace = SamplingTrace(
            steps: [step(0, decision: .runFull, reason: .noHistory),
                    step(1, decision: .reuse),
                    step(2, decision: .reuse),
                    step(3, decision: .runFull, reason: .consecutiveCap)],
            stepSeconds: [60, 1.2, 1.3, 62])
        #expect(trace.medianFullStepSeconds == 61)
        #expect(abs(trace.medianReusedStepSeconds - 1.25) < 1e-9)
    }

    @Test("a cache-free run reports every step as a full step")
    func denseRunHasNoDecisionsButAllFullSteps() {
        // A dense render records no decisions because there is no cache to make
        // any. Reading that empty list as "no full steps" reported `nan` for
        // the dense control — the one arm whose full-step cost a kernel change
        // most needs to be measured against.
        let dense = SamplingTrace(steps: [], stepSeconds: [60, 58, 59, 61])
        #expect(dense.medianFullStepSeconds == 59.5)
        #expect(dense.stepsSkipped == 0)
    }

    @Test("only one CFG branch contributes to the per-population step cost")
    func branchesDoNotDoubleCountWallClock() {
        // A step's wall clock covers both forwards. Counting it once per branch
        // would report each full step twice and leave the median unchanged but
        // the reasoning wrong — and it would break the moment the two branches
        // disagreed about a step.
        let trace = SamplingTrace(
            steps: [step(0, decision: .runFull, reason: .noHistory, branch: .conditional),
                    step(0, decision: .reuse, branch: .unconditional)],
            stepSeconds: [60])
        #expect(trace.medianFullStepSeconds == 60)
        #expect(trace.medianReusedStepSeconds.isNaN)
    }

    @Test("refusals are attributed, so a cap cannot masquerade as a threshold")
    func reasonCounts() {
        // Two configurations can skip the same number of steps for completely
        // different reasons. This is the field that tells them apart.
        let trace = SamplingTrace(steps: [
            step(0, decision: .runFull, reason: .noHistory),
            step(1), step(2), step(3),
            step(4, decision: .runFull, reason: .consecutiveCap),
            step(5, audio: 0.9, decision: .runFull, reason: .audioAboveThreshold),
            step(6, decision: .runFull, reason: .cooldown)
        ])
        #expect(trace.stepsSkipped == 3)
        #expect(trace.stepsRun == 4)
        #expect(trace.reasonCounts[.consecutiveCap] == 1)
        #expect(trace.reasonCounts[.audioAboveThreshold] == 1)
        #expect(trace.reasonCounts[.belowThreshold] == 3)
    }

    @Test("both CFG branches are kept as separate rows")
    func branchesNotMerged() {
        // The two forwards hold separate caches and can disagree about every
        // step. Merging them would average away the case where only one branch
        // is skipping, which is a real configuration and not a healthy one.
        let trace = SamplingTrace(steps: [
            step(3, decision: .reuse, branch: .conditional),
            step(3, decision: .runFull, reason: .wholeSequenceAboveThreshold, branch: .unconditional)
        ])
        #expect(trace.steps.count == 2)
        #expect(trace.stepsSkipped == 1 && trace.stepsRun == 1)
    }

    @Test("non-finite values are written as empty CSV fields, never as inf or 0")
    func csvOmitsNonFinite() {
        // A reader that coerces `inf` to 0 would show the first step of every
        // render as the quietest one — an artefact that looks exactly like a
        // finding.
        let trace = SamplingTrace(
            steps: [step(0, whole: .infinity, video: .infinity, audio: .infinity,
                         decision: .runFull, reason: .noHistory)],
            stepSeconds: [4.5])
        let lines = trace.csv.split(separator: "\n")
        #expect(lines.count == 2)
        let fields = lines[1].split(separator: ",", omittingEmptySubsequences: false)
        #expect(fields[3].isEmpty && fields[4].isEmpty && fields[5].isEmpty)
        #expect(fields[6] == "full")
        #expect(fields[7] == "noHistory")
        #expect(fields[11] == "4.500000")
        #expect(!trace.csv.contains("inf") && !trace.csv.contains("nan"))
    }

    @Test("a step trace round-trips through JSON")
    func codableRoundTrip() throws {
        // The records outlive the session that wrote them; that is their only
        // purpose. A decision encoded as an unreadable case name, or a field
        // that fails to decode, makes the whole archive worthless.
        let trace = SamplingTrace(steps: [step(1), step(2, decision: .runFull,
                                                reason: .wholeSequenceAboveThreshold)],
                                  stepSeconds: [1, 2, 3])
        let data = try JSONEncoder().encode(trace)
        #expect(String(decoding: data, as: UTF8.self).contains("\"reused\""))
        #expect(try JSONDecoder().decode(SamplingTrace.self, from: data) == trace)
    }

    @Test("the infinite first step encodes, as null")
    func infiniteStepEncodes() throws {
        // **This is why the first control sweep produced no benchmark records
        // at all.** `JSONEncoder` throws `invalidValue` on any non-finite
        // Double, and step 0 of every render has no predecessor and therefore
        // an infinite change by construction — so every single record failed to
        // write. The CSV survived, because it handles non-finite explicitly,
        // and the receipt dutifully recorded "benchmark record not written".
        //
        // The previous round-trip test used finite values throughout, which is
        // exactly the fixture that cannot catch this.
        let trace = SamplingTrace(
            steps: [step(0, whole: .infinity, video: .infinity, audio: .infinity,
                         decision: .runFull, reason: .noHistory)],
            stepSeconds: [4.5])
        let data = try JSONEncoder().encode(trace)
        let json = String(decoding: data, as: UTF8.self)
        #expect(json.contains("null"))
        #expect(!json.contains("inf"))
        // And infinity comes back as infinity, not as zero — a first step
        // decoding to a change of 0 would read as the quietest of the render.
        let back = try JSONDecoder().decode(SamplingTrace.self, from: data)
        #expect(back == trace)
        #expect(back.steps[0].wholeSequenceChange == .infinity)
    }

    @Test("a whole record with a real trace survives the encoder")
    func recordWithInfinitiesEncodes() throws {
        // The unit above covers the trace; this covers the thing actually
        // written to disk, because that is the path that failed.
        let trace = SamplingTrace(
            steps: [step(0, whole: .infinity, video: .infinity, audio: .infinity,
                         decision: .runFull, reason: .noHistory),
                    step(1)],
            stepSeconds: [60, 1.2])
        let record = BenchmarkRecord(
            arm: "control-cached",
            identity: .init(promptDigest: "abc", negativePromptDigest: nil, seed: 7,
                            width: 864, height: 480, seconds: 5, steps: 20, cfgScale: 1,
                            ditDigest: nil, ditSizeBytes: 0),
            configuration: .init(qualityProfile: "balanced", attentionBackend: "sdpa",
                                 cacheThreshold: 0.1, cacheMaxConsecutiveSkips: 3,
                                 cachePerStreamProbe: true),
            machine: .init(model: "m", cores: 1, physicalMemoryBytes: 1,
                           operatingSystem: "x", libraryVersion: "y", mlxSwiftVersion: "z"),
            phaseSeconds: ["total": 100],
            memory: .init(mlxPeakBytes: 1, mlxActiveBytesAtEnd: 1),
            trace: trace)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(record)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        #expect(try decoder.decode(BenchmarkRecord.self, from: data).trace == trace)
    }
}

/// The reasons behind the decisions, which the plain `decide` throws away.
@Suite("step cache policy reasons")
struct StepCachePolicyReasonTests {

    static let p = StepCachePolicy(threshold: 0.5, maxConsecutiveSkips: 3)

    @Test("explain never disagrees with decide")
    func explainAgreesWithDecide() {
        // `decide` is implemented as `explain(...).decision`, and this is what
        // keeps that true if either is ever edited separately.
        for change in [0.0, 0.1, 0.49, 0.5, 0.9, Double.infinity] {
            for audio in [nil, 0.1, 0.9] as [Double?] {
                for step in [0, 1, 5, 18, 19] {
                    for skips in [0, 3] {
                        for history in [true, false] {
                            let v = Self.p.explain(wholeSequenceChange: change, audioChange: audio,
                                                   step: step, totalSteps: 20,
                                                   consecutiveSkips: skips,
                                                   haveCachedResidual: history)
                            let d = Self.p.decide(wholeSequenceChange: change, audioChange: audio,
                                                  step: step, totalSteps: 20,
                                                  consecutiveSkips: skips,
                                                  haveCachedResidual: history)
                            #expect(v.decision == d)
                            #expect((v.reason == .belowThreshold) == (d == .reuse))
                        }
                    }
                }
            }
        }
    }

    @Test("the structural rules are reported in priority order")
    func structuralReasons() {
        func reason(step: Int, skips: Int, history: Bool) -> StepCachePolicy.Reason {
            Self.p.explain(wholeSequenceChange: 0.01, audioChange: nil, step: step,
                           totalSteps: 20, consecutiveSkips: skips,
                           haveCachedResidual: history).reason
        }
        #expect(reason(step: 5, skips: 0, history: false) == .noHistory)
        #expect(reason(step: 0, skips: 0, history: true) == .warmup)
        #expect(reason(step: 19, skips: 0, history: true) == .cooldown)
        #expect(reason(step: 5, skips: 3, history: true) == .consecutiveCap)
        #expect(reason(step: 5, skips: 0, history: true) == .belowThreshold)
    }

    @Test("the binding stream is named")
    func bindingStream() {
        let perStream = StepCachePolicy(threshold: 0.5, perStream: true)
        func reason(_ whole: Double, _ audio: Double?) -> StepCachePolicy.Reason {
            perStream.explain(wholeSequenceChange: whole, audioChange: audio, step: 5,
                              totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true).reason
        }
        // Audio 2.6% of the rows, moving 18x more than the average: the veto
        // that only a per-stream probe casts, and it must be attributed to
        // audio or a sweep will read it as a video-driven refusal.
        #expect(reason(0.05, 0.9) == .audioAboveThreshold)
        #expect(reason(0.9, 0.05) == .wholeSequenceAboveThreshold)
        // The whole-sequence arm has no audio input at all, so every refusal
        // there is a video one by construction.
        let whole = StepCachePolicy(threshold: 0.5, perStream: false)
        #expect(whole.explain(wholeSequenceChange: 0.9, audioChange: 0.9, step: 5,
                              totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true).reason == .wholeSequenceAboveThreshold)
    }

    @Test("non-finite is its own reason, not an above-threshold refusal")
    func nonFiniteReason() {
        // These are different events. Infinity means the comparison could not
        // be made; above-threshold means it was made and failed. A sweep that
        // conflated them would count every render's first step as evidence
        // that its threshold was too tight.
        #expect(Self.p.explain(wholeSequenceChange: .infinity, audioChange: nil, step: 5,
                               totalSteps: 20, consecutiveSkips: 0,
                               haveCachedResidual: true).reason == .nonFinite)
        #expect(Self.p.explain(wholeSequenceChange: .nan, audioChange: nil, step: 5,
                               totalSteps: 20, consecutiveSkips: 0,
                               haveCachedResidual: true).reason == .nonFinite)
    }
}
