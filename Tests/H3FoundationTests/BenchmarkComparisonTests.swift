// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
@testable import H3Foundation

/// The table a promotion decision gets made from.
@Suite("benchmark comparison")
struct BenchmarkComparisonTests {

    private func record(_ arm: String, stepSeconds: [Double], cache: Double = 0,
                        seed: UInt64 = 7, machine: String = "Mac Studio M3 Ultra",
                        skipped: Int = 0, at: TimeInterval = 0,
                        overrides: [String: String] = [:]) -> BenchmarkRecord {
        let steps = (0 ..< max(skipped, 0)).map {
            StepTrace(step: $0, branch: .conditional, sigma: 1, wholeSequenceChange: 0.01,
                      videoChange: 0.01, audioChange: 0.01, decision: .reuse,
                      reason: .belowThreshold, consecutiveSkipsBefore: 0)
        }
        return BenchmarkRecord(
            arm: arm,
            identity: .init(promptDigest: BenchmarkRecord.digest("kite"),
                            negativePromptDigest: nil, seed: seed, width: 864, height: 480,
                            seconds: 5, steps: 20, cfgScale: 1.0, ditDigest: "abc",
                            ditSizeBytes: 66),
            configuration: .init(qualityProfile: "balanced", attentionBackend: "sdpa",
                                 cacheThreshold: cache, cacheMaxConsecutiveSkips: 3,
                                 cachePerStreamProbe: true, overrides: overrides),
            machine: .init(model: machine, cores: 32, physicalMemoryBytes: 275,
                           operatingSystem: "15.0", libraryVersion: "0.1.0-dev",
                           mlxSwiftVersion: "0.31.6"),
            phaseSeconds: ["total": 100],
            memory: .init(mlxPeakBytes: 53_000_000_000, mlxActiveBytesAtEnd: 0),
            trace: SamplingTrace(steps: steps, stepSeconds: stepSeconds),
            recordedAt: Date(timeIntervalSince1970: at))
    }

    @Test("the dense arm is chosen as the control when none is named")
    func defaultControlIsTheDenseArm() {
        // The only baseline that is not itself an approximation. Defaulting to
        // "first recorded" would happily measure one cache setting against
        // another and call the result a speed-up over dense.
        let c = BenchmarkComparison(records: [
            record("cached", stepSeconds: [20], cache: 0.1, at: 0),
            record("dense", stepSeconds: [40], cache: 0, at: 1)
        ])
        #expect(c.controlArm == "dense")
        #expect(c.rows.first { $0.arm == "cached" }?.speedup == 2.0)
    }

    @Test("repeats give a spread, and one run gives none")
    func repeatSpread() {
        // 38 to 42 around a median of 40 is a 10% spread, which is larger than
        // most of the gains this roadmap is chasing — exactly the comparison
        // the column exists to force.
        let c = BenchmarkComparison(records: [
            record("dense", stepSeconds: [38]), record("dense", stepSeconds: [40]),
            record("dense", stepSeconds: [42])
        ])
        let row = try! #require(c.rows.first)
        #expect(row.runs == 3)
        #expect(row.meanStepSeconds == 40)
        #expect(abs(row.repeatSpread - 0.1) < 1e-9)
        #expect(c.report.contains("a gain smaller than that is not a gain"))

        let single = BenchmarkComparison(records: [record("dense", stepSeconds: [40])])
        #expect(single.rows.first?.repeatSpread == 0)
        #expect(single.report.contains("no variance estimate"))
    }

    @Test("an incomparable arm is listed with its reason, never silently dropped")
    func refusalsAreVisible() {
        // The failure this whole contract exists to prevent: an arm that cannot
        // be lined up against the control must not vanish from the table,
        // because a five-row table read as five tested arms is how a wrong
        // conclusion travels.
        let c = BenchmarkComparison(records: [
            record("dense", stepSeconds: [40], cache: 0),
            record("cached", stepSeconds: [20], cache: 0.1, seed: 99)
        ])
        #expect(c.unusable == ["cached"])
        let row = try! #require(c.rows.first { $0.arm == "cached" })
        #expect(row.speedup == nil)
        #expect(row.refusals.contains { $0.hasPrefix("seed:") })
        // Present in the table, with a dash where the number would be.
        #expect(c.report.contains("cached"))
        #expect(c.report.contains("could not be compared"))
        #expect(c.report.contains("seed: 99 vs 7"))
    }

    @Test("a different machine is refused even when everything else lines up")
    func machineMismatchRefused() {
        let c = BenchmarkComparison(records: [
            record("dense", stepSeconds: [40], cache: 0),
            record("cached", stepSeconds: [10], cache: 0.1, machine: "MacBook Pro M4 Max")
        ])
        #expect(c.rows.first { $0.arm == "cached" }?.speedup == nil)
        #expect(c.report.contains("machine:"))
    }

    @Test("a knob that differs from the control is stated, not treated as a refusal")
    func configurationDriftIsReported() throws {
        // Configuration is what an experiment varies, so this must not block
        // the comparison — but an arm carrying an unmentioned difference is how
        // a gain gets credited to the wrong change. The number and the caveat
        // have to arrive together.
        let c = BenchmarkComparison(records: [
            record("dense", stepSeconds: [40], cache: 0,
                   overrides: ["resolved.fusedModulation": "off"]),
            record("fused", stepSeconds: [36], cache: 0,
                   overrides: ["resolved.fusedModulation": "on", "H3_SOL_BETA": "1.2"])
        ], control: "dense")
        let row = try #require(c.rows.first { $0.arm == "fused" })
        #expect(row.speedup != nil)              // still compared
        #expect(row.refusals.isEmpty)            // and not refused
        #expect(row.configurationDrift.count == 2)
        #expect(c.report.contains("resolved.fusedModulation: on vs control's off"))
        #expect(c.report.contains("H3_SOL_BETA: 1.2 vs control's unset"))
    }

    @Test("the reused fraction is pooled across every repeat and branch")
    func reusedFraction() {
        let c = BenchmarkComparison(records: [
            record("cached", stepSeconds: [20], cache: 0.1, skipped: 12),
            record("dense", stepSeconds: [40], cache: 0, skipped: 0)
        ])
        #expect(c.rows.first { $0.arm == "cached" }?.stepsSkippedFraction == 1.0)
        #expect(c.rows.first { $0.arm == "dense" }?.stepsSkippedFraction == 0.0)
    }
}
