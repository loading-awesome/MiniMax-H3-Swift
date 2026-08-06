// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
@testable import H3Foundation

/// The comparison contract.
///
/// The rule these tests enforce is the one the Sol-Attn work needed and did not
/// have: **a speed-up figure may not be produced from two runs that were not
/// the same experiment.** Not a warning in a log, not a caveat in a doc — a
/// thrown error, so that the number simply does not exist.
@Suite("benchmark record")
struct BenchmarkRecordTests {

    static func identity(seed: UInt64 = 7, steps: Int = 20, width: Int = 864,
                         prompt: String = "a red kite", dit: String? = "abc123")
        -> BenchmarkRecord.Identity {
        .init(promptDigest: BenchmarkRecord.digest(prompt), negativePromptDigest: nil,
              seed: seed, width: width, height: 480, seconds: 5, steps: steps,
              cfgScale: 1.0, ditDigest: dit, ditSizeBytes: 66_000_000_000)
    }

    static func record(arm: String, stepSeconds: [Double],
                       identity: BenchmarkRecord.Identity = identity(),
                       cacheThreshold: Double = 0,
                       machine: String = "Mac Studio M3 Ultra",
                       mlx: String = "0.31.6") -> BenchmarkRecord {
        BenchmarkRecord(
            arm: arm, identity: identity,
            configuration: .init(qualityProfile: "balanced", attentionBackend: "sdpa",
                                 cacheThreshold: cacheThreshold, cacheMaxConsecutiveSkips: 3,
                                 cachePerStreamProbe: true),
            machine: .init(model: machine, cores: 32, physicalMemoryBytes: 275_000_000_000,
                           operatingSystem: "Version 15.0", libraryVersion: "0.1.0-dev",
                           mlxSwiftVersion: mlx),
            phaseSeconds: ["sampling": stepSeconds.reduce(0, +), "total": 1000],
            memory: .init(mlxPeakBytes: 53_000_000_000, mlxActiveBytesAtEnd: 1_000),
            trace: SamplingTrace(steps: [], stepSeconds: stepSeconds))
    }

    @Test("a speed-up over an identical control is the ratio of median step times")
    func speedupOverControl() throws {
        let control = Self.record(arm: "control-dense", stepSeconds: [40, 40, 40, 40, 40])
        let cached = Self.record(arm: "cached", stepSeconds: [40, 20, 20, 20, 40],
                                 cacheThreshold: 0.1)
        // Medians are 40 and 20. Wall-clock totals are 200 and 140, which would
        // have said 1.43x — the first and last steps every arm must run in full
        // diluting the figure by a fixed amount that varies with step count.
        #expect(try cached.speedup(over: control) == 2.0)
    }

    @Test("a different seed makes the two runs incomparable")
    func seedMismatchThrows() {
        let control = Self.record(arm: "control", stepSeconds: [40])
        let other = Self.record(arm: "test", stepSeconds: [20],
                                identity: Self.identity(seed: 8))
        #expect(throws: BenchmarkRecord.Incomparable.self) { try other.speedup(over: control) }
        #expect(other.incomparabilities(with: control).contains { $0.hasPrefix("seed:") })
    }

    @Test("a different machine or MLX revision makes the two runs incomparable")
    func environmentMismatchThrows() {
        // Kernel timings move between MLX releases for reasons that have
        // nothing to do with this tree, and a gain measured against a control
        // from another box is not a gain.
        let control = Self.record(arm: "control", stepSeconds: [40])
        #expect(throws: BenchmarkRecord.Incomparable.self) {
            try Self.record(arm: "t", stepSeconds: [20], machine: "MacBook Pro M4 Max")
                .speedup(over: control)
        }
        #expect(throws: BenchmarkRecord.Incomparable.self) {
            try Self.record(arm: "t", stepSeconds: [20], mlx: "0.32.0").speedup(over: control)
        }
    }

    @Test("an unverified checkpoint is an absence, reported as one")
    func unverifiedCheckpoint() {
        // Two runs that both skipped verification cannot be shown to have used
        // the same 66 GB file, but they are also not known to differ — and the
        // report must say which of those it is rather than "checkpoint: a vs b".
        let verified = Self.record(arm: "a", stepSeconds: [40])
        let unverified = Self.record(arm: "b", stepSeconds: [40],
                                     identity: Self.identity(dit: nil))
        let reasons = unverified.incomparabilities(with: verified)
        #expect(reasons.contains { $0.contains("did not verify") })
        // Both unverified: still comparable. Refusing here would make the
        // contract unusable on any machine that has not hashed 66 GB, which is
        // most of them, and an unusable contract gets bypassed.
        let bothUnverified = Self.record(arm: "c", stepSeconds: [40],
                                         identity: Self.identity(dit: nil))
        #expect(unverified.incomparabilities(with: bothUnverified).isEmpty)
    }

    @Test("differing configuration does not block comparison — that is the point")
    func configurationMayDiffer() throws {
        // Identity and machine must match; configuration is precisely what is
        // being varied. A contract that required these to be equal would refuse
        // every comparison anyone wanted to make.
        let dense = Self.record(arm: "control-dense", stepSeconds: [40], cacheThreshold: 0)
        let cached = Self.record(arm: "cached", stepSeconds: [20], cacheThreshold: 0.1)
        #expect(cached.incomparabilities(with: dense).isEmpty)
        #expect(try cached.speedup(over: dense) == 2.0)
    }

    @Test("a record round-trips through JSON")
    func codableRoundTrip() throws {
        let r = Self.record(arm: "control", stepSeconds: [40, 20])
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let back = try decoder.decode(BenchmarkRecord.self,
                                      from: try encoder.encode(r))
        #expect(back.identity == r.identity)
        #expect(back.configuration == r.configuration)
        #expect(back.trace == r.trace)
        #expect(back.schemaVersion == BenchmarkRecord.schemaVersion)
    }

    @Test("the prompt is pinned by digest and not recorded")
    func promptIsDigestedNotStored() throws {
        // The receipt deliberately excludes prompt text; the benchmark record
        // has to pin prompt identity anyway. A digest does both — two runs can
        // be proven to share a prompt without the prompt appearing in a file
        // anyone might publish alongside a clip.
        let secret = "a very specific prompt someone would rather not publish"
        let r = Self.record(arm: "a", stepSeconds: [1], identity: Self.identity(prompt: secret))
        let json = String(decoding: try JSONEncoder().encode(r), as: UTF8.self)
        #expect(!json.contains("very specific"))
        #expect(json.contains(BenchmarkRecord.digest(secret)))
        #expect(BenchmarkRecord.digest(secret) != BenchmarkRecord.digest(secret + "."))
    }
}
