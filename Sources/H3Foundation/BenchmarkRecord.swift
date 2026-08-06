// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import CryptoKit

/// Everything needed to say two renders are comparable — and to prove it later.
///
/// **This type exists because of a specific failure.** The Sol-Attn work
/// produced a speed number, a quality number and a conclusion, and the
/// conclusion turned out to rest on runs made with the cache disabled — which
/// nobody noticed until a viewer watched the output. The measurements were not
/// wrong; they were answers to a question different from the one being asked,
/// and nothing in the artefacts they produced could have revealed that. A
/// wall-clock figure with no record of what was on carries no information.
///
/// So the rule for the phases that follow is: **no optimisation is evaluated
/// without a control from the same machine, in the same configuration, recorded
/// in this form.** Not a convention — the comparison helpers below refuse to
/// compare records that disagree on anything that would change the answer.
///
/// Prompt text and absolute paths are excluded here for the same reason
/// `RenderReceipt` excludes them. Identity is pinned by digest instead, which
/// is what a comparison actually needs: two runs can be *proven* to share a
/// prompt without the prompt appearing in a file anyone might publish.
package struct BenchmarkRecord: Codable, Sendable, Equatable {

    package static let schemaVersion = 1

    /// The inputs. Two records whose `Identity` differ are measuring different
    /// things, whatever their timings say.
    package struct Identity: Codable, Sendable, Equatable {
        /// SHA-256 of the prompt text, hex. Pins the prompt without recording it.
        package let promptDigest: String
        package let negativePromptDigest: String?
        package let seed: UInt64
        package let width: Int
        package let height: Int
        package let seconds: Int
        package let steps: Int
        package let cfgScale: Double
        /// SHA-256 of the DiT checkpoint when the run verified it, else nil.
        /// Nil is honest: an unverified 66 GB file is not a pinned input, and a
        /// record that quietly pretended otherwise would be worse than one that
        /// admits the gap.
        package let ditDigest: String?
        package let ditSizeBytes: UInt64

        package init(promptDigest: String, negativePromptDigest: String?, seed: UInt64,
                    width: Int, height: Int, seconds: Int, steps: Int, cfgScale: Double,
                    ditDigest: String?, ditSizeBytes: UInt64) {
            self.promptDigest = promptDigest
            self.negativePromptDigest = negativePromptDigest
            self.seed = seed
            self.width = width
            self.height = height
            self.seconds = seconds
            self.steps = steps
            self.cfgScale = cfgScale
            self.ditDigest = ditDigest
            self.ditSizeBytes = ditSizeBytes
        }
    }

    /// What was switched on. The field that the Sol-Attn runs did not have.
    package struct Configuration: Codable, Sendable, Equatable {
        package let qualityProfile: String
        package let attentionBackend: String
        /// 0 means the cross-step cache was off — a faithful, dense control.
        package let cacheThreshold: Double
        package let cacheMaxConsecutiveSkips: Int
        package let cachePerStreamProbe: Bool
        /// Free-form, for arms that a named profile does not describe: the
        /// environment overrides a sweep sets. Recorded verbatim so an arm can
        /// never be reconstructed wrongly from a label.
        package let overrides: [String: String]

        package init(qualityProfile: String, attentionBackend: String,
                    cacheThreshold: Double, cacheMaxConsecutiveSkips: Int,
                    cachePerStreamProbe: Bool, overrides: [String: String] = [:]) {
            self.qualityProfile = qualityProfile
            self.attentionBackend = attentionBackend
            self.cacheThreshold = cacheThreshold
            self.cacheMaxConsecutiveSkips = cacheMaxConsecutiveSkips
            self.cachePerStreamProbe = cachePerStreamProbe
            self.overrides = overrides
        }
    }

    /// The machine. A 5% gain measured against a control from a different box
    /// is not a gain.
    package struct Machine: Codable, Sendable, Equatable {
        package let model: String
        package let cores: Int
        package let physicalMemoryBytes: UInt64
        package let operatingSystem: String
        package let libraryVersion: String
        /// The pinned mlx-swift revision. Kernel timings move between MLX
        /// releases for reasons that have nothing to do with this tree.
        package let mlxSwiftVersion: String

        package init(model: String, cores: Int, physicalMemoryBytes: UInt64,
                    operatingSystem: String, libraryVersion: String, mlxSwiftVersion: String) {
            self.model = model
            self.cores = cores
            self.physicalMemoryBytes = physicalMemoryBytes
            self.operatingSystem = operatingSystem
            self.libraryVersion = libraryVersion
            self.mlxSwiftVersion = mlxSwiftVersion
        }
    }

    package struct Memory: Codable, Sendable, Equatable {
        /// What MLX allocated at peak. **Not resident set** — the checkpoints
        /// are memory-mapped and do not appear here, which is why a render
        /// holding 117.8 GB of mapped pages can report a 53.1 GB peak and both
        /// numbers be honest.
        package let mlxPeakBytes: UInt64
        package let mlxActiveBytesAtEnd: UInt64

        package init(mlxPeakBytes: UInt64, mlxActiveBytesAtEnd: UInt64) {
            self.mlxPeakBytes = mlxPeakBytes
            self.mlxActiveBytesAtEnd = mlxActiveBytesAtEnd
        }
    }

    /// Whatever the quality tools reported, if they were run. Left open rather
    /// than typed per tool, because the set of tools grows and a record written
    /// last month must still decode.
    ///
    /// **PSNR may be recorded here but is not a gate.** Generated media shifts
    /// spatially between configurations by amounts a viewer would not notice
    /// and PSNR punishes severely; ranking on it selects for blur. The gates
    /// are detail, speech, audio spectrum, face stability and lip-sync.
    package typealias Quality = [String: Double]

    package let schemaVersion: Int
    package let recordedAt: Date
    /// What this arm is called in the sweep. `control-dense`, `control-cached`,
    /// and then whatever is being tested.
    package let arm: String
    package let identity: Identity
    package let configuration: Configuration
    package let machine: Machine
    /// Seconds per phase, plus `total`. Every boundary is drawn after an
    /// explicit MLX evaluation — under lazy execution a phase that returns an
    /// unevaluated array hands its cost to whichever phase forces it, and the
    /// resulting split is fiction.
    package let phaseSeconds: [String: Double]
    package let memory: Memory
    package let trace: SamplingTrace
    package var quality: Quality

    package init(arm: String, identity: Identity, configuration: Configuration,
                machine: Machine, phaseSeconds: [String: Double], memory: Memory,
                trace: SamplingTrace, quality: Quality = [:], recordedAt: Date = .now) {
        schemaVersion = Self.schemaVersion
        self.recordedAt = recordedAt
        self.arm = arm
        self.identity = identity
        self.configuration = configuration
        self.machine = machine
        self.phaseSeconds = phaseSeconds
        self.memory = memory
        self.trace = trace
        self.quality = quality
    }

    /// Hex SHA-256, the digest form used throughout.
    package static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Comparison

    /// Why two records cannot be compared. Empty means they can.
    ///
    /// The point of returning the reasons rather than a Bool: a sweep that
    /// silently drops incomparable arms produces a clean table with a hole in
    /// it, and the hole is invisible. These strings go in the report.
    package func incomparabilities(with other: BenchmarkRecord) -> [String] {
        var out: [String] = []
        func check(_ name: String, _ a: String, _ b: String) {
            if a != b { out.append("\(name): \(a) vs \(b)") }
        }
        if identity != other.identity {
            check("prompt", identity.promptDigest.prefix(12).description,
                  other.identity.promptDigest.prefix(12).description)
            check("seed", "\(identity.seed)", "\(other.identity.seed)")
            check("dimensions", "\(identity.width)x\(identity.height)x\(identity.seconds)s",
                  "\(other.identity.width)x\(other.identity.height)x\(other.identity.seconds)s")
            check("steps", "\(identity.steps)", "\(other.identity.steps)")
            check("cfg", "\(identity.cfgScale)", "\(other.identity.cfgScale)")
            // A nil digest on either side is not a mismatch, it is an absence —
            // reported separately so it cannot be read as "the checkpoints
            // differ".
            switch (identity.ditDigest, other.identity.ditDigest) {
            case let (a?, b?): check("checkpoint", a.prefix(12).description, b.prefix(12).description)
            case (nil, nil): break
            default: out.append("checkpoint: one run did not verify its DiT, so identity is unproven")
            }
            if identity.ditSizeBytes != other.identity.ditSizeBytes {
                out.append("checkpoint size: \(identity.ditSizeBytes) vs \(other.identity.ditSizeBytes)")
            }
        }
        check("machine", machine.model, other.machine.model)
        check("mlx-swift", machine.mlxSwiftVersion, other.machine.mlxSwiftVersion)
        return out
    }

    /// This record's median step time relative to a control's. Above 1 is
    /// faster.
    ///
    /// Throws rather than returning a number when the two are not comparable.
    /// The alternative — returning a ratio anyway and leaving a caveat in a log
    /// line — is exactly how the cache-disabled Sol-Attn numbers travelled.
    package func speedup(over control: BenchmarkRecord) throws -> Double {
        let problems = incomparabilities(with: control)
        guard problems.isEmpty else { throw Incomparable(reasons: problems) }
        let mine = trace.medianStepSeconds, theirs = control.trace.medianStepSeconds
        guard mine.isFinite, theirs.isFinite, mine > 0 else { return .nan }
        return theirs / mine
    }

    package struct Incomparable: Error, CustomStringConvertible {
        package let reasons: [String]
        package var description: String {
            "these runs are not comparable — " + reasons.joined(separator: "; ")
        }
    }
}
