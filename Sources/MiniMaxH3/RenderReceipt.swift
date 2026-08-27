// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Durable, privacy-conscious account of a render. Prompt text and absolute
/// input paths are deliberately excluded.
public struct RenderReceipt: Codable, Sendable {
    public enum Status: String, Codable, Sendable {
        case running, succeeded, failed, cancelled
    }

    public struct RequestSummary: Codable, Sendable {
        public let mode: String
        public let width: Int
        public let height: Int
        public let seconds: Int
        public let steps: Int
        public let seed: UInt64
        public let promptCharacters: Int
        public let qualityProfile: String
        public let cacheThreshold: Double
        public let referenceImages: Int
        public let referenceVideos: Int
        public let referenceAudio: Int
        /// Visual anchors, counting `firstFrame` and `lastFrame`. Their frame
        /// positions are not recorded: this file is meant to be shareable, and
        /// an anchor schedule describes the content.
        public let keyframeAnchors: Int
    }

    public struct Environment: Codable, Sendable {
        public let libraryVersion: String
        public let operatingSystem: String
        public let machine: String
        public let availableMemoryBytes: UInt64
    }

    public struct Checkpoint: Codable, Sendable {
        public enum Verification: String, Codable, Sendable {
            case notRequested, verified
        }

        public let role: String
        public let filename: String
        public let sizeBytes: UInt64
        public let sha256: String?
        public let verification: Verification
    }

    /// The arithmetic that produced this render.
    ///
    /// A receipt exists so a render can be accounted for after the fact, and
    /// until schema 4 it could not distinguish two renders from the same seed
    /// and checkpoint that produced visibly different videos — because what
    /// separated them was the arithmetic, and the arithmetic was not recorded.
    /// Any change here reselects the diffusion sample rather than degrading it,
    /// so these fields are not a performance note; they are part of what
    /// identifies the output.
    public struct Compute: Codable, Sendable {
        /// `bf16` in production. `fp16` and `fp32` are diagnostic overrides.
        public let ditDType: String
        /// Projections computed on the Neural Engine, empty on the default
        /// path. Observed during the render, not read from configuration.
        public let aneRoutedProjections: [String]
        /// Projections offered to the engine that fell back to MLX — same
        /// numbers, no speedup, but a different run from one that never
        /// offered.
        public let aneDeclinedProjections: [String]
        /// GPU attention on one CFG branch ran beside engine linears on the
        /// other. Optional so a v4 file still reads, and **nil means
        /// unrecorded, not false** — which is why the schema version moves
        /// rather than this field being quietly added to v4. Two files both
        /// claiming 4 with different shapes is the ambiguity the version exists
        /// to prevent, and the arithmetic fields are part of what identifies
        /// the output.
        public let aneCFGOverlap: Bool?

        public init(ditDType: String, aneRoutedProjections: [String],
                    aneDeclinedProjections: [String], aneCFGOverlap: Bool? = nil) {
            self.ditDType = ditDType
            self.aneRoutedProjections = aneRoutedProjections
            self.aneDeclinedProjections = aneDeclinedProjections
            self.aneCFGOverlap = aneCFGOverlap
        }
    }

    public struct Output: Codable, Sendable {
        public let kind: String
        public let filename: String
        public let sizeBytes: UInt64
    }

    public let schemaVersion: Int
    public let jobID: UUID
    public var status: Status
    public let startedAt: Date
    public var finishedAt: Date?
    public let request: RequestSummary
    public let environment: Environment
    public var checkpoints: [Checkpoint]
    public var outputs: [Output]
    /// Written after the render, because what matters is what the engine
    /// actually took, not what it was asked to take.
    public var compute: Compute?
    public var timings: [String: TimeInterval]
    public var warnings: [String]
    public var errorCode: String?
    public var errorMessage: String?

    /// The label of the configuration this run measured, matching the `arm`
    /// field of the `.h3-bench.json` written beside the video.
    public var benchmarkArm: String?
    /// Mean seconds per sampler step.
    ///
    /// Here as well as in the benchmark file because it is the one number worth
    /// reading without opening anything else. Mean rather than median: under
    /// the cache a render's step times are two populations — full steps around
    /// 60 s, reused steps around 1.3 s — and a median falls inside one of them
    /// and reports it as though it were the render's cost.
    public var secondsPerStep: Double?

    init(jobID: UUID, request: RequestSummary, environment: Environment) {
        // 2 adds `benchmarkArm`, `secondsPerStep`, and a `model_load` phase
        // that used to be counted in no phase at all.
        // 3 adds `request.keyframeAnchors`. A v2 reader ignores it; a v3 reader
        // handed a v2 file fails on the missing key, which is what the version
        // is for.
        // 4 adds `compute`. It is optional so a v3 file still reads, but its
        // absence now means "unrecorded", not "default" — a v3 render could
        // have been fp16 and nothing would say so.
        // 5 adds `compute.aneCFGOverlap`. Optional, so a v4 file still reads,
        // but a v4 file cannot say whether the CFG branches were overlapped and
        // a v5 file can.
        schemaVersion = 5
        self.jobID = jobID
        status = .running
        startedAt = .now
        self.request = request
        self.environment = environment
        checkpoints = []
        outputs = []
        timings = [:]
        warnings = []
    }
}
