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
        schemaVersion = 3
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
