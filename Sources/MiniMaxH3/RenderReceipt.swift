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

    init(jobID: UUID, request: RequestSummary, environment: Environment) {
        schemaVersion = 1
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
