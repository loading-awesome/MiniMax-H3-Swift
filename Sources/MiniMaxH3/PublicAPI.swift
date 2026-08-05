import Foundation
import H3Catalog
import H3Foundation
import H3Recipes
import H3Pipeline

// The implementation types live in downward package targets. These explicit
// aliases are the supported facade; unlike `@_exported import`, they do not
// make every public implementation declaration part of this module's apparent
// API.
public typealias RenderRequest = H3Pipeline.RenderRequest
public typealias RenderProgress = H3Pipeline.RenderProgress
public typealias RenderResult = H3Pipeline.RenderResult
public typealias RenderCancellation = H3Pipeline.RenderCancellation
public typealias RenderCancelled = H3Pipeline.RenderCancelled
public typealias H3Error = H3Foundation.H3Error
public typealias PolicyViolation = H3Foundation.PolicyViolation
public typealias H3RecipeID = H3Recipes.H3RecipeID
public typealias RenderMode = H3Catalog.RenderMode

/// A checkpoint file plus optional cryptographic identity.
public struct ModelFile: Sendable, Codable, Equatable {
    public let url: URL
    public let expectedSHA256: String?

    public init(url: URL, expectedSHA256: String? = nil) {
        self.url = url
        self.expectedSHA256 = expectedSHA256?.lowercased()
    }
}

/// Every model artifact required by one render.
public struct ModelSet: Sendable, Equatable {
    public enum Precision: String, Codable, Sendable, CaseIterable {
        case bf16
        case int8
        case prunedBF16 = "pruned_bf16"
        case prunedInt8 = "pruned_int8"
    }

    public let dit: ModelFile
    public let textEncoder: ModelFile
    public let tokenizer: URL
    public let videoVAE: ModelFile
    public let audioVAE: ModelFile
    public let precision: Precision

    public init(dit: ModelFile, textEncoder: ModelFile, tokenizer: URL,
                videoVAE: ModelFile, audioVAE: ModelFile,
                precision: Precision = .bf16) {
        self.dit = dit
        self.textEncoder = textEncoder
        self.tokenizer = tokenizer
        self.videoVAE = videoVAE
        self.audioVAE = audioVAE
        self.precision = precision
    }
}

/// Typed operational output. Diagnostic text remains useful to humans, while
/// progress and terminal events are stable enough for applications to switch
/// over without parsing log lines.
public enum RenderEvent: Sendable {
    case admitted(jobID: UUID)
    case progress(RenderProgress)
    case diagnostic(String)
    case warning(String)
    case completed(RenderResult)
    case failed(code: String, message: String)
}
