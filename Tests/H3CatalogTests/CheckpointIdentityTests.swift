import Testing
import Foundation
import H3Foundation
@testable import H3Catalog

/// Identification and refusal, exercised against synthetic safetensors headers
/// so the suite needs no 66 GB checkpoint and runs anywhere.
///
/// The header format is `[u64 length][JSON][blob]`, so a valid file with no
/// tensor bodies is a few hundred bytes — enough to test every decision this
/// layer makes.
@Suite("checkpoint identity")
struct CheckpointIdentityTests {

    /// Writes a header-only safetensors file. Offsets are all zero-length,
    /// which is legal and never read because identification is header-only.
    static func writeHeader(metadata: [String: String],
                            tensors: [String: [Int]]) throws -> URL {
        var obj: [String: Any] = [:]
        if !metadata.isEmpty { obj["__metadata__"] = metadata }
        for (name, shape) in tensors {
            obj[name] = ["dtype": "BF16", "shape": shape, "data_offsets": [0, 0]]
        }
        let json = try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3-test-\(UUID().uuidString).safetensors")
        try out.write(to: url)
        return url
    }

    static let ditTensors = ["blocks.0.attn.qkv_proj.weight": [21504, 5376],
                             "blocks.0.adaln_proj.linear.weight": [96768, 2688]]

    @Test("a DiT with no recognised vendor is refused, not guessed at")
    func unidentifiedVendorRefused() throws {
        // The two published bf16 conversions store fused attention in different
        // layouts with identical shape, dtype, mean and standard deviation.
        // Guessing gives cos 0.029 — uncorrelated output that reads as a subtle
        // numerical bug rather than a loading error.
        let url = try Self.writeHeader(metadata: [:], tensors: Self.ditTensors)
        defer { try? FileManager.default.removeItem(at: url) }
        let id = try CheckpointIdentity.identify(url: url)
        #expect(id.kind == .dit)
        #expect(id.vendor == nil)
        #expect(throws: H3Error.self) {
            try id.validate(forMode: .textToVideo, allowApproximate: false)
        }
    }

    @Test("a text encoder with no vendor metadata is not a problem")
    func textEncoderNeedsNoVendor() throws {
        // Vendor is a DiT concern: only the DiT carries fused attention weights
        // that can be silently misread. The first version of this check
        // reported the Qwen encoder as UNIDENTIFIED VENDOR and was wrong to.
        let url = try Self.writeHeader(metadata: [:],
                                       tensors: ["visual.blocks.0.attn.qkv.weight": [3456, 1152]])
        defer { try? FileManager.default.removeItem(at: url) }
        let id = try CheckpointIdentity.identify(url: url)
        #expect(id.kind == .textEncoder)
        #expect(id.vendor == nil)
        // No throw: a text encoder is judged on being present and correct, not
        // on metadata it was never published with.
        try id.validate(forMode: .textToVideo, allowApproximate: false)
    }

    @Test("the FL2VA checkpoint is refused for a reference render")
    func wrongPartitionRefused() throws {
        // The bug this whole layer exists to prevent: every reference render on
        // 2026-08-05 used FL2VA because one path was hard-coded. It loads, the
        // architecture matches, the render succeeds, and the weights were never
        // trained to consume reference blocks.
        let url = try Self.writeHeader(
            metadata: ["repo_id": "MiniMaxAI/MiniMax-H3", "partition": "FL2VA"],
            tensors: Self.ditTensors)
        defer { try? FileManager.default.removeItem(at: url) }
        let id = try CheckpointIdentity.identify(url: url)
        #expect(id.partition == .fl2va)
        #expect(id.vendor == .deepBeepMeep)

        try id.validate(forMode: .textToVideo, allowApproximate: false)
        try id.validate(forMode: .firstLastFrame, allowApproximate: false)
        #expect(throws: H3Error.self) {
            try id.validate(forMode: .reference, allowApproximate: false)
        }
    }

    @Test("the curve approximation is detected from shape, not only metadata")
    func approximateDetectedFromShape() throws {
        // "pruned" is not pruning: AdaLN becomes [96768, 64] rather than
        // [96768, 2688], which is exactly the 24.9 GB it saves. Checking the
        // shape as well as the metadata means a re-export that drops the
        // metadata is still caught.
        let url = try Self.writeHeader(
            metadata: ["repo_id": "MiniMaxAI/MiniMax-H3", "partition": "Ref2VA"],
            tensors: ["blocks.0.attn.qkv_proj.weight": [21504, 5376],
                      "blocks.0.adaln_proj.linear.weight": [96768, 64]])
        defer { try? FileManager.default.removeItem(at: url) }
        let id = try CheckpointIdentity.identify(url: url)
        #expect(id.isApproximate)
        #expect(throws: H3Error.self) {
            try id.validate(forMode: .reference, allowApproximate: false)
        }
        try id.validate(forMode: .reference, allowApproximate: true)
    }

    @Test("quantisation is read from the companion tensors when metadata is absent")
    func quantisationFromTensors() throws {
        let url = try Self.writeHeader(
            metadata: ["repo_id": "MiniMaxAI/MiniMax-H3", "partition": "FL2VA"],
            tensors: ["blocks.0.attn.qkv_proj.weight": [21504, 5376],
                      "blocks.0.attn.qkv_proj.weight_scale": [21504, 1]])
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try CheckpointIdentity.identify(url: url).quantisation == .int8ConvRot)
    }

    @Test("a partial configuration decodes")
    func partialConfigDecodes() throws {
        // A config naming only the checkpoints somebody owns is the normal
        // case. Swift's synthesised Codable ignores property defaults for
        // absent keys, so this needs an explicit decoder and a test that it
        // stays.
        //
        // Loaded through `H3Configuration.load`, not a hand-rolled decoder.
        // The first version of this test built its own `JSONDecoder` with
        // `.convertFromSnakeCase` and so tested a decoder the product does not
        // use — which is how the `video_vae` bug survived a passing suite in
        // the first place. A test that configures its own parser is testing its
        // own parser.
        let json = #"{"checkpoints":{"video_vae":"v.safetensors"}}"#
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3-cfg-\(UUID().uuidString).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        let (cfg, found) = try H3Configuration.load(from: url)
        #expect(found == url)
        #expect(cfg.checkpoints.videoVAE == "v.safetensors")
        #expect(cfg.checkpoints.fl2va.isEmpty)
        #expect(cfg.policy.allowApproximateWeights == false)
        #expect(cfg.attention.backend == "auto")
    }
}
