import Foundation

/// Where the checkpoints are and what the caller is willing to run, so neither
/// has to travel on the command line.
///
/// Four sources, each overriding the one before: **built-in defaults, then the
/// config file, then environment variables, then CLI flags.** The resolved
/// value carries its own provenance, because "why is it using that checkpoint"
/// is the first question anybody asks and the answer should not require reading
/// three files.
///
/// JSON rather than TOML for one reason: `JSONDecoder` is in the standard
/// library and a TOML parser is a dependency. The schema below is small enough
/// that the difference in ergonomics is a few quotation marks, and a
/// configuration format is a poor place to spend a supply-chain risk.
package struct H3Configuration: Codable, Sendable, Equatable {

    package struct Checkpoints: Codable, Sendable, Equatable {
        /// Everything else is resolved relative to this unless absolute.
        package var root: String?
        package var fl2va: [String: String]
        package var ref2va: [String: String]
        package var textEncoder: [String: String]
        package var videoVAE: String?
        package var audioVAE: String?
        package var tokenizer: String?

        package init(root: String? = nil,
                    fl2va: [String: String] = [:], ref2va: [String: String] = [:],
                    textEncoder: [String: String] = [:],
                    videoVAE: String? = nil, audioVAE: String? = nil,
                    tokenizer: String? = nil) {
            self.root = root
            self.fl2va = fl2va
            self.ref2va = ref2va
            self.textEncoder = textEncoder
            self.videoVAE = videoVAE
            self.audioVAE = audioVAE
            self.tokenizer = tokenizer
        }

        /// Explicit keys, because `.convertFromSnakeCase` does not round-trip
        /// an acronym.
        ///
        /// `video_vae` converts to `videoVae`, which does not match `videoVAE`,
        /// so both VAE paths decoded as nil — silently, since every field here
        /// is optional. `h3 doctor` listed eight checkpoints and neither VAE
        /// and looked entirely healthy doing it. A unit test caught it; the
        /// integration output did not.
        enum CodingKeys: String, CodingKey {
            case root, fl2va, ref2va
            case textEncoder = "text_encoder"
            case videoVAE = "video_vae"
            case audioVAE = "audio_vae"
            case tokenizer
        }

        /// Every key optional, defaulting to empty.
        ///
        /// Swift's synthesised `Codable` ignores property defaults for absent
        /// keys, so without this a configuration that only names the
        /// checkpoints it owns — a perfectly reasonable thing to write — fails
        /// to decode with `keyNotFound`. A partial config is the normal case,
        /// not an error.
        package init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            root = try c.decodeIfPresent(String.self, forKey: .root)
            fl2va = try c.decodeIfPresent([String: String].self, forKey: .fl2va) ?? [:]
            ref2va = try c.decodeIfPresent([String: String].self, forKey: .ref2va) ?? [:]
            textEncoder = try c.decodeIfPresent([String: String].self, forKey: .textEncoder) ?? [:]
            videoVAE = try c.decodeIfPresent(String.self, forKey: .videoVAE)
            audioVAE = try c.decodeIfPresent(String.self, forKey: .audioVAE)
            tokenizer = try c.decodeIfPresent(String.self, forKey: .tokenizer)
        }
    }

    package struct Policy: Codable, Sendable, Equatable {
        /// Gates the `pruned` variants, whose AdaLN is a rank-64 curve
        /// approximation rather than the released projection. Off by default:
        /// falling back to them silently would hand somebody a different model
        /// and let them compare its output to the gated configuration.
        package var allowApproximateWeights: Bool
        /// Wall-clock ceiling before a render is refused without an explicit
        /// opt-in.
        package var maxRenderSeconds: Double
        /// Fraction of available memory held back when planning. Memory
        /// pressure is not a cliff the planner can see coming.
        package var memoryMarginFraction: Double

        enum CodingKeys: String, CodingKey {
            case allowApproximateWeights = "allow_approximate_weights"
            case maxRenderSeconds = "max_render_seconds"
            case memoryMarginFraction = "memory_margin_fraction"
        }

        package init(allowApproximateWeights: Bool = false,
                    maxRenderSeconds: Double = 3 * 3600,
                    memoryMarginFraction: Double = 0.15) {
            self.allowApproximateWeights = allowApproximateWeights
            self.maxRenderSeconds = maxRenderSeconds
            self.memoryMarginFraction = memoryMarginFraction
        }

        package init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            allowApproximateWeights = try c.decodeIfPresent(Bool.self, forKey: .allowApproximateWeights) ?? false
            maxRenderSeconds = try c.decodeIfPresent(Double.self, forKey: .maxRenderSeconds) ?? 3 * 3600
            memoryMarginFraction = try c.decodeIfPresent(Double.self, forKey: .memoryMarginFraction) ?? 0.15
        }
    }

    package struct Attention: Codable, Sendable, Equatable {
        /// `auto`, or a registered backend identifier.
        package var backend: String
        package init(backend: String = "auto") { self.backend = backend }

        package init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            backend = try c.decodeIfPresent(String.self, forKey: .backend) ?? "auto"
        }
    }

    package var checkpoints: Checkpoints
    package var policy: Policy
    package var attention: Attention

    package init(checkpoints: Checkpoints = .init(), policy: Policy = .init(),
                attention: Attention = .init()) {
        self.checkpoints = checkpoints
        self.policy = policy
        self.attention = attention
    }

    package init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        checkpoints = try c.decodeIfPresent(Checkpoints.self, forKey: .checkpoints) ?? .init()
        policy = try c.decodeIfPresent(Policy.self, forKey: .policy) ?? .init()
        attention = try c.decodeIfPresent(Attention.self, forKey: .attention) ?? .init()
    }

    // MARK: loading

    package static var defaultURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/minimax-h3/config.json")
    }

    /// Loads from `url`, or returns defaults when the file does not exist.
    ///
    /// A missing config file is not an error — a caller passing every path on
    /// the command line is a legitimate way to use this. A config file that
    /// exists and cannot be parsed **is** an error, and names the line, because
    /// silently falling back to defaults after somebody edited a file is how a
    /// render ends up using a checkpoint nobody chose.
    package static func load(from url: URL? = nil) throws -> (H3Configuration, URL?) {
        let target = url ?? defaultURL
        guard FileManager.default.fileExists(atPath: target.path) else {
            if url != nil {
                throw H3Error.unreadable(path: target.path, reason: "no such config file")
            }
            return (H3Configuration(), nil)
        }
        let data: Data
        do { data = try Data(contentsOf: target) }
        catch { throw H3Error.unreadable(path: target.path, reason: error.localizedDescription) }
        do {
            // No `.convertFromSnakeCase`: it rewrites keys before CodingKeys
            // are consulted, so `video_vae` became `videoVae` and never matched
            // `videoVAE`. The two mechanisms fight and the strategy wins
            // silently. Explicit keys are the whole contract.
            return (try JSONDecoder().decode(H3Configuration.self, from: data), target)
        } catch {
            throw H3Error.unreadable(path: target.path,
                                     reason: "not valid configuration JSON: \(error)")
        }
    }

    /// Absolute URL for a path that may be relative to `checkpoints.root`.
    package func resolve(_ path: String?) -> URL? {
        guard let path, !path.isEmpty else { return nil }
        if path.hasPrefix("/") { return URL(fileURLWithPath: path) }
        guard let root = checkpoints.root else { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: root).appendingPathComponent(path)
    }

    /// A starting config pointing at nothing, for `h3 config init`.
    package static var template: H3Configuration {
        H3Configuration(
            checkpoints: .init(
                root: "/path/to/MiniMax-H3",
                fl2va: ["bf16": "MiniMax-H3-FL2VA_bf16.safetensors",
                        "int8": "MiniMax-H3-FL2VA_int8_convrot.safetensors",
                        "pruned_bf16": "MiniMax-H3-FL2VA-pruned_bf16.safetensors",
                        "pruned_int8": "MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors"],
                ref2va: ["bf16": "MiniMax-H3-Ref2VA_bf16.safetensors",
                         "int8": "MiniMax-H3-Ref2VA_int8_convrot.safetensors",
                         "pruned_bf16": "MiniMax-H3-Ref2VA-pruned_bf16.safetensors",
                         "pruned_int8": "MiniMax-H3-Ref2VA-pruned_int8_convrot.safetensors"],
                textEncoder: ["bf16": "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors",
                              "int8": "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"],
                videoVAE: "MiniMax-H3-video_vae_fp16.safetensors",
                audioVAE: "MiniMax-H3-audio_vae_fp32.safetensors",
                tokenizer: "tokenizer"),
            policy: .init(), attention: .init())
    }

    package func encoded() throws -> Data {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try e.encode(self)
    }
}
