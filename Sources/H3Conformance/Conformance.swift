import Foundation

/// Auditable ownership for every numbered entry in `FRAGILE_CONTRACTS.md`.
///
/// This target is deliberately MLX-free. It proves on every commit that the
/// evidence ledger is complete; the mechanisms it points to live in the CPU,
/// MLX, and production release suites named by `tier`.
package enum H3Conformance {
    package enum Tier: String, Sendable, CaseIterable {
        case cpu
        case mlx
        case release
    }

    package enum Mechanism: String, Sendable {
        case unitTest
        case runtimeInvariant
        case numericalFixture
        case productionParity
        case releaseQualityGate
    }

    package struct Contract: Sendable, Equatable {
        package let id: Int
        package let title: String
        package let owner: String
        package let tier: Tier
        package let mechanism: Mechanism
        package let evidence: String

        package init(id: Int, title: String, owner: String, tier: Tier,
                    mechanism: Mechanism, evidence: String) {
            self.id = id
            self.title = title
            self.owner = owner
            self.tier = tier
            self.mechanism = mechanism
            self.evidence = evidence
        }
    }

    private static func c(_ id: Int, _ title: String, _ owner: String,
                          _ tier: Tier, _ mechanism: Mechanism) -> Contract {
        Contract(id: id, title: title, owner: owner, tier: tier,
                 mechanism: mechanism,
                 evidence: "FRAGILE_CONTRACTS.md #\(id)")
    }

    package static let contracts: [Contract] = [
        c(1, "video and audio sigma shifts", "H3Foundation", .cpu, .unitTest),
        c(2, "nested latent stream semantics", "H3Pipeline", .mlx, .numericalFixture),
        c(3, "17k+5 frame lattice", "H3Foundation", .cpu, .unitTest),
        c(4, "layer-50 unnormalised text conditioning", "H3Modules", .mlx, .numericalFixture),
        c(5, "32 kHz and 40 audio latents per second", "H3Foundation", .cpu, .unitTest),
        c(6, "encoder-sized audio mux chunks", "H3Pipeline", .release, .releaseQualityGate),
        c(7, "FL2VA serves text and anchors", "H3Catalog", .cpu, .unitTest),
        c(8, "bf16 is the verified DiT ceiling", "H3Modules", .release, .productionParity),
        c(9, "vendor QKV layouts are not interchangeable", "H3Catalog", .cpu, .unitTest),
        c(10, "packed order is text-audio-video", "H3Foundation", .cpu, .unitTest),
        c(11, "audio velocity follows sigma derivative", "H3Pipeline", .mlx, .numericalFixture),
        c(12, "final AdaLN has one modality", "H3Modules", .mlx, .runtimeInvariant),
        c(13, "audio and video share temporal origin", "H3Foundation", .cpu, .unitTest),
        c(14, "tolerances are shape-specific", "H3Conformance", .release, .productionParity),
        c(15, "CUDA class depends on SDPA backend", "H3Conformance", .release, .productionParity),
        c(16, "AdaLN precision is numerically sensitive", "H3Modules", .mlx, .numericalFixture),
        c(17, "operator oracles use reference modules", "H3Conformance", .release, .productionParity),
        c(18, "text encoder fp32 and DiT bf16", "H3Modules", .mlx, .numericalFixture),
        c(19, "conditioning encoder omits final norm and head", "H3Modules", .mlx, .numericalFixture),
        c(20, "causal encoder attention and RoPE theta", "H3Modules", .mlx, .numericalFixture),
        c(21, "conditioning noise is recorded input", "H3Pipeline", .cpu, .runtimeInvariant),
        c(22, "video VAE architecture asymmetry", "H3Modules", .mlx, .numericalFixture),
        c(23, "VAE encoders return posterior mean", "H3Modules", .mlx, .numericalFixture),
        c(24, "batch size one prevents batched CFG", "H3Pipeline", .cpu, .runtimeInvariant),
        c(25, "keyframes use VAE and vision tower", "H3Pipeline", .mlx, .numericalFixture),
        c(26, "image prompts require mRoPE and DeepStack", "H3Modules", .mlx, .numericalFixture),
        c(27, "image presentation is not chat templated", "H3Modules", .mlx, .numericalFixture),
        c(28, "temporal encode repeats tail and drops once", "H3Modules", .mlx, .numericalFixture),
        c(29, "reference video truncates to generation", "H3Pipeline", .cpu, .unitTest),
        c(30, "paired soundtrack label precedes video", "H3Pipeline", .cpu, .unitTest),
        c(31, "reference resize is PIL Lanczos on uint8", "H3Pipeline", .release, .releaseQualityGate),
        c(32, "zero-width equivalence class is not a gate", "H3Conformance", .release, .productionParity),
        c(33, "candidate writer streams", "H3Foundation", .cpu, .unitTest),
        c(34, "decoded audio RMS is normalised", "H3Conformance", .release, .releaseQualityGate),
        c(35, "quality is judged inside trained envelope", "H3Recipes", .cpu, .unitTest),
        c(36, "canvas-sized media bypasses resize", "H3Pipeline", .cpu, .runtimeInvariant),
    ]

    package static let contractsCovered = contracts.map(\.id)
}
