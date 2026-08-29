// swift-tools-version: 6.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich
import PackageDescription

// The target graph is the architecture, so it is worth reading as one.
//
// Dependencies only ever point downward, and the two lowest layers —
// H3Foundation and H3Hardware — do not link MLX at all. That is not tidiness:
// it means `swift test` for geometry, the 17k+5 lattice, safetensors headers,
// checkpoint identification, the flow schedule and the memory planner runs in
// seconds on any machine, with no GPU and no 66 GB checkpoint. Those are
// exactly the parts where a silent error stays silent, so they are the parts
// that must stay cheap to check.
//
//   H3Foundation   errors, geometry, config, safetensors        (no MLX)
//   H3Hardware     chip + memory detection, the memory planner  (no MLX)
//   H3Catalog      checkpoint discovery and identification      (no MLX)
//   H3Recipes      capability-aware recipe resolution           (no MLX)
//   H3Attention    the attention backend seam                   (MLX)
//   H3Modules      DiT, VAEs, vision tower, text encoder        (MLX)
//   H3Pipeline     conditioning, layout, sampler, decode, mux   (MLX)
//   H3Conformance  contract ownership + tier ledger             (no MLX)
//   MiniMaxH3      the public API and actor-owned runtime facade
//   h3             a thin CLI over the public API
let package = Package(
    name: "MiniMaxH3",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MiniMaxH3", targets: ["MiniMaxH3"]),
        .executable(name: "h3", targets: ["h3"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.7.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.2"),
    ],
    targets: [
        .target(name: "H3Foundation"),
        .target(name: "H3Hardware", dependencies: ["H3Foundation"]),
        .target(name: "H3Catalog", dependencies: ["H3Foundation"]),
        .target(name: "H3Recipes", dependencies: ["H3Foundation", "H3Hardware", "H3Catalog"]),
        .target(
            name: "H3Attention",
            dependencies: [
                "H3Foundation", "H3Hardware", "H3ANEBridge",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "H3ANEBridge",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("IOSurface"),
                .linkedFramework("Metal"),
            ]
        ),
        .target(
            name: "H3Modules",
            dependencies: [
                "H3Foundation", "H3Catalog", "H3Attention", "H3ANEBridge",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "H3Pipeline",
            dependencies: [
                "H3Foundation", "H3Hardware", "H3Catalog", "H3Recipes", "H3Modules",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MiniMaxH3",
            dependencies: ["H3Foundation", "H3Hardware", "H3Catalog", "H3Recipes", "H3Pipeline"]
        ),
        .target(
            name: "H3Conformance",
            dependencies: ["H3Foundation"]
        ),
        .executableTarget(
            name: "h3",
            dependencies: [
                "MiniMaxH3",
                "H3Foundation", "H3Hardware", "H3Catalog", "H3Recipes",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        // The external oracles in Tools/. They ask "is the output any good"
        // rather than "does it match a golden", which is the only question that
        // can be asked of a render nobody has captured a reference for — and
        // the only one that survives an arithmetic change, since a change in
        // precision reselects the sample rather than degrading it (see
        // `docs/ANE_PRECISION_RESULTS.md`).
        //
        // They are targets rather than loose files because they were neither:
        // `LipSyncCheck` still imported `H3Core`, a module this package has not
        // had for some time, so it could not be built by any documented means
        // and nothing noticed.
        .executableTarget(
            name: "h3-decode-audio",
            dependencies: [
                "H3Foundation", "H3Modules",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tools/FastH3", sources: ["decode_audio_latent.swift"]
        ),
        .executableTarget(
            name: "h3-facecheck",
            dependencies: [.product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Tools", exclude: ["ANE", "LipSyncCheck.swift", "__pycache__", "anchor_check.py", "arm_compare.py", "audio_match.py", "coherence_check.py", "speech_check.py"], sources: ["FaceCheck.swift"]
        ),
        .executableTarget(
            name: "h3-lipsync",
            dependencies: [
                "H3Foundation", "H3Pipeline",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools", exclude: ["ANE", "FaceCheck.swift", "__pycache__", "anchor_check.py", "arm_compare.py", "audio_match.py", "coherence_check.py", "speech_check.py"], sources: ["LipSyncCheck.swift"]
        ),
        .testTarget(name: "H3FoundationTests", dependencies: ["H3Foundation"]),
        .testTarget(name: "H3CatalogTests", dependencies: ["H3Catalog"]),
        .testTarget(name: "H3HardwareTests", dependencies: ["H3Hardware"]),
        .testTarget(name: "H3RecipesTests", dependencies: ["H3Recipes"]),
        // MLX-linked test targets need `Scripts/bootstrap-metal.sh` to have run;
        // see `H3Hardware.MetalLibrary` for why, and for what the failure looks
        // like when it has not. An earlier version of this comment blamed an
        // empty `resources: [.copy("Fixtures")]` declaration. That was wrong —
        // removing it changed nothing, and the real cause is that SwiftPM never
        // builds MLX's Metal kernels at all.
        .testTarget(name: "H3AttentionTests",
                    dependencies: ["H3Attention", "H3Hardware",
                                   .product(name: "MLX", package: "mlx-swift"),
                                   .product(name: "MLXFast", package: "mlx-swift"),
                                   .product(name: "MLXRandom", package: "mlx-swift")]),
        .testTarget(name: "H3ModulesTests",
                    dependencies: ["H3Modules", "H3Attention", "H3Hardware", "H3ANEBridge",
                                   .product(name: "MLX", package: "mlx-swift")],
                    linkerSettings: [.linkedFramework("Metal")]),
        .testTarget(name: "H3PipelineTests",
                    dependencies: ["H3Pipeline", "H3Foundation", "H3Catalog",
                                   .product(name: "MLX", package: "mlx-swift")]),
        .testTarget(name: "H3ConformanceTests", dependencies: ["H3Conformance"]),
        .testTarget(name: "MiniMaxH3Tests", dependencies: ["MiniMaxH3"]),
    ]
)
