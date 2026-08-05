// swift-tools-version: 6.0
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
//   H3Conformance  the retained numerical checks                (MLX)
//   MiniMaxH3      the public API, and nothing else
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
                "H3Foundation", "H3Hardware",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "H3Modules",
            dependencies: [
                "H3Foundation", "H3Catalog", "H3Attention",
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
        .target(name: "MiniMaxH3", dependencies: ["H3Pipeline"]),
        .target(
            name: "H3Conformance",
            dependencies: ["H3Foundation", "H3Modules",
                           .product(name: "MLX", package: "mlx-swift")]
        ),
        .executableTarget(
            name: "h3",
            dependencies: [
                "MiniMaxH3",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(name: "H3FoundationTests", dependencies: ["H3Foundation"]),
        .testTarget(name: "H3CatalogTests", dependencies: ["H3Catalog"]),
        .testTarget(name: "H3HardwareTests", dependencies: ["H3Hardware"]),
        .testTarget(name: "H3RecipesTests", dependencies: ["H3Recipes"]),
        // No `resources:` until Fixtures actually holds something. Declaring a
        // copy of an empty directory produces a malformed test bundle, and the
        // symptom is unrelated and baffling: MLX fails to find its metallib.
        .testTarget(
            name: "H3ConformanceTests",
            dependencies: ["H3Conformance", "H3Foundation"]
        ),
    ]
)
