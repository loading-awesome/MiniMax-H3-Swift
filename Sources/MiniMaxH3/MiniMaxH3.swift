// The public surface. Everything a caller needs, and nothing that would make
// an internal change a breaking one.
//
// MLX types deliberately do not appear here. A caller passes URLs, structs and
// enums and gets back files, progress and typed errors — which is what lets the
// attention backend, the quantisation scheme and the compute backend change
// without a major version.
@_exported import H3Foundation
@_exported import H3Hardware
@_exported import H3Catalog
@_exported import H3Recipes

public enum MiniMaxH3 {
    /// Semantic version of the public API.
    public static let version = "0.1.0-dev"
}
