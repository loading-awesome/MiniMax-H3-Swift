// The public surface. Everything a caller needs, and nothing that would make
// an internal change a breaking one.
//
// MLX types deliberately do not appear here. A caller passes URLs, structs and
// enums and gets back files, progress and typed errors — which is what lets the
// attention backend, the quantisation scheme and the compute backend change
// without a major version.
// The rule is enforced rather than merely stated: the pipeline's phase types
// carry MLXArray, so they are internal to H3Pipeline. What crosses this line is
// RenderRequest, RenderProgress, RenderResult and `H3Pipeline.render` — URLs,
// structs, enums and typed errors.
@_exported import H3Foundation
@_exported import H3Hardware
@_exported import H3Catalog
@_exported import H3Recipes
@_exported import H3Pipeline

public enum MiniMaxH3 {
    /// Semantic version of the public API.
    public static let version = "0.1.0-dev"
}
