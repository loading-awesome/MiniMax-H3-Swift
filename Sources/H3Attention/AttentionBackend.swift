import Foundation
import MLX
import MLXFast
import H3Foundation
import H3Hardware

/// Where in the render an attention call is happening.
///
/// **The first version of this seam passed only q, k, v, scale and a mask, and
/// that was wrong.** It assumed a backend is chosen once and then behaves the
/// same for every call — true of dense attention and false of every sparse
/// method worth having. Sol-Attn alone needs four things that signature cannot
/// express:
///
///  * **where in the schedule we are.** It runs dense before 20% and after 90%
///    of the sigma schedule, because the early steps set global structure and
///    the late ones set fine detail, and approximating either shows.
///  * **which block this is.** The first and last transformer blocks are the
///    most approximation-sensitive: their error reaches the output with no
///    later block to absorb it, so they are commonly excluded.
///  * **where the conditioning rows are.** H3 packs
///    `[text][cond][ref][audio][video]` into one self-attention sequence.
///    Sparsifying the conditioning costs lip-sync and prompt adherence, so
///    those rows are kept exact — as keys for every query (~3% of the cost), and
///    optionally as queries too, which makes the generated audio stream exact
///    for ~17% more.
///  * **how long the sequence is.** Below a few thousand tokens the routing
///    overhead exceeds the saving.
///
/// A context object is the difference between a seam a sparse backend can use
/// and one it has to be bolted around.
public struct AttentionContext: Sendable {
    /// 0-based index of the transformer block making this call.
    public let blockIndex: Int
    public let blockCount: Int
    /// Fraction of the sampling schedule completed, in [0, 1].
    public let scheduleProgress: Double
    /// Total packed sequence length.
    public let sequenceLength: Int
    /// `[start, stop)` of the target video segment in the packed sequence.
    ///
    /// Everything before `start` is conditioning — text, keyframes, references,
    /// target audio — and is what a sparse backend keeps exact.
    public let videoSpan: Range<Int>?

    public init(blockIndex: Int, blockCount: Int, scheduleProgress: Double,
                sequenceLength: Int, videoSpan: Range<Int>?) {
        self.blockIndex = blockIndex
        self.blockCount = blockCount
        self.scheduleProgress = scheduleProgress
        self.sequenceLength = sequenceLength
        self.videoSpan = videoSpan
    }
}

/// The seam an attention implementation plugs into.
///
/// **There is no batch axis, and that is not an oversight.** MiniMax-H3 is
/// batch-size-1 by construction: the reference raises outright above one
/// (`comfy/ldm/minimax/model.py:509`) and the stack below it takes
/// `[S, hidden]` with nowhere to put a second sample.
///
/// ## Numerics are part of the contract
///
/// Attention is where a "faster" implementation most often becomes a different
/// model. A backend therefore declares an `equivalenceClass`: the relative-RMS
/// band inside which its output counts as the same answer as `sdpa`'s.
/// Conformance runs per backend and gates against that number rather than one
/// global tolerance. A backend that cannot state its class has not been
/// characterised and must not ship.
///
/// A backend may also decline any individual call by returning `nil` from
/// `attend`, and the caller falls back to dense. That is how schedule windows,
/// dense-block exclusions and minimum-length gates are expressed without the
/// pipeline having to know any backend's policy.
public protocol H3AttentionBackend: Sendable {

    /// Stable identifier, as it appears in config files and logs.
    static var identifier: String { get }

    /// Whether this backend can run here at all.
    static func isAvailable(on machine: Machine) -> Bool

    /// Relative RMS within which this backend agrees with `sdpa`. `0` means
    /// bit-identical.
    static var equivalenceClass: Float { get }

    /// Whether the backend materialises the `[S, S]` score matrix.
    ///
    /// Consumed by the memory planner, not merely reported: a backend that
    /// never materialises scores removes the quadratic term from the activation
    /// estimate, which is the difference between a long sequence fitting and
    /// not fitting on a small machine.
    static var materialisesScores: Bool { get }

    /// Whether this backend benefits from Morton-ordered video tokens.
    ///
    /// Reordering is a *pipeline* transform, not an attention one — it permutes
    /// hidden states and position ids around the whole block stack — so the
    /// backend does not perform it. It only says whether it wants it, and the
    /// pipeline decides once. See `TokenOrdering`.
    static var prefersMortonOrder: Bool { get }

    /// `[heads, S, headDim]` in, `[heads, S, headDim]` out.
    ///
    /// Returns `nil` to decline this call, in which case the caller runs dense.
    func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                scale: Float, mask: MLXArray?, context: AttentionContext) -> MLXArray?
}

/// MLX's fused scaled dot-product attention. The reference implementation for
/// this port and the fallback everywhere. Never declines.
public struct SDPABackend: H3AttentionBackend {
    public static let identifier = "sdpa"
    public static let equivalenceClass: Float = 0
    public static let materialisesScores = true
    public static let prefersMortonOrder = false
    public static func isAvailable(on machine: Machine) -> Bool { true }

    public init() {}

    public func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                       scale: Float, mask: MLXArray?,
                       context: AttentionContext) -> MLXArray? {
        MLXFast.scaledDotProductAttention(queries: queries, keys: keys,
                                          values: values, scale: scale, mask: mask)
    }
}

/// Morton (Z-order) reordering of the target video tokens.
///
/// Not an attention concern, and deliberately modelled separately: it permutes
/// hidden states and position ids around the block stack, and is **exactly
/// neutral for dense attention** because tokens and their positions move
/// together and are restored before the final layer.
///
/// What it buys a block-sparse backend is that a 64-token block becomes a
/// compact 3D neighbourhood rather than a two-row strip of one frame, so a
/// block-level routing decision is about a region of the video rather than a
/// horizontal sliver of it.
///
/// **H3's temporal axis is not uniformly spaced** — `videoTGrid` places latent
/// frames on a stretched grid — so interleaving t, h and w equally can be worse
/// than Z-ordering within each frame and leaving frame order alone. That is why
/// the curve is a choice and why the per-frame variant is the safer default
/// here.
public enum TokenOrdering: String, Sendable, CaseIterable {
    case none
    /// Interleave t, h and w equally.
    case morton3D = "3d"
    /// Z-order within each frame; frame order untouched.
    case mortonPerFrame = "2d_frame"
}

/// Resolves the configured backend against what the machine can actually run.
///
/// Falls back to `sdpa` and says so. A silent fallback would mean a render
/// whose numerics differ from the one the caller asked for with no record of
/// why — the same class of failure as loading the wrong checkpoint vendor, and
/// treated with the same suspicion.
public enum AttentionRegistry {

    public struct Selection: Sendable {
        public let identifier: String
        public let equivalenceClass: Float
        public let materialisesScores: Bool
        public let ordering: TokenOrdering
        public let backend: any H3AttentionBackend
        public let reason: String
    }

    /// Backends compiled into this build, most preferred first.
    ///
    /// **Sol-Attn is not here yet, and the reason is specific rather than a
    /// to-do.** The published implementation
    /// (`kijai/ComfyUI-SolAttn_triton`, arXiv 2607.24027) is a Triton kernel,
    /// and Triton does not target Metal — so this is a reimplementation against
    /// `MLXFast.metalKernel`, not a binding. What ports directly is the
    /// structure, and it is all H3-shaped already:
    ///
    ///  * on-the-fly block thresholding with proxy-score reuse, `tau` the
    ///    threshold — reported as ~16% of blocks kept exact at 1.0, ~7% at 1.5,
    ///    ~2.7% at 2.0;
    ///  * an exact-KV sink over the packed conditioning rows;
    ///  * Morton reordering of the video span;
    ///  * dense first and last blocks;
    ///  * bf16 with head_dim 128, which is exactly this model.
    ///
    /// It reports ~2.1x end-to-end on video generation. Against the roofline
    /// measured here — the model already at ~90% of achievable matmul
    /// throughput, attention quadratic in a 15-20k sequence — that is the one
    /// available lever that is not a smaller model.
    static let available: [any H3AttentionBackend.Type] = [SDPABackend.self]

    public static func resolve(requested: String, machine: Machine,
                              ordering: TokenOrdering? = nil) throws -> Selection {
        func make(_ t: any H3AttentionBackend.Type, _ reason: String) -> Selection {
            Selection(identifier: t.identifier, equivalenceClass: t.equivalenceClass,
                      materialisesScores: t.materialisesScores,
                      ordering: ordering ?? (t.prefersMortonOrder ? .mortonPerFrame : .none),
                      backend: SDPABackend(), reason: reason)
        }
        if requested == "auto" {
            for t in available where t.isAvailable(on: machine) {
                return make(t, "auto-selected; highest-preference backend available here")
            }
            return make(SDPABackend.self, "auto: nothing else available")
        }
        guard let t = available.first(where: { $0.identifier == requested }) else {
            let known = available.map { $0.identifier }.joined(separator: ", ")
            throw H3Error.notImplemented(
                feature: "attention backend '\(requested)'",
                detail: "this build registers: \(known). A backend must declare an "
                      + "equivalence class before it can be selected, because conformance "
                      + "gates per backend.")
        }
        guard t.isAvailable(on: machine) else {
            throw H3Error.notImplemented(
                feature: "attention backend '\(requested)'",
                detail: "not available on \(machine.chip). Ask for 'auto' to fall back "
                      + "explicitly rather than silently.")
        }
        return make(t, "requested explicitly")
    }
}
