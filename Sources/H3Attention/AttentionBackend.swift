// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

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
///  * **where in the schedule we are.** The paper specifies a dense *warm-up* —
///    the first 20% of steps and the first layer — attributed to prior work and
///    never ablated. It specifies no late-schedule dense phase; the ComfyUI
///    port's `end_percent` is the implementer's addition. Either way the
///    backend has to know where in the schedule it is.
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
package struct AttentionContext: Sendable {
    /// 0-based index of the transformer block making this call.
    package let blockIndex: Int
    package let blockCount: Int
    /// Fraction of the sampling schedule completed, in [0, 1].
    package let scheduleProgress: Double
    /// Total packed sequence length.
    package let sequenceLength: Int
    /// `[start, stop)` of the target video segment in the packed sequence.
    ///
    /// Everything before `start` is conditioning — text, keyframes, references,
    /// target audio — and is what a sparse backend keeps exact.
    package let videoSpan: Range<Int>?

    package init(blockIndex: Int, blockCount: Int, scheduleProgress: Double,
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
package protocol H3AttentionBackend: Sendable {

    /// Required so the registry can instantiate the backend it selected.
    ///
    /// Without it `resolve` can only return a concrete type it names itself,
    /// which is how the first version came to report one identifier while
    /// handing back another instance.
    init()

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
    /// **False for H3 as far as NVIDIA is concerned**: their official H3
    /// integration states the packed video tail "is already a contiguous
    /// grid-ordered block, and the routing works on it directly". Morton is
    /// absent from the paper and from the official backend; it is a ComfyUI-port
    /// addition aimed at Wan. Kept as a seam because it is free to keep and the
    /// claim is worth testing, but it defaults off.
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
package struct SDPABackend: H3AttentionBackend {
    package static let identifier = "sdpa"
    package static let equivalenceClass: Float = 0
    package static let materialisesScores = true
    package static let prefersMortonOrder = false
    package static func isAvailable(on machine: Machine) -> Bool { true }

    package init() {}

    package func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                       scale: Float, mask: MLXArray?,
                       context: AttentionContext) -> MLXArray? {
        // MLX's fused kernel takes `[B, heads, S, headDim]` and traps on rank 3,
        // so the batch axis this protocol deliberately does not have is added
        // here and removed again. Putting it in the protocol instead would put a
        // dimension H3 can never use into every future backend's signature.
        let o = MLXFast.scaledDotProductAttention(
            queries: queries.expandedDimensions(axis: 0),
            keys: keys.expandedDimensions(axis: 0),
            values: values.expandedDimensions(axis: 0),
            scale: scale, mask: mask)
        return o.squeezed(axis: 0)
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
package enum TokenOrdering: String, Sendable, CaseIterable {
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
package enum AttentionRegistry {

    package struct Selection: Sendable {
        package let identifier: String
        package let equivalenceClass: Float
        package let materialisesScores: Bool
        package let ordering: TokenOrdering
        package let backend: any H3AttentionBackend
        package let reason: String
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
    ///  * per-query-block thresholding, `tau_i = mu_i + beta*sigma_i`, where
    ///    beta is a Gaussian tail cutoff: density is `1 - Phi(beta)`, so ~16% of
    ///    blocks stay exact at 1.0, ~7% at 1.5, ~2.7% at 2.0;
    ///  * a rank-1 correction for rejected blocks — pooled key times summed
    ///    value, into both the numerator and the denominator — which is what
    ///    makes it an approximation rather than a mask;
    ///  * an exact-KV sink over the packed conditioning rows, which for H3 is
    ///    not optional: NVIDIA's own notes record a text-only sink where "the
    ///    picture scored best of its set while its dialogue fell apart";
    ///  * bf16 with head_dim 128, which is exactly this model.
    ///
    /// **It is not the biggest lever available, and an earlier version of this
    /// comment said it was.** NVIDIA's H3 breakdown puts cross-step caching
    /// first (1.534x -> 3.95x of a 3.95x total), fused RMS-AdaLN second, and
    /// Sol-Attn at 1.25x additional. Amdahl agrees: attention is 37% of DiT
    /// FLOPs at our verified shape, so sparsification alone caps near 1.5x
    /// there — rising with sequence length, which is where it earns its place.
    /// See docs/SOL_ATTN.md.
    static let available: [any H3AttentionBackend.Type] = [SDPABackend.self]

    package static func resolve(requested: String, machine: Machine,
                               ordering: TokenOrdering? = nil) throws -> Selection {
        try resolve(requested: requested, machine: machine, ordering: ordering,
                    backends: available)
    }

    /// The registry above is a `let`, so the tests supply their own list here
    /// rather than mutating shared state. With one backend compiled in there is
    /// no way to catch a resolver that returns the wrong *instance*; a second,
    /// fake backend is what makes that observable.
    static func resolve(requested: String, machine: Machine,
                        ordering: TokenOrdering?,
                        backends available: [any H3AttentionBackend.Type]) throws -> Selection {
        // `t.init()`, not a named type. An earlier version wrote `SDPABackend()`
        // here while reporting `t.identifier` and `t.equivalenceClass` — correct
        // only for as long as SDPA was the sole registered backend, and silently
        // wrong the moment a second one existed: a Selection labelled `sol`,
        // carrying sol's equivalence class, running dense attention. That is the
        // same failure shape as loading a checkpoint under the wrong vendor —
        // every field plausible, the answer from a different implementation than
        // the one named. `backendIsWhatWasNamed` in the tests pins it.
        func make(_ t: any H3AttentionBackend.Type, _ reason: String) -> Selection {
            Selection(identifier: t.identifier, equivalenceClass: t.equivalenceClass,
                      materialisesScores: t.materialisesScores,
                      ordering: ordering ?? (t.prefersMortonOrder ? .mortonPerFrame : .none),
                      backend: t.init(), reason: reason)
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
