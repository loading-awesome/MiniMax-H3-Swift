import Foundation
import MLX
import MLXFast
import H3Foundation
import H3Hardware

/// The seam an attention implementation plugs into.
///
/// **There is no batch axis, and that is not an oversight.** MiniMax-H3 is
/// batch-size-1 by construction: the reference raises outright above one
/// (`comfy/ldm/minimax/model.py:509`) and the stack below it takes
/// `[S, hidden]` with nowhere to put a second sample. A backend that expects a
/// leading batch dimension is solving a problem this model does not have.
///
/// Backends are chosen once, at model build, and the choice is logged. They are
/// not swapped per block or per step: a backend that changes numerics mid-render
/// produces a trajectory neither backend would produce alone.
///
/// ## Numerics are part of the contract
///
/// Attention is where a "faster" implementation most often becomes a different
/// model — flash-style kernels reassociate the softmax accumulation, and
/// low-precision variants change where rounding happens. So a backend declares
/// its `equivalenceClass`: the relative-RMS band inside which its output is
/// considered the same answer as `sdpa`'s. Conformance runs per backend and
/// gates against that number rather than a single global tolerance. A backend
/// that cannot state its class has not been characterised and should not ship.
public protocol H3AttentionBackend: Sendable {

    /// Stable identifier, as it appears in config files and logs.
    static var identifier: String { get }

    /// Whether this backend can run here at all.
    static func isAvailable(on machine: Machine) -> Bool

    /// Relative RMS within which this backend agrees with `sdpa` on the same
    /// inputs. `0` means bit-identical.
    static var equivalenceClass: Float { get }

    /// Whether the backend materialises the `[S, S]` score matrix.
    ///
    /// The memory planner's quadratic term exists only for backends that do.
    /// A streaming backend removes it, which is the difference between a
    /// sequence length fitting and not fitting on a small machine — so this is
    /// consumed by planning, not just reported.
    static var materialisesScores: Bool { get }

    /// `[heads, S, headDim]` in, `[heads, S, headDim]` out.
    func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                scale: Float, mask: MLXArray?) -> MLXArray
}

/// MLX's fused scaled dot-product attention. The reference implementation for
/// this port and the fallback everywhere.
public struct SDPABackend: H3AttentionBackend {
    public static let identifier = "sdpa"
    public static let equivalenceClass: Float = 0
    public static let materialisesScores = true
    public static func isAvailable(on machine: Machine) -> Bool { true }

    public init() {}

    public func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                       scale: Float, mask: MLXArray?) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: queries, keys: keys,
                                          values: values, scale: scale,
                                          mask: mask)
    }
}

/// Resolves the configured backend against what the machine can actually run.
///
/// Falls back to `sdpa` and says so. A silent fallback would mean a render
/// whose numerics differ from the one the caller asked for, with no record of
/// why — which is the same class of failure as loading the wrong checkpoint
/// vendor, and is treated with the same suspicion.
public enum AttentionRegistry {

    public struct Selection: Sendable {
        public let identifier: String
        public let equivalenceClass: Float
        public let materialisesScores: Bool
        public let backend: any H3AttentionBackend
        public let reason: String
    }

    /// Backends compiled into this build, most preferred first.
    ///
    /// `sol` is expected here once its API, numerics and memory profile are
    /// specified. Registering it is a two-line change; characterising its
    /// equivalence class is the real work, and it has to happen before it can
    /// be selected by `auto`.
    static let available: [any H3AttentionBackend.Type] = [SDPABackend.self]

    public static func resolve(requested: String, machine: Machine) throws -> Selection {
        func make(_ t: any H3AttentionBackend.Type, _ reason: String) -> Selection {
            Selection(identifier: t.identifier, equivalenceClass: t.equivalenceClass,
                      materialisesScores: t.materialisesScores,
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
