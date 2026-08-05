import Foundation
import H3Foundation

/// Which checkpoint precision this machine can hold for a given render, and the
/// arithmetic behind the answer.
///
/// **The central fact, and the one that makes this more than a lookup table:
/// activations do not shrink when the weights do.** They scale with packed
/// sequence length, and attention is quadratic in it. So a smaller checkpoint
/// does not buy a bigger render — on a memory-constrained machine the plan has
/// to reach for resolution and duration as well as precision, and a planner
/// that only swaps weights will confidently produce a configuration that still
/// does not fit.
///
/// The second fact is that two different quantities both get called "memory
/// use" and only one of them is what jetsam counts:
///
///  * `MLX.GPU.peakMemory` counts what MLX **allocated** — activations, mostly.
///  * The checkpoints are **memory-mapped**, so they never enter that figure at
///    all, and on the run that was killed they were 118 GB of it.
///
/// This planner models resident footprint, weights included, because that is
/// the number that gets a process killed.
public struct MemoryPlan: Sendable, Equatable {

    /// Sizes as published, in bytes. Measured from the files, not estimated.
    public enum Precision: String, Sendable, CaseIterable {
        case bf16
        case int8
        case prunedBF16 = "pruned_bf16"
        case prunedInt8 = "pruned_int8"

        /// DiT bytes for one partition.
        public var ditBytes: UInt64 {
            switch self {
            case .bf16:       66_280_000_000
            case .int8:       34_040_000_000
            case .prunedBF16: 41_400_000_000
            case .prunedInt8: 22_140_000_000
            }
        }

        /// Whether the weights are an approximation of the released ones.
        ///
        /// The `pruned` variants replace AdaLN with a rank-64 curve: the
        /// projection is `[96768, 64]` rather than `[96768, 2688]`, which is
        /// exactly the 24.9 GB they save. That changes numerics by
        /// construction, so it is a different model and is labelled one.
        public var isApproximate: Bool {
            self == .prunedBF16 || self == .prunedInt8
        }

        /// Whether the runtime keeps the weights quantised.
        ///
        /// **Not yet true for any variant.** int8 checkpoints are stored as I8
        /// plus per-output-channel F32 scales, and dequantising them at load
        /// returns the full bf16 footprint — saving disk and nothing else.
        /// Keeping them resident needs a quantised matmul whose layout differs
        /// from MLX's group-wise one. Until that lands, `residentBytes` says so
        /// rather than promising a saving that does not exist.
        public var isResidentQuantised: Bool { false }

        public var residentDITBytes: UInt64 {
            isResidentQuantised ? ditBytes : dequantisedDITBytes
        }

        /// What the DiT costs in memory once loaded, as opposed to on disk.
        var dequantisedDITBytes: UInt64 {
            switch self {
            case .bf16, .int8:             66_280_000_000
            case .prunedBF16, .prunedInt8: 41_400_000_000
            }
        }

        public var textEncoderBytes: UInt64 {
            self == .bf16 || self == .prunedBF16 ? 51_510_000_000 : 26_720_000_000
        }
    }

    public static let vaeBytes: UInt64 = 5_820_000_000

    /// A phase of the render, with what it holds while it runs.
    ///
    /// The pipeline loads and releases per phase, so the peak is the largest
    /// phase and not the sum. That ordering is a memory contract: encoding
    /// every condition *before* the DiT is loaded is what keeps the text
    /// encoder and the DiT from being resident together.
    public enum Phase: String, Sendable, CaseIterable {
        case textEncode, vaeEncode, sampling, decode
    }

    public let precision: Precision
    public let phaseBytes: [Phase: UInt64]
    public let peakBytes: UInt64
    public let availableBytes: UInt64
    public let headroomBytes: Int64

    public var peakGB: Double { Double(peakBytes) / 1e9 }
    public var availableGB: Double { Double(availableBytes) / 1e9 }
    public var fits: Bool { headroomBytes > 0 }

    /// Activation bytes during sampling, from packed sequence length.
    ///
    /// Linear in `S` for the residual stream and the GEMM working set,
    /// quadratic in `S` for attention scores. The coefficients are fitted to a
    /// measured point — 53.1 GB of MLX-allocated peak at S = 20,392 on the
    /// verified configuration — and should be re-fitted whenever the attention
    /// backend changes, because a backend that never materialises the score
    /// matrix removes the quadratic term entirely.
    public static func samplingActivationBytes(packedTokens S: Int) -> UInt64 {
        let s = Double(S)
        let linear = 1.6e6 * s          // residual stream, qkv, mlp working set
        let quadratic = 2.9e1 * s * s   // attention scores
        return UInt64(max(0, linear + quadratic))
    }

    public static func plan(precision: Precision, packedTokens: Int,
                            availableBytes: UInt64,
                            marginFraction: Double = 0.15) -> MemoryPlan {
        let act = samplingActivationBytes(packedTokens: packedTokens)
        let phases: [Phase: UInt64] = [
            .textEncode: precision.textEncoderBytes + 2_000_000_000,
            .vaeEncode:  vaeBytes + 2_000_000_000,
            .sampling:   precision.residentDITBytes + act,
            .decode:     vaeBytes + UInt64(Double(act) * 0.25),
        ]
        let peak = phases.values.max() ?? 0
        let withMargin = UInt64(Double(peak) * (1.0 + marginFraction))
        return MemoryPlan(precision: precision, phaseBytes: phases, peakBytes: peak,
                          availableBytes: availableBytes,
                          headroomBytes: Int64(availableBytes) - Int64(withMargin))
    }

    /// The best precision that fits, or nil when none does.
    ///
    /// Ordered most faithful first. `allowApproximate` gates the two `pruned`
    /// variants, because falling back to them silently would hand somebody a
    /// different model and let them compare its output to the gated one.
    public static func best(packedTokens: Int, availableBytes: UInt64,
                            allowApproximate: Bool) -> MemoryPlan? {
        let order: [Precision] = allowApproximate
            ? [.bf16, .int8, .prunedBF16, .prunedInt8]
            : [.bf16, .int8]
        for p in order {
            let plan = plan(precision: p, packedTokens: packedTokens,
                            availableBytes: availableBytes)
            if plan.fits { return plan }
        }
        return nil
    }

    public var explanation: String {
        var out = [String]()
        out.append(String(format: "  precision      %@%@", precision.rawValue,
                          precision.isApproximate ? "  (approximate weights)" : ""))
        for phase in Phase.allCases {
            let b = phaseBytes[phase] ?? 0
            out.append(String(format: "  %-14@ %6.1f GB%@", phase.rawValue as NSString,
                              Double(b) / 1e9, b == peakBytes ? "   <- peak" : ""))
        }
        out.append(String(format: "  available      %6.1f GB", availableGB))
        out.append(String(format: "  headroom       %+6.1f GB after a 15%% margin",
                          Double(headroomBytes) / 1e9))
        if !precision.isResidentQuantised && (precision == .int8 || precision == .prunedInt8) {
            out.append("  note: int8 weights are dequantised at load, so this saves disk and "
                       + "not memory until a resident quantised matmul lands.")
        }
        return out.joined(separator: "\n")
    }
}
