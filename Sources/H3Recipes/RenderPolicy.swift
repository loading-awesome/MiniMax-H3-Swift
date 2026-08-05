import Foundation
import H3Foundation

/// Refuses render configurations that are known to produce bad output or to be
/// unaffordable, and says exactly why.
///
/// This exists because the defaults were both wrong and the failure was silent.
/// `--duration 4` snaps to 107 frames — the only duration value in the whole
/// 4-15 s range that lands outside the model's trained 124-362 window — and
/// `--resolution 2k` is 133,200 packed tokens, an estimated **14.6 hours** for
/// a 20-step render on an M3 Ultra. Together they were the out-of-box config.
///
/// A render at 107 frames and 6 steps produced 52 flash events and visible
/// green blobs; the same prompt at 124 frames and 20 steps produced zero, with
/// a face detected in 124/124 frames and its dialogue transcribed correctly.
/// That is the difference between "the port is broken" and "the config was" —
/// and it cost an evening to tell apart, which is why these are refusals rather
/// than warnings. A warning was already printed. It was ignored.
///
/// Every threshold here is measured, and each violation carries the measurement
/// in its text so the caller can judge rather than obey.
package enum H3RenderPolicy {

    /// The configuration that has been verified end to end on this port:
    /// 864x480, 124 frames, 20 `res_multistep` steps.
    ///
    /// It is simultaneously the parity-gated shape (225 gating taps inside the
    /// CUDA-measured class) and the shape proven to render a detectable,
    /// intelligible talking head. Nothing else has both properties.
    package static let verifiedWidth = 864
    package static let verifiedHeight = 480
    package static let verifiedFrames = 124
    package static let verifiedSteps = 20
    package static let verifiedSeconds = 1381.0

    /// Below this, `res_multistep` has not converged and the output flickers.
    /// Measured: 6 steps gave 52 flash events and a worst per-tile luminance
    /// jump of 0.559 between adjacent frames of a static shot; 20 gave 0 and a
    /// frame-to-frame luminance sd of 0.0009.
    package static let minSteps = 20

    /// Sustained matmul throughput measured by `h3-parity bench-gemm` at the
    /// model's own shapes. bf16, fp16 and fp32 all land within 6% of this, so
    /// it is not a precision-dependent number.
    package static let measuredTFLOPS = 15.28

    /// Wall-clock beyond which a render is refused without an explicit opt-in.
    /// Three hours is roughly eight times the verified configuration.
    package static let maxEstimatedSeconds = 3 * 3600.0

    package struct Violation: Sendable {
        package let rule: String
        package let reason: String
        package let remedy: String
    }

    /// Estimated sampling seconds, from first principles rather than a fitted
    /// curve: count the DiT's FLOPs at this packed length and divide by the
    /// throughput `bench-gemm` measured. Attention is quadratic in sequence
    /// length, which is why doubling the resolution does far worse than double
    /// the time.
    package static func estimatedSeconds(tokens: Int, steps: Int,
                                        config: H3Config = H3Config()) -> Double {
        let s = Double(tokens)
        let h = Double(config.hiddenSize)
        let f = Double(config.ffnHidden)
        let inner = Double(config.innerDim)
        let perBlock = 2 * s * h * (3 * inner)      // qkv
                     + 4 * s * s * inner            // QK^T and AV
                     + 2 * s * inner * h            // out
                     + 3 * 2 * s * h * f            // SwiGLU
        let total = perBlock * Double(config.numLayers) * Double(steps)
        return total / (measuredTFLOPS * 1e12)
    }

    /// Every reason this configuration is worse than what has been verified.
    /// Empty means it is safe to run.
    package static func check(width: Int, height: Int, frameCount: Int,
                             steps: Int, tokens: Int,
                             config: H3Config = H3Config()) -> [Violation] {
        var out: [Violation] = []

        if !H3Video.trainedFrameRange.contains(frameCount) {
            let lo = H3Video.trainedFrameRange.lowerBound
            let hi = H3Video.trainedFrameRange.upperBound
            out.append(Violation(
                rule: "frame count outside the trained range",
                reason: "\(frameCount) frames is outside \(lo)-\(hi). Frame counts snap up "
                      + "onto the 17k+5 lattice, so 4 s becomes 107 — still under the floor. "
                      + "Measured at 107 frames and 6 steps: 52 flash events, worst per-tile "
                      + "luminance jump 0.559, visible green blobs. The artifacts mimic a "
                      + "decoder bug and are not one.",
                remedy: "ask for 5 s or more (5 s -> 124 frames, the verified shape)"))
        }

        if steps < minSteps {
            out.append(Violation(
                rule: "step count below the verified floor",
                reason: "\(steps) steps is under \(minSteps). Measured: 6 steps produced 52 "
                      + "flash events and frame-to-frame luminance sd 0.0087; 20 steps "
                      + "produced 0 events and sd 0.0009 on the same prompt.",
                remedy: "use --steps \(minSteps) or more"))
        }

        let est = estimatedSeconds(tokens: tokens, steps: steps, config: config)
        if est > maxEstimatedSeconds {
            out.append(Violation(
                rule: "estimated render time",
                reason: String(format: "%dx%d at %d frames is %d packed tokens, an estimated "
                               + "%.1f hours for %d steps at the %.2f TFLOP/s this machine "
                               + "was measured to sustain. Attention is quadratic in sequence "
                               + "length, so resolution costs far more than linearly.",
                               width, height, frameCount, tokens,
                               est / 3600, steps, measuredTFLOPS),
                remedy: String(format: "use --resolution 768p or --width %d --height %d "
                               + "(the verified shape, ~%.0f min)",
                               verifiedWidth, verifiedHeight, verifiedSeconds / 60)))
        }

        return out
    }

    /// A one-line estimate for the run banner, whether or not anything is wrong.
    package static func estimateLine(tokens: Int, steps: Int,
                                    config: H3Config = H3Config()) -> String {
        let est = estimatedSeconds(tokens: tokens, steps: steps, config: config)
        let rel = est / verifiedSeconds
        return String(format: "%d packed tokens, estimated %.0f min sampling (%.1fx the "
                      + "verified 864x480x124 config)", tokens, est / 60, rel)
    }
}
