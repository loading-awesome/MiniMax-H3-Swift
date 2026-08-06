// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Foundation
import H3Hardware

/// Sol-Attn as a registered attention backend.
///
/// ## This is opt-in, and the reason is what the class turned out to be
///
/// `AttentionRegistry.available` lists this **after** `SDPABackend`, so `auto`
/// still resolves to dense and reaching this requires asking for it by name.
///
/// The class below is now measured here, against q/k/v captured from a real
/// render on this machine at `S = 15993`, progress 0.25, beta 1.2 with the sink
/// (docs/SOL_ATTN.md §11):
///
/// | block | rel_rms vs dense | conditioning rows |
/// |---|---|---|
/// | 0  | 0.193 | 0.183 |
/// | 24 | **0.287** | 0.369 |
/// | 49 | 0.222 | 0.481 |
///
/// **This port gates DiT blocks against CUDA at rel_rms 8.5e-03 to 2.3e-02.**
/// 0.287 is more than ten times outside that, which is not a defect — it is
/// what sparse attention costs, and it is why the protocol has per-backend
/// classes at all. But it does mean the question "is this good enough" cannot be
/// answered by a tensor comparison. It has to be answered on rendered output:
/// WER on the generated waveform and face-landmark detection across frames,
/// against a dense render of the same seed. Until that A/B exists, dense stays
/// the default.
///
/// ## What is established
///
/// The kernel reproduces `SolAttnReference` to rel_rms < 2e-3 at head dim 128,
/// including the short tail block, the conditioning sink, bf16 input and routing
/// blocks spanning several threadgroups; and the reference is pinned to dense
/// exactly in the two cases where the method degenerates to it. So the
/// implementation is correct with respect to the method, and the number above
/// is a fact about Sol-Attn on H3 rather than about this port.
///
/// The measurement also reproduces §8's qualitative finding from a completely
/// independent implementation — **the middle of the stack is hardest to
/// approximate**, block 24 worst in both — while sitting consistently ~0.04-0.08
/// above the Triton kernel's figures at the same beta. The most likely cause is
/// the Cornish-Fisher correction, which kijai's node applies using the third and
/// fourth moments and this does not. §8 measured H3's proxy distribution at
/// excess kurtosis +2.01, so that correction is doing real work there.
package struct SolAttnBackend: H3AttentionBackend {

    package static let identifier = "sol"

    /// Measured on this machine, worst of blocks 0, 24 and 49 at beta 1.2 with
    /// the exact-KV sink. Rounded up from 0.287, not down.
    ///
    /// Three blocks at one point in the schedule is a floor on the real figure
    /// rather than a survey of it; a block or a step that was not sampled can
    /// only make it worse.
    package static let equivalenceClass: Float = 0.29

    /// The `[S, S]` score matrix is never built — the kernel keeps one tile of
    /// keys in threadgroup memory and a running softmax in registers.
    ///
    /// The memory planner consumes this rather than merely reporting it, so it
    /// removes the quadratic term from the activation estimate. At 15,731 packed
    /// tokens that term is what decides whether a long render fits.
    package static let materialisesScores = false

    /// Off, on NVIDIA's evidence: their H3 integration states the packed video
    /// tail "is already a contiguous grid-ordered block, and the routing works
    /// on it directly". Morton is a ComfyUI-port addition aimed at Wan.
    package static let prefersMortonOrder = false

    package static func isAvailable(on machine: Machine) -> Bool {
        // Any Apple silicon GPU compiles the kernel; there is no feature here
        // beyond `quad_shuffle_xor` and threadgroup memory.
        true
    }

    package let config: SolAttnConfig

    /// The registry constructs backends through `init()`, so environment
    /// overrides are read here. Anything non-default is announced on stderr
    /// once, because a render tuned by environment variable that does not say
    /// so is the same class of failure as a silent backend fallback — the
    /// output is fine, and nobody can reproduce it.
    package init() {
        let c = SolAttnConfig(environment: ProcessInfo.processInfo.environment)
        self.init(config: c)
        if let overrides = c.overridesDescription {
            FileHandle.standardError.write(Data("    sol-attn overrides: \(overrides)\n".utf8))
        }
    }

    package init(config: SolAttnConfig) { self.config = config }

    package func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                       scale: Float, mask: MLXArray?,
                       context: AttentionContext) -> MLXArray? {
        // A mask is a conditioning contract this backend does not implement:
        // the selection would have to intersect it per block, and H3's packed
        // self-attention passes none. Declining is the honest answer.
        guard mask == nil else { return nil }
        guard config.admits(context) else { return nil }

        guard let out = SolAttnMetalKernel.attend(
            queries: queries, keys: keys, values: values, scale: scale,
            config: config, videoSpan: context.videoSpan) else { return nil }
        return out.asType(queries.dtype)
    }
}
