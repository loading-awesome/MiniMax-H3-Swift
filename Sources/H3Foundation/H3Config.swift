// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// MiniMax-H3 architecture constants.
///
/// Every value was read out of the checkpoint or re-derived from tensor shapes
/// on the CUDA reference — see `docs/MODEL_MAP.md`. Nothing here is quoted from
/// documentation. ComfyUI derives all of these from tensor shapes at load time
/// (`comfy/model_detection.py:362`), so they are also what a loader should
/// verify against rather than assume.
package struct H3Config: Sendable, Equatable {
    package var hiddenSize: Int = 5376
    package var numLayers: Int = 50
    package var tokenRefinerLayers: Int = 2
    package var numHeads: Int = 56
    package var headDim: Int = 128
    package var ffnHidden: Int = 14336
    package var videoLatentDim: Int = 24
    package var audioLatentDim: Int = 32
    /// `[t, h, w]` — video patchify. Note h/w are 2, t is 1.
    package var patchSize: [Int] = [1, 2, 2]
    package var textDim: Int = 5120
    package var timestepInputDim: Int = 256
    package var timeEmbedHidden: Int = 5376
    package var timeEmbedDim: Int = 2688
    package var adalnOutFeatures: Int = 96_768
    package var finalAdalnOutFeatures: Int = 10_752
    package var ropeInvFreqLen: Int = 16
    package var normEps: Float = 1e-5
    package var qkNormEps: Float = 1e-5
    package var finalNormEps: Float = 1e-5

    package init() {}

    /// `heads * headDim` — the attention inner width. qkv_proj emits 3x this.
    package var innerDim: Int { numHeads * headDim }

    /// Video patchify collapses `patchSize` — one token per 1x2x2 latent cell.
    package var videoPatchDim: Int { videoLatentDim * patchSize.reduce(1, *) }
}

/// Flow-matching shift, applied in two places that MUST agree: the sampler's
/// sigma schedule and the DiT's internal grid mapping. The DiT falls back to
/// these same constants when `transformer_options` carries no override
/// (`comfy/ldm/minimax/model.py:527`), which is why no ComfyUI template ships
/// the SigmaShift node and the schedule is still correct.
package enum H3Shift {
    package static let video: Double = 12.0
    package static let audio: Double = 3.0
}

/// Audio constants. The audio VAE is the only component published at full
/// precision — do not downcast it to match the video VAE.
package enum H3Audio {
    package static let sampleRate: Int = 32_000
    /// `sampleRate / hopLength` — latents per second of audio.
    package static let latentFPS: Int = 40
}

/// Conditioning-row noise augmentation. These are the timesteps the cond rows
/// are pinned to, and simultaneously the mixing weight applied to their latents
/// (`VISUAL_COND_TIMESTEP` / `AUDIO_COND_TIMESTEP` in the reference).
///
/// Audio sits at exactly 1.0, so audio conditioning rows are never noised.
/// Visual sits just below, so they always are — and the reference draws that
/// noise from a `torch.Generator("cpu")`. See `docs/FRAGILE_CONTRACTS.md` #21.
package enum H3Cond {
    package static let visualNoiseAug: Float = 0.999
    package static let audioNoiseAug: Float = 1.0
}

package enum H3Video {
    package static let fps: Int = 24
    /// Frame counts live on a 17k+5 lattice; off-lattice values snap upward.
    package static let lattice: Int = 17
    package static let latticeOffset: Int = 5
    /// VAE spatial downsample. Latent H/W are `pixels / 16`.
    package static let spatialDownsample: Int = 16
    /// Trained duration range, in frames. Below this is out of distribution;
    /// above is untested. 124 frames is about 5s at 24fps.
    package static let trainedFrameRange: ClosedRange<Int> = 124...362
}

/// Which model answers a render, and therefore how many steps it takes.
///
/// These are two different checkpoints, not two settings on one. `turbo` is a
/// DMD-distilled student that jumps between four fixed noise levels — its step
/// count is a property of its weights, not a preference, so `--steps` does not
/// apply to it. `standard` is the base model following the flow schedule at
/// whatever step count is asked for, which is where the step cache earns its
/// keep: at four rungs warmup and cooldown already cover every step, and only a
/// longer schedule leaves anything for it to skip.
///
/// They are addressed in the configuration by a key prefix on the same
/// per-partition table the precision uses, so `turbo` at `bf16` reads
/// `turbo_bf16`. A distilled checkpoint filed under a plain precision key would
/// make `bf16` name a model rather than a dtype, which is exactly the confusion
/// this enum exists to end.
package enum H3RenderProfile: String, Sendable, CaseIterable, Codable {
    case turbo
    case standard

    /// The configuration key this profile reads for a given precision.
    package func checkpointKey(precision: String) -> String {
        self == .turbo ? "turbo_\(precision)" : precision
    }

    /// The step count to use when the caller did not ask for one. A distilled
    /// checkpoint overrides this from its own metadata; the base model does not.
    package var defaultSteps: Int { self == .turbo ? 4 : 20 }
}
