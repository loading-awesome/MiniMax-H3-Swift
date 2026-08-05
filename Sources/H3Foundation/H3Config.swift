import Foundation

/// MiniMax-H3 architecture constants.
///
/// Every value was read out of the checkpoint or re-derived from tensor shapes
/// on the CUDA reference — see `docs/MODEL_MAP.md`. Nothing here is quoted from
/// documentation. ComfyUI derives all of these from tensor shapes at load time
/// (`comfy/model_detection.py:362`), so they are also what a loader should
/// verify against rather than assume.
public struct H3Config: Sendable, Equatable {
    public var hiddenSize: Int = 5376
    public var numLayers: Int = 50
    public var tokenRefinerLayers: Int = 2
    public var numHeads: Int = 56
    public var headDim: Int = 128
    public var ffnHidden: Int = 14336
    public var videoLatentDim: Int = 24
    public var audioLatentDim: Int = 32
    /// `[t, h, w]` — video patchify. Note h/w are 2, t is 1.
    public var patchSize: [Int] = [1, 2, 2]
    public var textDim: Int = 5120
    public var timestepInputDim: Int = 256
    public var timeEmbedHidden: Int = 5376
    public var timeEmbedDim: Int = 2688
    public var adalnOutFeatures: Int = 96_768
    public var finalAdalnOutFeatures: Int = 10_752
    public var ropeInvFreqLen: Int = 16
    public var normEps: Float = 1e-5
    public var qkNormEps: Float = 1e-5
    public var finalNormEps: Float = 1e-5

    public init() {}

    /// `heads * headDim` — the attention inner width. qkv_proj emits 3x this.
    public var innerDim: Int { numHeads * headDim }

    /// Video patchify collapses `patchSize` — one token per 1x2x2 latent cell.
    public var videoPatchDim: Int { videoLatentDim * patchSize.reduce(1, *) }
}

/// Flow-matching shift, applied in two places that MUST agree: the sampler's
/// sigma schedule and the DiT's internal grid mapping. The DiT falls back to
/// these same constants when `transformer_options` carries no override
/// (`comfy/ldm/minimax/model.py:527`), which is why no ComfyUI template ships
/// the SigmaShift node and the schedule is still correct.
public enum H3Shift {
    public static let video: Double = 12.0
    public static let audio: Double = 3.0
}

/// Audio constants. The audio VAE is the only component published at full
/// precision — do not downcast it to match the video VAE.
public enum H3Audio {
    public static let sampleRate: Int = 32_000
    /// `sampleRate / hopLength` — latents per second of audio.
    public static let latentFPS: Int = 40
}

/// Conditioning-row noise augmentation. These are the timesteps the cond rows
/// are pinned to, and simultaneously the mixing weight applied to their latents
/// (`VISUAL_COND_TIMESTEP` / `AUDIO_COND_TIMESTEP` in the reference).
///
/// Audio sits at exactly 1.0, so audio conditioning rows are never noised.
/// Visual sits just below, so they always are — and the reference draws that
/// noise from a `torch.Generator("cpu")`. See `docs/FRAGILE_CONTRACTS.md` #21.
public enum H3Cond {
    public static let visualNoiseAug: Float = 0.999
    public static let audioNoiseAug: Float = 1.0
}

public enum H3Video {
    public static let fps: Int = 24
    /// Frame counts live on a 17k+5 lattice; off-lattice values snap upward.
    public static let lattice: Int = 17
    public static let latticeOffset: Int = 5
    /// VAE spatial downsample. Latent H/W are `pixels / 16`.
    public static let spatialDownsample: Int = 16
    /// Trained duration range, in frames. Below this is out of distribution;
    /// above is untested. 124 frames is about 5s at 24fps.
    public static let trainedFrameRange: ClosedRange<Int> = 124...362
}
