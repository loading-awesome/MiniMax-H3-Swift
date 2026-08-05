// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXNN
import H3Foundation

/// The DAC encoder side of the MiniMax H3 stereo audio VAE.
///
/// Everything here runs in **NLC** layout because that is what MLX's `conv1d`
/// takes, while the reference is PyTorch's NCL. Weights are transposed once at
/// load (`[O, I, K] -> [O, K, I]`) and per-channel parameters are reshaped to
/// `[1, 1, C]` rather than the reference's `[1, C, 1]`.
///
/// Two things about `encode` are easy to assume wrong, and both are load-bearing:
///
///  * **It returns the posterior mean. There is no sampling.** `logs_proj` is in
///    the checkpoint and the reference never calls it — the comment there says
///    so outright. An encoder that samples injects run-to-run variance the
///    reference does not have and can never match a golden.
///  * **The waveform is right-padded with zeros to a multiple of 800 samples**
///    (the hop length) before anything else touches it. Skipping that silently
///    truncates the tail of the clip.
package struct Snake1d {
    /// Stored **raw**, not in log scale — that is `SnakeBeta`, which is the
    /// decoder's activation. `Snake1d` uses alpha directly and passes it as
    /// beta as well, so the two are the same tensor.
    package let alpha: MLXArray

    package init(alpha: MLXArray) {
        self.alpha = alpha
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        // the checkpoint stores [1, C, 1]; NLC wants [1, 1, C]
        let a = alpha.reshaped([1, 1, alpha.size])
        return snake(x, alpha: a, beta: a)
    }
}

/// `Snake1d -> Conv1d(k=7, dilation=d) -> Snake1d -> Conv1d(k=1)`, plus residual.
///
/// The dilation is the whole point of the unit — the three units in a block run
/// at 1, 3 and 9, which is what gives the encoder its receptive field. Running
/// them all at dilation 1 leaves every shape correct.
struct AudioResidualUnit {
    let act1: Snake1d
    let conv1: VaeConv1d
    let act2: Snake1d
    let conv2: VaeConv1d

    init(dilation: Int, prefix: String, weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.act1 = Snake1d(alpha: try get(prefix + "block.0.alpha"))
        self.conv1 = VaeConv1d(weight: try get(prefix + "block.1.weight").transposed(0, 2, 1),
                               bias: try get(prefix + "block.1.bias"),
                               stride: 1, padding: ((7 - 1) * dilation) / 2, dilation: dilation)
        self.act2 = Snake1d(alpha: try get(prefix + "block.2.alpha"))
        self.conv2 = VaeConv1d(weight: try get(prefix + "block.3.weight").transposed(0, 2, 1),
                               bias: try get(prefix + "block.3.bias"),
                               stride: 1, padding: 0, dilation: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv2(act2(conv1(act1(x))))
        // The reference centre-crops the residual when the block shortens it.
        // With padding = 3*dilation the lengths match, so this is a guard, not
        // a code path — but it is the reference's guard.
        let pad = (x.dim(1) - y.dim(1)) / 2
        let skip = pad > 0 ? x[0..., pad ..< (x.dim(1) - pad), 0...] : x
        return y + skip
    }
}

/// Three residual units at dilation 1/3/9, then a strided downsample.
struct AudioEncoderBlock {
    let units: [AudioResidualUnit]
    let act: Snake1d
    let down: VaeConv1d

    init(stride: Int, prefix: String, weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.units = try [1, 3, 9].enumerated().map { i, d in
            try AudioResidualUnit(dilation: d, prefix: prefix + "block.\(i).", weights: weights)
        }
        self.act = Snake1d(alpha: try get(prefix + "block.3.alpha"))
        // kernel is 2*stride and padding is ceil(stride/2) — NOT (kernel-stride)/2,
        // which agrees for even strides and is wrong by one for stride 5.
        self.down = VaeConv1d(weight: try get(prefix + "block.4.weight").transposed(0, 2, 1),
                              bias: try get(prefix + "block.4.bias"),
                              stride: stride, padding: (stride + 1) / 2, dilation: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for u in units { h = u(h) }
        return down(act(h))
    }
}

/// Causal attention that also pools 2048 channels down to the 32-wide latent.
///
/// The pooling is `adaptive_avg_pool1d(mean_over_heads(attn), 32)`. With 8 heads
/// the head dim is 256, and 256/32 = 8, so it is a mean over consecutive groups
/// of 8. Dropping the pool and picking a head count that happens to make the
/// head dim 32 gives the right shape and the wrong values.
struct AudioCausalAttention {
    let qkv: MLXArray
    let qkvBias: MLXArray
    let proj: MLXArray
    let projBias: MLXArray
    let heads: Int
    let headDim: Int
    let outDim: Int

    init(inDim: Int, outDim: Int, heads: Int, prefix: String,
         weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.heads = heads
        self.headDim = inDim / heads
        self.outDim = outDim
        self.qkv = try get(prefix + "qkv.weight")
        // q_bias | zero_k_bias | v_bias — the k half is a registered buffer of
        // zeros, present so the concatenation has the right width.
        self.qkvBias = concatenated([try get(prefix + "q_bias"),
                                     try get(prefix + "zero_k_bias"),
                                     try get(prefix + "v_bias")], axis: 0)
        self.proj = try get(prefix + "proj.weight")
        self.projBias = try get(prefix + "proj.bias")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1)
        let f = matmul(x, qkv.T) + qkvBias
        let parts = f.reshaped([b, n, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
        let q = parts[0], k = parts[1], v = parts[2]

        let mask = MLXArray(0 ..< n).reshaped([n, 1]) .>= MLXArray(0 ..< n).reshaped([1, n])
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1.0 / Float(headDim).squareRoot(), mask: mask)

        // mean over heads, then adaptive average pool headDim -> outDim
        let pooled = o.mean(axis: 1)                                  // [B, N, headDim]
        precondition(headDim % outDim == 0,
                     "adaptive pool \(headDim) -> \(outDim) is not an integer ratio")
        let group = headDim / outDim
        let down = pooled.reshaped([b, n, outDim, group]).mean(axis: -1)
        return matmul(down, proj.T) + projBias
    }
}

/// `w2(gelu_tanh(w0(x)) * w1(x))` — note which branch is activated.
struct AudioGeGluMlp {
    let norm: VaeLayerNorm
    let w0: MLXArray, w0b: MLXArray
    let w1: MLXArray, w1b: MLXArray
    let w2: MLXArray, w2b: MLXArray

    init(prefix: String, weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.norm = VaeLayerNorm(weight: try get(prefix + "norm.weight"),
                                 bias: try get(prefix + "norm.bias"))
        self.w0 = try get(prefix + "w0.weight"); self.w0b = try get(prefix + "w0.bias")
        self.w1 = try get(prefix + "w1.weight"); self.w1b = try get(prefix + "w1.bias")
        self.w2 = try get(prefix + "w2.weight"); self.w2b = try get(prefix + "w2.bias")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm(x)
        // GELU is the tanh approximation in the reference, not the exact erf form.
        let a = geluApproximate(matmul(h, w0.T) + w0b)
        let b = matmul(h, w1.T) + w1b
        return matmul(a * b, w2.T) + w2b
    }
}

/// The encoder's posterior head: attention and a linear projection in parallel,
/// summed, then a GeGLU residual.
struct AudioAttnProjection {
    let norm1: VaeLayerNorm
    let norm2: VaeLayerNorm
    let norm3: VaeLayerNorm
    let attn: AudioCausalAttention
    let proj: MLXArray
    let projBias: MLXArray
    let mlp: AudioGeGluMlp

    init(inDim: Int, outDim: Int, heads: Int, prefix: String,
         weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.norm1 = VaeLayerNorm(weight: try get(prefix + "norm1.weight"),
                                  bias: try get(prefix + "norm1.bias"))
        self.norm2 = VaeLayerNorm(weight: try get(prefix + "norm2.weight"),
                                  bias: try get(prefix + "norm2.bias"))
        self.norm3 = VaeLayerNorm(weight: try get(prefix + "norm3.weight"),
                                  bias: try get(prefix + "norm3.bias"))
        self.attn = try AudioCausalAttention(inDim: inDim, outDim: outDim, heads: heads,
                                             prefix: prefix + "attn.", weights: weights)
        self.proj = try get(prefix + "proj.weight")
        self.projBias = try get(prefix + "proj.bias")
        self.mlp = try AudioGeGluMlp(prefix: prefix + "mlp.", weights: weights)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = matmul(norm3(x), proj.T) + projBias + attn(norm1(x))
        return h + mlp(norm2(h))
    }
}

package final class AudioVAEEncoder {
    /// 2 * 4 * 4 * 5 * 5 — audio samples per latent frame.
    package static let hopLength = 800
    package static let strides = [2, 4, 4, 5, 5]
    package static let latentChannels = 32

    let convIn: VaeConv1d
    let blocks: [AudioEncoderBlock]
    let actOut: Snake1d
    let convOut: VaeConv1d
    let preBlock: AudioAttnProjection
    let meanProj: MLXArray
    let meanProjBias: MLXArray
    let latentsMean: MLXArray
    let latentsStd: MLXArray

    package init(weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.latentsMean = try get("latents_mean")
        self.latentsStd = try get("latents_std")

        self.convIn = VaeConv1d(weight: try get("encoder.block.0.weight").transposed(0, 2, 1),
                                bias: try get("encoder.block.0.bias"),
                                stride: 1, padding: 3, dilation: 1)
        self.blocks = try Self.strides.enumerated().map { i, s in
            try AudioEncoderBlock(stride: s, prefix: "encoder.block.\(i + 1).", weights: weights)
        }
        self.actOut = Snake1d(alpha: try get("encoder.block.6.alpha"))
        self.convOut = VaeConv1d(weight: try get("encoder.block.7.weight").transposed(0, 2, 1),
                                 bias: try get("encoder.block.7.bias"),
                                 stride: 1, padding: 1, dilation: 1)

        // 8 heads, not 64 — the head dim is what the adaptive pool consumes.
        self.preBlock = try AudioAttnProjection(inDim: 2048, outDim: Self.latentChannels,
                                                heads: 8, prefix: "pre_block.",
                                                weights: weights)
        // Conv1d(32, 32, kernel 1) is a matmul in disguise.
        self.meanProj = try get("mean_proj.weight").reshaped([Self.latentChannels,
                                                              Self.latentChannels])
        self.meanProjBias = try get("mean_proj.bias")
        // `logs_proj` is deliberately not loaded: encode returns the mean.
    }

    /// Named after the reference's module paths. `encoder.block.N` is recorded
    /// in the reference's NCL layout, not this port's NLC, so a comparison has
    /// to transpose — which is exactly the kind of thing worth pinning.
    package struct Taps {
        /// `encoder.block.N` output, in NLC.
        package var encoderBlocks: [Int: MLXArray] = [:]
        package var preBlockAttn: MLXArray?
        package var preBlockMlp: MLXArray?
        package var preBlock: MLXArray?
        package var meanProj: MLXArray?
        package init() {}
    }

    /// Stereo waveform `[B, 2, L]` in [-1, 1] -> normalized latents `[B, 32, 2, T]`.
    package func encode(_ waveform: MLXArray) -> MLXArray {
        var t = Taps()
        return encode(waveform, taps: &t)
    }

    package func encode(_ waveform: MLXArray, taps: inout Taps) -> MLXArray {
        let b = waveform.dim(0), s = waveform.dim(1), l = waveform.dim(2)
        let padded = (l + Self.hopLength - 1) / Self.hopLength * Self.hopLength
        var w = waveform
        if padded > l {
            w = concatenated([w, MLXArray.zeros([b, s, padded - l], dtype: w.dtype)], axis: -1)
        }
        // stereo channels run through the mono encoder independently
        var x = w.reshaped([b * s, 1, padded]).transposed(0, 2, 1)   // [B*S, L, 1]

        x = convIn(x)
        taps.encoderBlocks[0] = x
        for (i, blk) in blocks.enumerated() {
            x = blk(x)
            taps.encoderBlocks[i + 1] = x
        }
        x = actOut(x)
        taps.encoderBlocks[6] = x
        x = convOut(x)                                               // [B*S, T, 2048]
        taps.encoderBlocks[7] = x

        let n1 = preBlock.norm1(x)
        taps.preBlockAttn = preBlock.attn(n1)
        let branch = matmul(preBlock.norm3(x), preBlock.proj.T) + preBlock.projBias
                   + taps.preBlockAttn!
        taps.preBlockMlp = preBlock.mlp(preBlock.norm2(branch))
        x = branch + taps.preBlockMlp!                               // [B*S, T, 32]
        taps.preBlock = x

        let z = matmul(x, meanProj.T) + meanProjBias
        taps.meanProj = z
        let zn = (z - latentsMean.reshaped([1, 1, Self.latentChannels]))
               / latentsStd.reshaped([1, 1, Self.latentChannels])

        let t = zn.dim(1)
        return zn.transposed(0, 2, 1)                                // [B*S, 32, T]
                 .reshaped([b, s, Self.latentChannels, t])
                 .transposed(0, 2, 1, 3)                             // [B, 32, 2, T]
    }
}
