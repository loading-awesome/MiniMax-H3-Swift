import Foundation
import MLX
import H3Foundation

/// Constructs the transformer from a validated checkpoint.
///
/// Every tensor is fetched by its exact reference name, so a rename or a missing
/// key fails here with that name rather than as a shape error fifty blocks
/// later. `H3Weights` has already validated structure from the header and
/// permutes the fused qkv when the vendor needs it, so this layer is pure
/// assembly — it decides nothing about layout.
extension H3Transformer {
    /// - Parameter computeDType: the dtype the block stack runs in. bf16 is the
    ///   model's native precision and what the reference uses (it takes the
    ///   dtype from the incoming context); fp32 exists to isolate whether a
    ///   discrepancy is precision or logic, and costs 2x residency.
    public init(weights: H3Weights, computeDType: DType = .bfloat16,
                fp32Attention: Bool = false,
                keepAdaLNFP32Resident: Bool = false) throws {
        let c = weights.config
        func w(_ n: String) throws -> MLXArray { try weights.tensor(n) }

        func attention(_ prefix: String) throws -> H3Attention {
            H3Attention(qkvWeight: try w(prefix + "attn.qkv_proj.weight"),
                        outWeight: try w(prefix + "attn.out_proj.weight"),
                        qNormWeight: try w(prefix + "attn.q_norm.weight"),
                        kNormWeight: try w(prefix + "attn.k_norm.weight"),
                        heads: c.numHeads, headDim: c.headDim, eps: c.qkNormEps,
                        fp32Attention: fp32Attention)
        }
        func mlp(_ prefix: String) throws -> H3MLP {
            H3MLP(fc1: try w(prefix + "mlp.fc1.weight"), fc2: try w(prefix + "mlp.fc2.weight"))
        }
        func norm(_ n: String, _ eps: Float) throws -> H3RMSNorm {
            H3RMSNorm(weight: try w(n), eps: eps)
        }

        var blocks: [DiTBlock] = []
        blocks.reserveCapacity(c.numLayers)
        for i in 0 ..< c.numLayers {
            let p = "blocks.\(i)."
            // 6 modulation tensors (shift/scale/gate for attn and mlp) across 3
            // modalities: [6 * hidden * 3, timeEmbedDim].
            let adaln = AdalnProj(weight: try w(p + "adaln_proj.linear.weight"),
                                  bias: try w(p + "adaln_proj.linear.bias"),
                                  expand: 6, modalities: 3, hidden: c.hiddenSize,
                                  keepFP32Resident: keepAdaLNFP32Resident)
            blocks.append(DiTBlock(norm1: try norm(p + "norm1.weight", c.normEps),
                                   norm2: try norm(p + "norm2.weight", c.normEps),
                                   attn: try attention(p), mlp: try mlp(p), adaln: adaln))
        }

        var refiner: [TokenRefiner.Block] = []
        for i in 0 ..< c.tokenRefinerLayers {
            let p = "token_refiner.blocks.\(i)."
            refiner.append(TokenRefiner.Block(norm1: try norm(p + "norm1.weight", c.normEps),
                                              norm2: try norm(p + "norm2.weight", c.normEps),
                                              attn: try attention(p), mlp: try mlp(p)))
        }

        // The final layer's AdaLN carries ONE modality, not three — its rows are
        // timestep rows alone.
        let finalAdaln = AdalnProj(weight: try w("final_layer.adaln_proj.linear.weight"),
                                   bias: try w("final_layer.adaln_proj.linear.bias"),
                                   expand: 2, modalities: 1, hidden: c.hiddenSize)
        let final = FinalLayer(norm: try norm("final_layer.norm.weight", c.finalNormEps),
                               adaln: finalAdaln,
                               videoOutWeight: try w("final_layer.video_out.weight"),
                               videoOutBias: try w("final_layer.video_out.bias"),
                               audioOutWeight: try w("final_layer.audio_out.weight"),
                               audioOutBias: try w("final_layer.audio_out.bias"))

        self.init(config: c,
                  conditionProj: (try w("condition_proj.weight"), try w("condition_proj.bias")),
                  videoPatchProj: (try w("video_patch_proj.weight"), try w("video_patch_proj.bias")),
                  audioPatchProj: (try w("audio_patch_proj.weight"), try w("audio_patch_proj.bias")),
                  tokenRefiner: TokenRefiner(blocks: refiner,
                                             finalNorm: try norm("token_refiner.final_norm.weight",
                                                                 c.finalNormEps)),
                  timeEmbedder: TimeEmbedder(projInWeight: try w("time_embedder.proj_in.weight"),
                                             projInBias: try w("time_embedder.proj_in.bias"),
                                             projOutWeight: try w("time_embedder.proj_out.weight"),
                                             projOutBias: try w("time_embedder.proj_out.bias"),
                                             inputDim: c.timestepInputDim),
                  blocks: blocks, finalLayer: final,
                  ropeInvFreq: try w("rope.inv_freq"),
                  computeDType: computeDType)
    }
}
