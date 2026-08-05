import Foundation
import MLX
import MLXNN
import MLXFast
import H3Foundation

/// The Qwen3-VL vision tower, as shipped inside the H3 conditioning checkpoint.
///
/// 351 of the file's 902 tensors, under `visual.*`: a patch embedding, a 48x48
/// learned position grid, 27 blocks at 1152-dim, a merger that projects into the
/// language model's 5120-dim space, and — the Qwen3-VL-specific part — three
/// **deepstack** mergers that tap layers 8, 16 and 24 and inject those features
/// back into the language stack at the image's token positions.
///
/// It is reachable only through **image prompts**. It has nothing to do with
/// I2V keyframes, which go through the video VAE; confusing the two is easy
/// because both take an image and both feed the DiT.
package struct VisionTowerConfig: Sendable, Equatable {
    package var hiddenSize = 1152
    package var intermediateSize = 4304
    package var depth = 27
    package var numHeads = 16
    package var patchSize = 16
    package var temporalPatchSize = 2
    package var inChannels = 3
    package var spatialMergeSize = 2
    /// 2304 = 48x48. The grid is bilinearly resampled to each image's shape.
    package var numPositionEmbeddings = 2304
    /// H3's language hidden size — what the merger projects into.
    package var outHiddenSize = 5120
    /// Layers whose output is tapped for deepstack injection.
    package var deepstackIndexes = [8, 16, 24]
    package init() {}

    package var headDim: Int { hiddenSize / numHeads }
    /// RoPE is applied over half the head dim, split again between row and col.
    package var rotaryDim: Int { headDim / 2 }
    package var gridPerSide: Int { Int(Double(numPositionEmbeddings).squareRoot()) }
    package var mergeUnit: Int { spatialMergeSize * spatialMergeSize }
    package var mergeDim: Int { hiddenSize * mergeUnit }
}

/// `[T, H, W]` patch counts for one image. H3 only ever sends `t = 1`.
package struct VisionGrid: Sendable, Equatable {
    package let t: Int, h: Int, w: Int
    package init(t: Int = 1, h: Int, w: Int) { self.t = t; self.h = h; self.w = w }
    package var tokens: Int { t * h * w }
}

package final class VisionTower {
    package let config: VisionTowerConfig

    struct Block {
        let norm1: VaeLayerNorm, norm2: VaeLayerNorm
        let qkv: MLXArray, qkvBias: MLXArray
        let proj: MLXArray, projBias: MLXArray
        let fc1: MLXArray, fc1Bias: MLXArray
        let fc2: MLXArray, fc2Bias: MLXArray
    }

    /// The two mergers differ in **where the norm sits**, and the checkpoint
    /// says so: `merger.norm.weight` is `[1152]`, the deepstack ones `[4608]`.
    /// The main merger normalizes each patch *before* the spatial merge;
    /// deepstack normalizes the merged 4x vector *after*. Same weight names,
    /// different semantics.
    struct Merger {
        let norm: VaeLayerNorm
        let fc1: MLXArray, fc1Bias: MLXArray
        let fc2: MLXArray, fc2Bias: MLXArray
        let postShuffleNorm: Bool
    }

    let patchProj: MLXArray, patchProjBias: MLXArray
    let posEmbed: MLXArray
    let blocks: [Block]
    let merger: Merger
    let deepstackMergers: [Merger]

    package enum Error: Swift.Error, CustomStringConvertible {
        case missing(String)
        package var description: String {
            switch self { case .missing(let n): "vision tower has no tensor named \(n)" }
        }
    }

    package init(weights: [String: MLXArray], config: VisionTowerConfig = VisionTowerConfig(),
                prefix: String? = nil) throws {
        self.config = config
        let resolvedPrefix: String
        if let prefix {
            resolvedPrefix = prefix
        } else {
            if weights.keys.contains(where: { $0.hasPrefix("model.visual.") }) {
                resolvedPrefix = "model.visual."
            } else {
                resolvedPrefix = "visual."
            }
        }
        func w(_ n: String) throws -> MLXArray {
            guard let a = weights[resolvedPrefix + n] else { throw Error.missing(resolvedPrefix + n) }
            return a
        }

        // The patch embedding is a Conv3d whose stride equals its kernel, applied
        // to inputs that are already one patch each — so it is a matmul wearing
        // a convolution's shape. [1152, 3, 2, 16, 16] flattens to [1152, 1536],
        // which is exactly the width of a `flatten_patches` row.
        let pw = try w("patch_embed.proj.weight")
        self.patchProj = pw.reshaped([config.hiddenSize, -1])
        self.patchProjBias = try w("patch_embed.proj.bias")
        self.posEmbed = try w("pos_embed.weight")

        self.blocks = try (0 ..< config.depth).map { i in
            let b = "blocks.\(i)."
            return Block(
                norm1: VaeLayerNorm(weight: try w(b + "norm1.weight"),
                                    bias: try w(b + "norm1.bias"), eps: 1e-6),
                norm2: VaeLayerNorm(weight: try w(b + "norm2.weight"),
                                    bias: try w(b + "norm2.bias"), eps: 1e-6),
                qkv: try w(b + "attn.qkv.weight"), qkvBias: try w(b + "attn.qkv.bias"),
                proj: try w(b + "attn.proj.weight"), projBias: try w(b + "attn.proj.bias"),
                fc1: try w(b + "mlp.linear_fc1.weight"), fc1Bias: try w(b + "mlp.linear_fc1.bias"),
                fc2: try w(b + "mlp.linear_fc2.weight"), fc2Bias: try w(b + "mlp.linear_fc2.bias"))
        }

        func merger(_ p: String, postShuffle: Bool) throws -> Merger {
            Merger(norm: VaeLayerNorm(weight: try w(p + "norm.weight"),
                                      bias: try w(p + "norm.bias"), eps: 1e-6),
                   fc1: try w(p + "linear_fc1.weight"), fc1Bias: try w(p + "linear_fc1.bias"),
                   fc2: try w(p + "linear_fc2.weight"), fc2Bias: try w(p + "linear_fc2.bias"),
                   postShuffleNorm: postShuffle)
        }
        self.merger = try merger("merger.", postShuffle: false)
        self.deepstackMergers = try (0 ..< config.deepstackIndexes.count).map {
            try merger("deepstack_merger_list.\($0).", postShuffle: true)
        }
    }

    // MARK: - position encodings

    /// Bilinear resample of the learned 48x48 grid onto this image's patch grid,
    /// then reordered into merge-block order.
    ///
    /// The reorder is the subtle half. Tokens are not in raster order: they are
    /// grouped so that each consecutive run of `mergeUnit` tokens is one 2x2
    /// block, because that is what the merger reshapes over. Getting the
    /// interpolation right and the permutation wrong leaves every value present
    /// and every one in the wrong row.
    func positionEmbeddings(_ grid: VisionGrid) -> MLXArray {
        let side = config.gridPerSide
        let m = config.spatialMergeSize

        func axis(_ n: Int) -> (floor: [Int32], ceil: [Int32], frac: [Float]) {
            // linspace(0, side-1, n) — endpoint inclusive, and n == 1 pins to 0.
            let step = n > 1 ? Double(side - 1) / Double(n - 1) : 0
            var f = [Int32](), c = [Int32](), d = [Float]()
            for i in 0 ..< n {
                let v = Double(i) * step
                // `.int()` in the reference truncates toward zero; v >= 0 here.
                let lo = Int32(v)
                f.append(lo)
                c.append(min(lo + 1, Int32(side - 1)))
                d.append(Float(v - Double(lo)))
            }
            return (f, c, d)
        }
        let (hf, hc, dh) = axis(grid.h)
        let (wf, wc, dw) = axis(grid.w)

        // Four corners of the bilinear tap, weighted and summed.
        let hFloor = MLXArray(hf).reshaped([grid.h, 1])
        let hCeil = MLXArray(hc).reshaped([grid.h, 1])
        let wFloor = MLXArray(wf).reshaped([1, grid.w])
        let wCeil = MLXArray(wc).reshaped([1, grid.w])
        let dhA = MLXArray(dh).reshaped([grid.h, 1])
        let dwA = MLXArray(dw).reshaped([1, grid.w])

        let corners = [
            (hFloor * Int32(side) + wFloor, (1.0 - dhA) * (1.0 - dwA)),
            (hFloor * Int32(side) + wCeil, (1.0 - dhA) * dwA),
            (hCeil * Int32(side) + wFloor, dhA * (1.0 - dwA)),
            (hCeil * Int32(side) + wCeil, dhA * dwA),
        ]
        var acc: MLXArray?
        for (idx, weight) in corners {
            let e = posEmbed[idx.flattened()] * weight.flattened().reshaped([-1, 1])
            acc = acc == nil ? e : acc! + e
        }
        var pos = acc!                                          // [h*w, hidden] raster

        // raster -> merge-block order
        pos = pos.reshaped([grid.h / m, m, grid.w / m, m, config.hiddenSize])
                 .transposed(0, 2, 1, 3, 4)
                 .reshaped([grid.h * grid.w, config.hiddenSize])
        if grid.t > 1 {
            pos = tiled(pos, repetitions: [grid.t, 1])
        }
        return pos
    }

    /// 2-D RoPE frequencies: `[tokens, rotaryDim]`, row half then column half.
    func ropeFrequencies(_ grid: VisionGrid) -> MLXArray {
        let m = config.spatialMergeSize
        let half = config.rotaryDim / 2                          // 18 for 1152/16
        let theta: Float = 10_000

        let exponent = MLXArray(stride(from: 0, to: config.rotaryDim, by: 2).map { Float($0) })
            / Float(config.rotaryDim)
        let invFreq = 1.0 / pow(MLXArray(theta), exponent)       // [half]
        let maxHW = max(grid.h, grid.w)
        let table = MLXArray(0 ..< maxHW).asType(.float32).reshaped([maxHW, 1])
            * invFreq.reshaped([1, half])                        // [maxHW, half]

        // Row/col index per token, in the same merge-block order as the position
        // embeddings — built by the same reshape rather than a second formula,
        // so the two cannot drift apart.
        let rows = broadcast(MLXArray(0 ..< grid.h).reshaped([grid.h, 1]), to: [grid.h, grid.w])
        let cols = broadcast(MLXArray(0 ..< grid.w).reshaped([1, grid.w]), to: [grid.h, grid.w])
        func blockOrder(_ a: MLXArray) -> MLXArray {
            a.reshaped([grid.h / m, m, grid.w / m, m])
             .transposed(0, 2, 1, 3)
             .reshaped([grid.h * grid.w])
        }
        var r = blockOrder(rows), c = blockOrder(cols)
        if grid.t > 1 {
            r = tiled(r, repetitions: [grid.t])
            c = tiled(c, repetitions: [grid.t])
        }
        return concatenated([table[r], table[c]], axis: -1)      // [tokens, rotaryDim]
    }

    /// Split-half rotation over the full head dim.
    ///
    /// The reference builds `emb = cat(rot, rot)` and then splits cos/sin back
    /// in half, so both halves see the same angle — which is plain
    /// rotate-half RoPE written the long way round.
    static func applyRoPE(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let lo = x[.ellipsis, 0 ..< half]
        let hi = x[.ellipsis, half ..< (2 * half)]
        return concatenated([lo * c - hi * s, hi * c + lo * s], axis: -1)
    }

    // MARK: - forward

    package struct Taps {
        package var patchEmbed: MLXArray?
        package var blocks: [Int: MLXArray] = [:]
        package var mergerNorm: MLXArray?
        package init() {}
    }

    package struct Output {
        /// `[mergedTokens, outHiddenSize]` — what splices into the prompt.
        package let merged: MLXArray
        /// One per deepstack index, same shape as `merged`.
        package let deepstack: [MLXArray]
    }

    /// - Parameter patches: `[tokens, inChannels * temporalPatch * patch * patch]`,
    ///   the `flatten_patches` layout — already resized, normalized and
    ///   permuted. See ``VisionPreprocess``.
    package func callAsFunction(patches: MLXArray, grid: VisionGrid,
                               taps: inout Taps) -> Output {
        precondition(patches.dim(0) == grid.tokens,
                     "grid \(grid) wants \(grid.tokens) patches, got \(patches.dim(0))")
        let s = grid.tokens
        var x = matmul(patches.asType(.float32), patchProj.T) + patchProjBias
        taps.patchEmbed = x
        x = x + positionEmbeddings(grid)

        let freqs = ropeFrequencies(grid)                        // [S, rotaryDim]
        let c = cos(freqs).expandedDimensions(axis: 1)           // [S, 1, rotaryDim]
        let sn = sin(freqs).expandedDimensions(axis: 1)

        // Attention runs per image. H3 sends one image at a time, so there is a
        // single segment and no mask is needed — the reference splits on
        // `cu_seqlens` for exactly this reason and would need a block-diagonal
        // mask if it did not.
        let scale = 1.0 / Float(config.headDim).squareRoot()
        for (i, b) in blocks.enumerated() {
            let h = b.norm1(x)
            let qkv = matmul(h, b.qkv.T) + b.qkvBias
            let parts = qkv.reshaped([s, 3, config.numHeads, config.headDim])
                           .transposed(1, 0, 2, 3)
            let q = Self.applyRoPE(parts[0], cos: c, sin: sn)
            let k = Self.applyRoPE(parts[1], cos: c, sin: sn)
            let v = parts[2]

            let o = MLXFast.scaledDotProductAttention(
                queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
                keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
                values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
                scale: scale, mask: nil)
            let attn = o.squeezed(axis: 0).transposed(1, 0, 2).reshaped([s, config.hiddenSize])
            x = x + matmul(attn, b.proj.T) + b.projBias

            let y = b.norm2(x)
            // GELU is the tanh approximation throughout this tower.
            let up = geluApproximate(matmul(y, b.fc1.T) + b.fc1Bias)
            x = x + matmul(up, b.fc2.T) + b.fc2Bias
            taps.blocks[i] = x
            if i % 10 == 0 { eval(x) }
        }

        var deepstack: [MLXArray] = []
        for (slot, layer) in config.deepstackIndexes.enumerated() {
            guard let feat = taps.blocks[layer] else { continue }
            deepstack.append(apply(deepstackMergers[slot], feat))
        }
        taps.mergerNorm = merger.norm(x)
        return Output(merged: apply(merger, x), deepstack: deepstack)
    }

    package func callAsFunction(patches: MLXArray, grid: VisionGrid) -> Output {
        var t = Taps()
        return callAsFunction(patches: patches, grid: grid, taps: &t)
    }

    /// `fc2(gelu(fc1(norm(x))))` with the merge reshape either side of the norm.
    private func apply(_ m: Merger, _ x: MLXArray) -> MLXArray {
        let merged = m.postShuffleNorm
            ? m.norm(x.reshaped([-1, config.mergeDim]))          // norm after merge
            : m.norm(x).reshaped([-1, config.mergeDim])          // norm before merge
        let h = geluApproximate(matmul(merged, m.fc1.T) + m.fc1Bias)
        return matmul(h, m.fc2.T) + m.fc2Bias
    }
}
