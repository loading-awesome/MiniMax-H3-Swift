import Foundation
import MLX
import MLXNN
import MLXFast
import H3Foundation

struct VaeRMSNorm {
    let weight: MLXArray?
    let eps: Float
    
    init(weight: MLXArray? = nil, eps: Float = 1e-5) {
        self.weight = weight
        self.eps = eps
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let n = f * rsqrt(mean(f * f, axis: -1, keepDims: true) + eps)
        if let weight {
            return (n * weight.asType(.float32)).asType(x.dtype)
        }
        return n.asType(x.dtype)
    }
}

struct VaeLayerNorm {
    let weight: MLXArray
    let bias: MLXArray
    let eps: Float
    
    init(weight: MLXArray, bias: MLXArray, eps: Float = 1e-5) {
        self.weight = weight
        self.bias = bias
        self.eps = eps
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let mu = mean(f, axis: -1, keepDims: true)
        let diff = f - mu
        let variance = mean(diff * diff, axis: -1, keepDims: true)
        let n = diff * rsqrt(variance + eps)
        return (n * weight.asType(.float32) + bias.asType(.float32)).asType(x.dtype)
    }
}

struct RotaryEmbeddingND {
    let invFreq: MLXArray
    let angleScale: Float = 2.0 * Float.pi
    
    init(dim: Int, theta: Float = 100.0) {
        let step = 6.0 / Float(dim)
        var sValues: [Float] = []
        var curr: Float = 0.0
        while curr < 1.0 - 1e-6 {
            sValues.append(curr)
            curr += step
        }
        self.invFreq = 1.0 / pow(MLXArray(theta), MLXArray(sValues))
    }
    
    func callAsFunction(_ imgIds: MLXArray) -> MLXArray {
        let B = imgIds.dim(0)
        let S = imgIds.dim(1)
        let imgIdsExpanded = imgIds.expandedDimensions(axis: -1)
        let invFreqReshaped = invFreq.reshaped([1, 1, 1, invFreq.dim(0)])
        let angles = imgIdsExpanded.asType(.float32) * angleScale * invFreqReshaped
        let anglesFlat = angles.reshaped([B, S, angles.dim(2) * angles.dim(3)])
        
        let c = cos(anglesFlat)
        let s = sin(anglesFlat)
        
        let table = stacked([c, -s, s, c], axis: -1)
        return table.reshaped([B, S, 1, anglesFlat.dim(2), 2, 2])
    }
}

struct VaeAttention {
    let normQ: VaeRMSNorm
    let normK: VaeRMSNorm
    let toQkvWeight: MLXArray
    let toQkvBias: MLXArray
    let toOutWeight: MLXArray
    let toOutBias: MLXArray
    let heads: Int
    let dimHead: Int
    
    func callAsFunction(_ x: MLXArray, rotaryPosEmb: MLXArray?) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        
        let qkv = matmul(x, toQkvWeight.T) + toQkvBias
        let reshaped = qkv.reshaped([B, S, heads, 3 * dimHead])
        
        let query = reshaped[0..., 0..., 0..., 0 ..< dimHead]
        let key = reshaped[0..., 0..., 0..., dimHead ..< (2 * dimHead)]
        let value = reshaped[0..., 0..., 0..., (2 * dimHead)...]
        
        var q = normQ(query)
        var k = normK(key)
        
        if let table = rotaryPosEmb {
            let half = table.dim(-3)
            let rot = half * 2
            
            let c = table[0..., 0..., 0, 0..., 0, 0].expandedDimensions(axis: 2)
            let negS = table[0..., 0..., 0, 0..., 0, 1].expandedDimensions(axis: 2)
            let s = table[0..., 0..., 0, 0..., 1, 0].expandedDimensions(axis: 2)
            let c2 = table[0..., 0..., 0, 0..., 1, 1].expandedDimensions(axis: 2)
            
            let qA = q[0..., 0..., 0..., 0 ..< half]
            let qB = q[0..., 0..., 0..., half ..< rot]
            let qRa = c * qA + negS * qB
            let qRb = s * qA + c2 * qB
            let qRotated = concatenated([qRa, qRb], axis: -1)
            q = concatenated([qRotated, q[0..., 0..., 0..., rot...]], axis: -1)
            
            let kA = k[0..., 0..., 0..., 0 ..< half]
            let kB = k[0..., 0..., 0..., half ..< rot]
            let kRa = c * kA + negS * kB
            let kRb = s * kA + c2 * kB
            let kRotated = concatenated([kRa, kRb], axis: -1)
            k = concatenated([kRotated, k[0..., 0..., 0..., rot...]], axis: -1)
        }
        
        let qh = q.transposed(0, 2, 1, 3)
        let kh = k.transposed(0, 2, 1, 3)
        let vh = value.transposed(0, 2, 1, 3)
        
        let scale = 1.0 / Float(dimHead).squareRoot()
        let out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh,
            scale: scale, mask: nil
        )
        
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, S, heads * dimHead])
        return matmul(merged, toOutWeight.T) + toOutBias
    }
}

struct VaeFeedForward {
    let w1Weight: MLXArray
    let w1Bias: MLXArray
    let w2Weight: MLXArray
    let w2Bias: MLXArray
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = matmul(x, w1Weight.T) + w1Bias
        let innerDim = h.dim(-1) / 2
        let gate = h[0..., 0..., 0 ..< innerDim]
        let up = h[0..., 0..., innerDim...]
        return matmul(silu(gate) * up, w2Weight.T) + w2Bias
    }
}

struct VaeTransformerBlock {
    let norm1: VaeRMSNorm
    let norm2: VaeRMSNorm
    let attn: VaeAttention
    let ff: VaeFeedForward
    let scale1: MLXArray
    let scale2: MLXArray
    
    func callAsFunction(_ x: MLXArray, rotaryPosEmb: MLXArray?) -> MLXArray {
        let h1 = norm1(x)
        let x1 = x + attn(h1, rotaryPosEmb: rotaryPosEmb) * scale1
        let h2 = norm2(x1)
        return x1 + ff(h2) * scale2
    }
}

struct ViT3DDecoder {
    let patchSize: Int = 16
    let patchSizeT: Int = 4
    let outChannels: Int = 3
    let numRegisterTokens: Int = 4
    
    let posEmbed: RotaryEmbeddingND
    let xEmbedderWeight: MLXArray
    let xEmbedderBias: MLXArray
    let registerTokens: MLXArray
    let normOut: VaeLayerNorm
    let projOutWeight: MLXArray
    let projOutBias: MLXArray
    let blocks: [VaeTransformerBlock]
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let C = x.dim(1)
        let latentT = x.dim(2)
        let latentH = x.dim(3)
        let latentW = x.dim(4)
        
        let flattened = x.reshaped([B, C, -1])
        let transposed = flattened.transposed(0, 2, 1)
        
        var h = matmul(transposed, xEmbedderWeight.T) + xEmbedderBias
        
        let numPatches = h.dim(1)
        let numSuffix = 1 + numRegisterTokens
        
        let regExpanded = broadcast(registerTokens, to: [B, numRegisterTokens, registerTokens.dim(-1)])
        let zeroSuffix = MLXArray.zeros([B, 1, h.dim(-1)], dtype: h.dtype)
        
        h = concatenated([h, regExpanded, zeroSuffix], axis: 1)
        
        let imgIds = createTokenIds(latentT: latentT, latentH: latentH, latentW: latentW, dtype: h.dtype)
        let imgIdsExpanded = broadcast(imgIds, to: [B, imgIds.dim(1), imgIds.dim(2)])
        let suffixIds = MLXArray.zeros([B, numSuffix, 3], dtype: h.dtype)
        let imgIdsFull = concatenated([imgIdsExpanded, suffixIds], axis: 1)
        
        let rotaryPosEmb = posEmbed(imgIdsFull)
        
        for block in blocks {
            h = block(h, rotaryPosEmb: rotaryPosEmb)
        }
        
        var output = matmul(normOut(h), projOutWeight.T) + projOutBias
        
        output = output[0..., 0 ..< numPatches, 0...]
        
        output = output.reshaped([
            B, latentT, latentH, latentW,
            outChannels, patchSizeT, patchSize, patchSize
        ])
        
        output = output.transposed(0, 4, 1, 5, 2, 6, 3, 7)
        
        output = output.reshaped([
            B, outChannels,
            latentT * patchSizeT,
            latentH * patchSize,
            latentW * patchSize
        ])
        
        return output
    }
    
    private func createTokenIds(latentT: Int, latentH: Int, latentW: Int, dtype: DType) -> MLXArray {
        func makeCoords(dimSize: Int) -> MLXArray {
            let coords = (MLXArray(0 ..< dimSize).asType(dtype) + 0.5) / Float(dimSize)
            return 2.0 * coords - 1.0
        }
        let coordsT = makeCoords(dimSize: latentT)
        let coordsH = makeCoords(dimSize: latentH)
        let coordsW = makeCoords(dimSize: latentW)
        
        let gridT = broadcast(coordsT.reshaped([latentT, 1, 1]), to: [latentT, latentH, latentW])
        let gridH = broadcast(coordsH.reshaped([1, latentH, 1]), to: [latentT, latentH, latentW])
        let gridW = broadcast(coordsW.reshaped([1, 1, latentW]), to: [latentT, latentH, latentW])
        
        let coords = stacked([gridT, gridH, gridW], axis: -1)
        return coords.reshaped([1, latentT * latentH * latentW, 3])
    }
}

public final class VideoVAE {
    public let url: URL
    private let postQuantConvWeight: MLXArray
    private let postQuantConvBias: MLXArray
    private let decoder: ViT3DDecoder
    
    public let latentsMean: MLXArray
    public let latentsStd: MLXArray
    
    public let pixelMean = MLXArray(IMAGENET_MEAN).reshaped([1, 3, 1, 1, 1])
    public let pixelStd = MLXArray(IMAGENET_STD).reshaped([1, 3, 1, 1, 1])
    
    static let IMAGENET_MEAN: [Float] = [0.485, 0.456, 0.406]
    static let IMAGENET_STD: [Float] = [0.229, 0.224, 0.225]
    
    public init(url: URL) throws {
        self.url = url
        let w = try MLX.loadArrays(url: url)
        
        func get(_ name: String) throws -> MLXArray {
            guard let a = w[name] else {
                throw H3Weights.Error.missing(name)
            }
            return a
        }
        
        self.postQuantConvWeight = try get("post_quant_conv.weight")
        self.postQuantConvBias = try get("post_quant_conv.bias")
        self.latentsMean = try get("latents_mean")
        self.latentsStd = try get("latents_std")
        
        let posEmbed = RotaryEmbeddingND(dim: 48, theta: 100.0)
        let xEmbedderWeight = try get("decoder.x_embedder.weight")
        let xEmbedderBias = try get("decoder.x_embedder.bias")
        let registerTokens = try get("decoder.register_tokens")
        let normOutWeight = try get("decoder.norm_out.weight")
        let normOutBias = try get("decoder.norm_out.bias")
        let normOut = VaeLayerNorm(weight: normOutWeight, bias: normOutBias, eps: 1e-5)
        let projOutWeight = try get("decoder.proj_out.weight")
        let projOutBias = try get("decoder.proj_out.bias")
        
        var blocks: [VaeTransformerBlock] = []
        for i in 0 ..< 36 {
            let p = "decoder.transformer_blocks.\(i)."
            let norm1 = VaeRMSNorm(weight: try get(p + "norm1.weight"), eps: 1e-5)
            let norm2 = VaeRMSNorm(weight: try get(p + "norm2.weight"), eps: 1e-5)
            let scale1 = try get(p + "scale1")
            let scale2 = try get(p + "scale2")
            
            let toQkvWeight = try get(p + "attn.to_qkv.weight")
            let toQkvBias = try get(p + "attn.to_qkv.bias")
            let toOutWeight = try get(p + "attn.to_out.weight")
            let toOutBias = try get(p + "attn.to_out.bias")
            
            let attn = VaeAttention(
                normQ: VaeRMSNorm(eps: 1e-5),
                normK: VaeRMSNorm(eps: 1e-5),
                toQkvWeight: toQkvWeight, toQkvBias: toQkvBias,
                toOutWeight: toOutWeight, toOutBias: toOutBias,
                heads: 32, dimHead: 64
            )
            
            let ff = VaeFeedForward(
                w1Weight: try get(p + "ff.w1.weight"),
                w1Bias: try get(p + "ff.w1.bias"),
                w2Weight: try get(p + "ff.w2.weight"),
                w2Bias: try get(p + "ff.w2.bias")
            )
            
            blocks.append(VaeTransformerBlock(
                norm1: norm1, norm2: norm2,
                attn: attn, ff: ff,
                scale1: scale1, scale2: scale2
            ))
        }
        
        self.decoder = ViT3DDecoder(
            posEmbed: posEmbed,
            xEmbedderWeight: xEmbedderWeight,
            xEmbedderBias: xEmbedderBias,
            registerTokens: registerTokens,
            normOut: normOut,
            projOutWeight: projOutWeight,
            projOutBias: projOutBias,
            blocks: blocks
        )
    }
    
    public func postQuantConv(_ z: MLXArray) -> MLXArray {
        let zT = z.transposed(0, 2, 3, 4, 1)
        var out = matmul(zT, postQuantConvWeight.reshaped([24, 24]).T)
        out = out + postQuantConvBias
        return out.transposed(0, 4, 1, 2, 3)
    }
    
    public func decodePixels(_ z: MLXArray) -> MLXArray {
        let pq = postQuantConv(z)
        return decoder(pq)
    }
    
    public func splitTiles(inputLen: Int, tileSize: Int = 256, tileOverlapMin: Int = 64, vaeRatio: Int = 16) -> (starts: [Int], lengths: [Int], overlaps: [Int]) {
        if tileSize >= inputLen {
            return ([0], [inputLen], [])
        }
        var N = Int(ceil(Double(inputLen) / Double(tileSize)))
        var overlaps: [Int] = []
        while true {
            overlaps = Array(repeating: tileOverlapMin, count: N - 1)
            let sumOverlaps = overlaps.reduce(0, +)
            let remaining = tileSize * N - sumOverlaps - inputLen
            if remaining < 0 {
                N += 1
            } else {
                break
            }
        }
        let remaining = tileSize * N - overlaps.reduce(0, +) - inputLen
        let remainingUnits = remaining / vaeRatio
        for i in 0 ..< remainingUnits {
            overlaps[i % (N - 1)] += vaeRatio
        }
        var tileStartIdx = [0]
        for i in 0 ..< (N - 1) {
            tileStartIdx.append(tileStartIdx.last! + tileSize - overlaps[i])
        }
        return (tileStartIdx, Array(repeating: tileSize, count: N), overlaps)
    }
    
    public func blend(_ a: MLXArray, _ b: MLXArray, blendExtent: Int, dim: Int) -> MLXArray {
        let ndim = a.ndim
        // Callers pass -1 and -2. MLXArray.dim() and sliceDim() both accept
        // negative axes; a Swift Array subscript does not, and indexing
        // weightShape[-1] traps. Normalise once, here.
        let axis = dim < 0 ? ndim + dim : dim
        precondition(axis >= 0 && axis < ndim, "blend axis \(dim) outside 0..<\(ndim)")
        let actualExtent = min(a.dim(axis), b.dim(axis), blendExtent)
        if actualExtent <= 0 {
            return b
        }
        let positions = MLXArray(0 ..< actualExtent).asType(b.dtype)
        let weightA = 1.0 - (positions / Float(actualExtent))
        let weightB = positions / Float(actualExtent)

        var weightShape = Array(repeating: 1, count: ndim)
        weightShape[axis] = actualExtent
        let wA = weightA.reshaped(weightShape)
        let wB = weightB.reshaped(weightShape)
        
        let sliceA = sliceDim(a, dim: axis, start: a.dim(axis) - actualExtent, end: a.dim(axis))
        let sliceB = sliceDim(b, dim: axis, start: 0, end: actualExtent)

        let blended = sliceA * wA + sliceB * wB

        if actualExtent < b.dim(axis) {
            let sliceBRest = sliceDim(b, dim: axis, start: actualExtent, end: b.dim(axis))
            return concatenated([blended, sliceBRest], axis: axis)
        }
        return blended
    }
    
    private func sliceDim(_ array: MLXArray, dim: Int, start: Int, end: Int) -> MLXArray {
        let size = array.dim(dim)
        let s = max(0, min(size, start))
        let e = max(0, min(size, end))
        let actualDim = dim < 0 ? array.ndim + dim : dim
        
        var indices: [any MLXArrayIndex] = []
        for i in 0 ..< array.ndim {
            if i == actualDim {
                indices.append(s ..< e)
            } else {
                indices.append(0 ..< array.dim(i))
            }
        }
        return array[indices]
    }
    
    public func tiledDecode(_ z: MLXArray) -> MLXArray {
        let vaeRatio = 16
        let height = z.dim(-2) * vaeRatio
        let width = z.dim(-1) * vaeRatio
        
        let (yIdx, yLen, yOverlap) = splitTiles(inputLen: height)
        let (xIdx, xLen, xOverlap) = splitTiles(inputLen: width)
        
        var rowTensors: [MLXArray] = []
        var rowTails: [MLXArray] = []
        
        for (i, (iPos, iLen)) in zip(yIdx, yLen).enumerated() {
            let zi = iPos / vaeRatio
            let zl = iLen / vaeRatio
            var newTails: [MLXArray] = []
            var leftTail: MLXArray? = nil
            var rowTiles: [MLXArray] = []
            
            for (j, (jPos, jLen)) in zip(xIdx, xLen).enumerated() {
                let zj = jPos / vaeRatio
                let zw = jLen / vaeRatio
                
                let zSlice = z[0..., 0..., 0..., zi ..< (zi + zl), zj ..< (zj + zw)]
                var tile = decodePixels(zSlice)
                
                if i < yIdx.count - 1 {
                    let overlapY = yOverlap[i]
                    let start = tile.dim(-2) - overlapY
                    let tail = sliceDim(tile, dim: -2, start: start, end: tile.dim(-2))
                    newTails.append(tail)
                }
                var nextLeftTail: MLXArray? = nil
                if j < xIdx.count - 1 {
                    let overlapX = xOverlap[j]
                    let start = tile.dim(-1) - overlapX
                    nextLeftTail = sliceDim(tile, dim: -1, start: start, end: tile.dim(-1))
                }
                
                if i > 0 {
                    tile = blend(rowTails[j], tile, blendExtent: yOverlap[i - 1], dim: -2)
                }
                if j > 0, let left = leftTail {
                    tile = blend(left, tile, blendExtent: xOverlap[j - 1], dim: -1)
                }
                
                leftTail = nextLeftTail
                
                if i < yIdx.count - 1 {
                    tile = sliceDim(tile, dim: -2, start: 0, end: tile.dim(-2) - yOverlap[i])
                }
                if j < xIdx.count - 1 {
                    tile = sliceDim(tile, dim: -1, start: 0, end: tile.dim(-1) - xOverlap[j])
                }
                
                rowTiles.append(tile)
            }
            
            rowTails = newTails
            let rowTensor = concatenated(rowTiles, axis: -1)
            rowTensors.append(rowTensor)
        }
        
        return concatenated(rowTensors, axis: -2)
    }
    
    public func decodeTemporalPadFrames(zLen: Int, padTokens: Int) -> Int {
        if padTokens <= 0 { return 0 }
        let clipLength = 17
        let vaeRatioT = 4
        let tokensChunkSize = 5
        let intraTail = clipLength % vaeRatioT
        if intraTail == 0 {
            return padTokens * vaeRatioT
        }
        let zLenBeforePad = zLen - padTokens
        var sum = 0
        for k in 0 ..< padTokens {
            if (zLenBeforePad + k) % tokensChunkSize == 0 {
                sum += intraTail
            } else {
                sum += vaeRatioT
            }
        }
        return sum
    }
    
    public func decodeTemporalFramePlan(zLen: Int, numChunks: Int, padTokens: Int) -> Int {
        let tokensChunkSize = 5
        let vaeRatioT = 4
        let tokenOverlap = 2
        let framePrePadding = 3
        let chunkDec = tokensChunkSize * vaeRatioT
        let tokenDrop = 3
        let splitCount = (tokenDrop > 0 ? 1 : 0) + 1
        
        var totalFrames = 0
        var finalOverlapFrames = 0
        
        for i in 0 ..< numChunks {
            let tStartIdx = i * tokensChunkSize
            let tEndIdx = tStartIdx + tokensChunkSize + tokenOverlap
            let clipTokenLen = max(0, min(tEndIdx, zLen) - min(tStartIdx, zLen))
            let clipFrameLen = clipTokenLen * vaeRatioT
            
            for j in 0 ..< splitCount {
                let fStartIdx = j * chunkDec
                let fEndIdx = min(fStartIdx + chunkDec, clipFrameLen)
                let chunkFrames = max(0, fEndIdx - fStartIdx - framePrePadding)
                if j == 0 {
                    totalFrames += chunkFrames
                } else {
                    finalOverlapFrames = chunkFrames
                }
            }
        }
        
        totalFrames += finalOverlapFrames
        return totalFrames - decodeTemporalPadFrames(zLen: zLen, padTokens: padTokens)
    }
    
    public func decodeTemporal(_ z: MLXArray) -> MLXArray {
        let tokensChunkSize = 5
        let tokenOverlap = 2
        let frameOverlap = 5
        let framePrePadding = 3
        let chunkDec = tokensChunkSize * 4
        let tokenDrop = 3
        let splitCount = (tokenDrop > 0 ? 1 : 0) + 1
        
        var z = z
        let pseudoTotalTokens = z.dim(2) + tokenDrop
        var padTokens = 0
        let remainder = pseudoTotalTokens % tokensChunkSize
        if remainder != 0 {
            padTokens = tokensChunkSize - remainder
        }
        var numChunks = (pseudoTotalTokens + padTokens) / tokensChunkSize - (tokenDrop > 0 ? 1 : 0)
        if numChunks < 1 {
            padTokens += tokensChunkSize
            numChunks += 1
        }
        
        if padTokens > 0 {
            let lastZ = z[0..., 0..., (z.dim(2) - 1) ..< z.dim(2), 0..., 0...]
            let padZ = broadcast(lastZ, to: [z.dim(0), z.dim(1), padTokens, z.dim(3), z.dim(4)])
            z = concatenated([z, padZ], axis: 2)
        }
        
        let outputFrames = decodeTemporalFramePlan(zLen: z.dim(2), numChunks: numChunks, padTokens: padTokens)
        
        var partsToConcat: [MLXArray] = []
        var totalWrittenFrames = 0
        
        func writePart(_ part: MLXArray) {
            let partFrames = part.dim(2)
            if partFrames <= 0 { return }
            let copyFrames = min(partFrames, max(0, outputFrames - totalWrittenFrames))
            if copyFrames > 0 {
                let sliced = sliceDim(part, dim: 2, start: 0, end: copyFrames)
                partsToConcat.append(sliced)
                totalWrittenFrames += copyFrames
            }
        }
        
        var decOverlap: MLXArray? = nil
        
        for i in 0 ..< numChunks {
            let tStartIdx = i * tokensChunkSize
            let tEndIdx = tStartIdx + tokensChunkSize + tokenOverlap
            let clipZ = z[0..., 0..., tStartIdx ..< tEndIdx, 0..., 0...]
            
            let clipDec = tiledDecode(clipZ)
            
            for j in 0 ..< splitCount {
                let fStartIdx = j * chunkDec
                let fEndIdx = min(fStartIdx + chunkDec, clipDec.dim(2))
                var clipDecChunk = sliceDim(clipDec, dim: 2, start: fStartIdx, end: fEndIdx)
                clipDecChunk = sliceDim(clipDecChunk, dim: 2, start: framePrePadding, end: clipDecChunk.dim(2))
                
                if j == 0 {
                    if let overlap = decOverlap {
                        clipDecChunk = blend(overlap, clipDecChunk, blendExtent: frameOverlap, dim: 2)
                        decOverlap = nil
                    }
                    writePart(clipDecChunk)
                } else {
                    decOverlap = clipDecChunk
                }
            }
        }
        
        if let overlap = decOverlap {
            writePart(overlap)
            decOverlap = nil
        }
        
        return concatenated(partsToConcat, axis: 2)
    }
    
    public func decode(_ z: MLXArray) -> MLXArray {
        let meanVal = latentsMean.reshaped([1, 24, 1, 1, 1])
        let stdVal = latentsStd.reshaped([1, 24, 1, 1, 1])
        let scaledZ = z * stdVal + meanVal
        
        var dec: MLXArray
        if z.dim(2) == 1 {
            dec = tiledDecode(scaledZ)
            dec = sliceDim(dec, dim: 2, start: dec.dim(2) - 1, end: dec.dim(2))
        } else {
            dec = decodeTemporal(scaledZ)
        }
        
        let fDec = dec.asType(.float32)
        let out = (fDec * pixelStd.asType(.float32) + pixelMean.asType(.float32))
        let clamped = minimum(maximum(out, 0.0), 1.0)
        return clamped * 2.0 - 1.0
    }
}
