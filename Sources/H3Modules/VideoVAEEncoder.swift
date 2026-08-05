import Foundation
import MLX
import MLXNN
import H3Foundation

/// `EncoderFCN3D` — the conv half of the MiniMax H3 video VAE.
///
/// The encoder and the decoder are **not** mirror images: the decoder is a ViT
/// (`ViT3DDecoder`, 440 tensors in the checkpoint), this is a 6-level causal-conv
/// ResNet (116 tensors). Porting one teaches you nothing about the other.
///
/// Convolutions run in MLX's channels-last layout; the model's own tensors stay
/// in the reference's `[B, C, T, H, W]` between blocks, and each conv transposes
/// in and out. That keeps the shapes readable against `comfy/ldm/minimax/vae.py`
/// at the cost of some shuffling.

/// Spatial tiling, shared by encode and decode.
///
/// Lifted out of ``VideoVAE`` — which had them as instance methods that never
/// touched instance state — so the encoder can reuse them rather than grow a
/// second copy that drifts.
public enum VaeTiling {
    public static let tileSize = 256
    public static let tileOverlapMin = 64
    /// `prod(space_down)` — pixels per latent cell.
    public static let vaeRatio = 16

    /// Tile starts, lengths and overlaps for one axis.
    ///
    /// Overlaps are grown in whole `vaeRatio` units so that every latent-space
    /// overlap is an integer; a fractional one would make the blend seams land
    /// between latent cells.
    public static func splitTiles(inputLen: Int, tileSize: Int = tileSize,
                                  tileOverlapMin: Int = tileOverlapMin,
                                  vaeRatio: Int = vaeRatio)
        -> (starts: [Int], lengths: [Int], overlaps: [Int]) {
        if tileSize >= inputLen { return ([0], [inputLen], []) }
        var n = Int(ceil(Double(inputLen) / Double(tileSize)))
        var overlaps: [Int] = []
        while true {
            overlaps = Array(repeating: tileOverlapMin, count: n - 1)
            if tileSize * n - overlaps.reduce(0, +) - inputLen < 0 { n += 1 } else { break }
        }
        let remaining = tileSize * n - overlaps.reduce(0, +) - inputLen
        for i in 0 ..< (remaining / vaeRatio) { overlaps[i % (n - 1)] += vaeRatio }
        var starts = [0]
        for i in 0 ..< (n - 1) { starts.append(starts.last! + tileSize - overlaps[i]) }
        return (starts, Array(repeating: tileSize, count: n), overlaps)
    }

    static func sliceDim(_ a: MLXArray, dim: Int, start: Int, end: Int) -> MLXArray {
        let axis = dim < 0 ? a.ndim + dim : dim
        let size = a.dim(axis)
        let s = max(0, min(size, start)), e = max(0, min(size, end))
        var idx: [any MLXArrayIndex] = []
        for i in 0 ..< a.ndim { idx.append(i == axis ? s ..< e : 0 ..< a.dim(i)) }
        return a[idx]
    }

    /// Linear cross-fade of `a`'s trailing `blendExtent` into `b`'s leading one.
    public static func blend(_ a: MLXArray, _ b: MLXArray,
                             blendExtent: Int, dim: Int) -> MLXArray {
        let ndim = a.ndim
        // Callers pass -1 and -2. MLXArray.dim() and sliceDim() both accept
        // negative axes; a Swift Array subscript does not, and indexing
        // weightShape[-1] traps. Normalise once, here.
        let axis = dim < 0 ? ndim + dim : dim
        precondition(axis >= 0 && axis < ndim, "blend axis \(dim) outside 0..<\(ndim)")
        let extent = min(a.dim(axis), b.dim(axis), blendExtent)
        if extent <= 0 { return b }

        let positions = MLXArray(0 ..< extent).asType(b.dtype)
        var shape = Array(repeating: 1, count: ndim)
        shape[axis] = extent
        let wA = (1.0 - positions / Float(extent)).reshaped(shape)
        let wB = (positions / Float(extent)).reshaped(shape)

        let blended = sliceDim(a, dim: axis, start: a.dim(axis) - extent, end: a.dim(axis)) * wA
                    + sliceDim(b, dim: axis, start: 0, end: extent) * wB
        if extent < b.dim(axis) {
            return concatenated([blended, sliceDim(b, dim: axis, start: extent, end: b.dim(axis))],
                                axis: axis)
        }
        return blended
    }
}

/// GroupNorm with statistics taken **per frame**: time is folded into the batch
/// so a frame never borrows another frame's mean.
///
/// Written out rather than reached for from MLXNN, because MLXNN's `GroupNorm`
/// normalizes the **last** axis and the reference normalizes the second. Handing
/// it `[B*T, C, 1, H, W]` normalizes over width and is silent about it.
struct TemporalIsolatedGroupNorm {
    let weight: MLXArray
    let bias: MLXArray
    let groups: Int
    /// 1e-6 here, not the 1e-5 that most frameworks default to.
    let eps: Float

    init(weight: MLXArray, bias: MLXArray, groups: Int = 32, eps: Float = 1e-6) {
        self.weight = weight
        self.bias = bias
        self.groups = groups
        self.eps = eps
    }

    /// `x` is `[B, C, T, H, W]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), c = x.dim(1), t = x.dim(2), h = x.dim(3), w = x.dim(4)
        let f = x.asType(.float32)
        // group over channels only; each (frame, group) gets its own statistics
        let g = f.transposed(0, 2, 1, 3, 4).reshaped([b * t, groups, (c / groups) * h * w])
        let mu = mean(g, axis: -1, keepDims: true)
        let d = g - mu
        let v = mean(d * d, axis: -1, keepDims: true)
        let n = (d * rsqrt(v + eps)).reshaped([b, t, c, h, w]).transposed(0, 2, 1, 3, 4)
        let shape = [1, c, 1, 1, 1]
        return (n * weight.asType(.float32).reshaped(shape)
                  + bias.asType(.float32).reshaped(shape)).asType(x.dtype)
    }
}

/// Reflect padding on the spatial axes, causal zero padding on time.
///
/// Causal means the whole temporal pad goes on the **front** and is twice the
/// nominal width — the convolution never sees a future frame.
struct CausalConv3d {
    /// `[O, kT, kH, kW, I]` — MLX's conv3d layout, transposed once at load.
    let weight: MLXArray
    let bias: MLXArray?
    let stride: [Int]
    let padding: (t: Int, h: Int, w: Int)

    init(weight: MLXArray, bias: MLXArray?, stride: [Int] = [1, 1, 1],
         padding: (t: Int, h: Int, w: Int) = (0, 0, 0)) {
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
    }

    /// Reflect pad, excluding the edge row itself — `F.pad(..., mode="reflect")`.
    private static func reflect(_ x: MLXArray, axis: Int, width: Int) -> MLXArray {
        guard width > 0 else { return x }
        let n = x.dim(axis)
        precondition(width < n, "reflect pad \(width) needs at least \(width + 1) rows on axis \(axis)")
        let lead = (1 ... width).reversed().map { x.take(MLXArray(Int32($0)), axis: axis)
                                                   .expandedDimensions(axis: axis) }
        let tail = (1 ... width).map { x.take(MLXArray(Int32(n - 1 - $0)), axis: axis)
                                        .expandedDimensions(axis: axis) }
        return concatenated(lead + [x] + tail, axis: axis)
    }

    /// `x` is `[B, C, T, H, W]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var p = x
        if padding.h > 0 || padding.w > 0 {
            p = Self.reflect(p, axis: 3, width: padding.h)
            p = Self.reflect(p, axis: 4, width: padding.w)
        }
        if padding.t > 0 {
            // Front-only, double width, zeros. On a single frame the reference
            // instead drops the temporal taps that would read the padding; the
            // two agree because those taps are multiplied by zero either way.
            let z = MLXArray.zeros([p.dim(0), p.dim(1), padding.t * 2, p.dim(3), p.dim(4)],
                                   dtype: p.dtype)
            p = concatenated([z, p], axis: 2)
        }
        var out = conv3d(p.transposed(0, 2, 3, 4, 1), weight,
                         stride: .init((stride[0], stride[1], stride[2])),
                         padding: .init((0, 0, 0)))
        if let bias { out = out + bias.reshaped([1, 1, 1, 1, bias.size]) }
        return out.transposed(0, 4, 1, 2, 3)
    }
}

struct VideoResnetBlock3D {
    let norm1: TemporalIsolatedGroupNorm
    let conv1: CausalConv3d
    let norm2: TemporalIsolatedGroupNorm
    let conv2: CausalConv3d
    let ninShortcut: CausalConv3d?

    init(inChannels: Int, outChannels: Int, prefix: String,
         weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.norm1 = TemporalIsolatedGroupNorm(weight: try get(prefix + "norm1.weight"),
                                               bias: try get(prefix + "norm1.bias"))
        self.norm2 = TemporalIsolatedGroupNorm(weight: try get(prefix + "norm2.weight"),
                                               bias: try get(prefix + "norm2.bias"))
        self.conv1 = CausalConv3d(weight: try get(prefix + "conv1.weight").transposed(0, 2, 3, 4, 1),
                                  bias: try get(prefix + "conv1.bias"), padding: (1, 1, 1))
        self.conv2 = CausalConv3d(weight: try get(prefix + "conv2.weight").transposed(0, 2, 3, 4, 1),
                                  bias: try get(prefix + "conv2.bias"), padding: (1, 1, 1))
        // kernel 1, so no padding at all — the 1x1x1 shortcut is a channel map.
        self.ninShortcut = inChannels == outChannels ? nil
            : CausalConv3d(weight: try get(prefix + "nin_shortcut.weight").transposed(0, 2, 3, 4, 1),
                           bias: try get(prefix + "nin_shortcut.bias"))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        return h + (ninShortcut?(x) ?? x)
    }
}

struct VideoDownsample3D {
    let conv: CausalConv3d
    let spaceStride: Int

    init(timeStride: Int, spaceStride: Int, prefix: String,
         weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.spaceStride = spaceStride
        self.conv = CausalConv3d(weight: try get(prefix + "conv.weight").transposed(0, 2, 3, 4, 1),
                                 bias: try get(prefix + "conv.bias"),
                                 stride: [timeStride, spaceStride, spaceStride],
                                 padding: (1, 0, 0))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard spaceStride == 2 else { return conv(x) }
        // One extra row and column on the trailing edge, reflected — not zeros.
        var p = x
        let h = p.dim(3), w = p.dim(4)
        p = concatenated([p, p[0..., 0..., 0..., (h - 2) ..< (h - 1), 0...]], axis: 3)
        p = concatenated([p, p[0..., 0..., 0..., 0..., (w - 2) ..< (w - 1)]], axis: 4)
        return conv(p)
    }
}

struct VideoEncoderLevel {
    let blocks: [VideoResnetBlock3D]
    let downsample: VideoDownsample3D?

    init(inChannels: Int, midChannels: Int, timeStride: Int, spaceStride: Int,
         numResBlocks: Int, prefix: String, weights: [String: MLXArray]) throws {
        self.blocks = try (0 ..< numResBlocks).map { i in
            try VideoResnetBlock3D(inChannels: i == 0 ? inChannels : midChannels,
                                   outChannels: midChannels,
                                   prefix: prefix + "block.\(i).", weights: weights)
        }
        self.downsample = spaceStride * timeStride > 1
            ? try VideoDownsample3D(timeStride: timeStride, spaceStride: spaceStride,
                                    prefix: prefix + "downsample.", weights: weights)
            : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for b in blocks { h = b(h) }
        return downsample?(h) ?? h
    }
}

public final class VideoVAEEncoder {
    /// ImageNet statistics. Pixels arrive in [-1, 1], are mapped to [0, 1], then
    /// standardised by these — the encoder never sees the raw range.
    public static let pixelMean: [Float] = [0.485, 0.456, 0.406]
    public static let pixelStd: [Float] = [0.229, 0.224, 0.225]

    static let chMult = [1, 2, 2, 4, 4, 8]
    static let spaceDown = [2, 2, 2, 2, 1, 1]
    static let timeDown = [1, 2, 2, 1, 1, 1]
    static let baseCh = 128
    static let numResBlocks = 2
    public static let zChannels = 24
    /// 17 frames per temporal clip, and 3 latent tokens dropped per clip.
    public static let clipLength = 17
    /// Tokens trimmed from the tail after the clips are concatenated.
    public static let tokenDrop = 3
    /// Above this on either spatial axis the reference tiles.
    public static let tileSize = 256

    let convIn: CausalConv3d
    let levels: [VideoEncoderLevel]
    let normOut: TemporalIsolatedGroupNorm
    let convOut: CausalConv3d
    let quantConv: MLXArray
    let quantConvBias: MLXArray
    let latentsMean: MLXArray
    let latentsStd: MLXArray

    public init(weights: [String: MLXArray]) throws {
        func get(_ n: String) throws -> MLXArray {
            guard let a = weights[n] else { throw H3Weights.Error.missing(n) }
            return a
        }
        self.latentsMean = try get("latents_mean")
        self.latentsStd = try get("latents_std")

        self.convIn = CausalConv3d(weight: try get("encoder.conv_in.weight").transposed(0, 2, 3, 4, 1),
                                   bias: try get("encoder.conv_in.bias"), padding: (1, 1, 1))

        let mid = Self.chMult.map { Self.baseCh * $0 }
        let inputs = [mid[0]] + mid.dropLast()
        self.levels = try (0 ..< Self.chMult.count).map { i in
            try VideoEncoderLevel(inChannels: inputs[i], midChannels: mid[i],
                                  timeStride: Self.timeDown[i], spaceStride: Self.spaceDown[i],
                                  numResBlocks: Self.numResBlocks,
                                  prefix: "encoder.down.\(i).", weights: weights)
        }

        self.normOut = TemporalIsolatedGroupNorm(weight: try get("encoder.norm_out.weight"),
                                                 bias: try get("encoder.norm_out.bias"))
        self.convOut = CausalConv3d(weight: try get("encoder.conv_out.weight").transposed(0, 2, 3, 4, 1),
                                    bias: try get("encoder.conv_out.bias"), padding: (1, 1, 1))
        // Conv3d with kernel 1: [2*embed, 2*z, 1, 1, 1] is a channel matmul.
        let q = try get("quant_conv.weight")
        self.quantConv = q.reshaped([q.dim(0), q.dim(1)])
        self.quantConvBias = try get("quant_conv.bias")
    }

    /// Named after the reference's own module paths, so a failing tap points at
    /// a line in `comfy/ldm/minimax/vae.py` rather than at "the encoder".
    public struct Taps {
        public var convIn: MLXArray?
        public var levelBlocks: [String: MLXArray] = [:]
        public var normOut: MLXArray?
        public var convOut: MLXArray?
        public var quantConv: MLXArray?
        public init() {}
    }

    /// The conv stack, returning `[B, 48, T_lat, H/16, W/16]` moments.
    public func moments(_ x: MLXArray, taps: inout Taps) -> MLXArray {
        var h = convIn(x)
        taps.convIn = h
        for (i, l) in levels.enumerated() {
            for (j, b) in l.blocks.enumerated() {
                h = b(h)
                taps.levelBlocks["encoder.down.\(i).block.\(j)"] = h
            }
            if let d = l.downsample {
                h = d(h)
                taps.levelBlocks["encoder.down.\(i).downsample"] = h
            }
        }
        let n = normOut(h)
        taps.normOut = n
        h = convOut(silu(n))
        taps.convOut = h
        // quant_conv, as a matmul over the channel axis
        let c = h.dim(1)
        let flat = h.transposed(0, 2, 3, 4, 1).reshaped([-1, c])
        let m = (matmul(flat, quantConv.T) + quantConvBias)
            .reshaped([h.dim(0), h.dim(2), h.dim(3), h.dim(4), quantConv.dim(0)])
            .transposed(0, 4, 1, 2, 3)
        taps.quantConv = m
        return m
    }

    /// Pixels `[B, 3, T, H, W]` in [-1, 1] -> normalized latents `[B, 24, T_lat, H/16, W/16]`.
    ///
    /// **Returns the posterior mean — there is no sampling.** The second half of
    /// the moments is the log-variance and the reference discards it.
    ///
    /// This is the single-shot path: no spatial tiling and no temporal
    /// chunking. Both of those are chunking strategies over this same function
    /// — see ``tiledMoments(_:)`` and ``temporalMoments(_:)`` — and ``encode(_:)``
    /// is what routes between them.
    /// `[-1,1] -> [0,1] -> ImageNet mean/std`. The encoder never sees raw
    /// signed pixels.
    func normalizePixels(_ pixels: MLXArray) -> MLXArray {
        let mean3 = MLXArray(Self.pixelMean).reshaped([1, 3, 1, 1, 1])
        let std3 = MLXArray(Self.pixelStd).reshaped([1, 3, 1, 1, 1])
        return ((pixels.asType(.float32) + 1.0) * 0.5 - mean3) / std3
    }

    public func encodeSingleShot(_ pixels: MLXArray, taps: inout Taps) -> MLXArray {
        let m = moments(normalizePixels(pixels), taps: &taps)
        let mean = m[0..., 0 ..< Self.zChannels, 0..., 0..., 0...]
        let zm = latentsMean.reshaped([1, Self.zChannels, 1, 1, 1])
        let zs = latentsStd.reshaped([1, Self.zChannels, 1, 1, 1])
        return (mean - zm) / zs
    }

    /// Single frame in, single latent frame out — the `T == 1` path, which is
    /// exactly what a keyframe or a reference image needs.
    ///
    /// The reference truncates to the last latent frame here because the causal
    /// front padding manufactures leading frames that carry no information.
    public func encodeImage(_ pixels: MLXArray, taps: inout Taps) -> MLXArray {
        precondition(pixels.dim(2) == 1,
                     "encodeImage wants one frame, got \(pixels.dim(2)); "
                     + "multi-frame clips need the temporal chunking that is not ported")
        let z = encodeSingleShot(pixels, taps: &taps)
        return z[0..., 0..., (z.dim(2) - 1) ..< z.dim(2), 0..., 0...]
    }

    public func encodeImage(_ pixels: MLXArray) -> MLXArray {
        var t = Taps()
        return encodeImage(pixels, taps: &t)
    }

    /// `tiled_encode` — a grid of `tileSize` tiles, cross-faded in latent space.
    ///
    /// This is not an optimisation that can be skipped at small cost. Anything
    /// wider or taller than 256 px goes through it in the reference: at 864x480
    /// that is a 5x3 grid, fifteen full passes over the conv stack, and the
    /// seams are blended rather than butted. A single-shot pass over the whole
    /// frame produces different numbers everywhere, not just near the seams,
    /// because the causal and reflect padding land at different places.
    public func tiledMoments(_ x: MLXArray) -> MLXArray {
        let h = x.dim(3), w = x.dim(4)
        let (yIdx, yLen, yOverlap) = VaeTiling.splitTiles(inputLen: h)
        let (xIdx, xLen, xOverlap) = VaeTiling.splitTiles(inputLen: w)

        var rows: [[MLXArray]] = []
        for (iPos, iLen) in zip(yIdx, yLen) {
            var row: [MLXArray] = []
            for (jPos, jLen) in zip(xIdx, xLen) {
                var t = VaeTiling.sliceDim(x, dim: 3, start: iPos, end: iPos + iLen)
                t = VaeTiling.sliceDim(t, dim: 4, start: jPos, end: jPos + jLen)
                var scratch = Taps()
                row.append(moments(t, taps: &scratch))
            }
            rows.append(row)
        }

        let latY = yOverlap.map { $0 / VaeTiling.vaeRatio }
        let latX = xOverlap.map { $0 / VaeTiling.vaeRatio }
        var resultRows: [MLXArray] = []
        for i in rows.indices {
            var resultRow: [MLXArray] = []
            for j in rows[i].indices {
                var tile = rows[i][j]
                if i > 0 { tile = VaeTiling.blend(rows[i - 1][j], tile, blendExtent: latY[i - 1], dim: -2) }
                if j > 0 { tile = VaeTiling.blend(rows[i][j - 1], tile, blendExtent: latX[j - 1], dim: -1) }
                if i < rows.count - 1 {
                    tile = VaeTiling.sliceDim(tile, dim: -2, start: 0, end: tile.dim(-2) - latY[i])
                }
                if j < rows[i].count - 1 {
                    tile = VaeTiling.sliceDim(tile, dim: -1, start: 0, end: tile.dim(-1) - latX[j])
                }
                resultRow.append(tile)
            }
            resultRows.append(concatenated(resultRow, axis: -1))
        }
        return concatenated(resultRows, axis: -2)
    }

    /// `_adaptive_encode` — the reference constructs the VAE with `tiling=True`,
    /// so this is always the tiled call. `splitTiles` degenerates to one tile
    /// when the frame fits, which is why the small-frame case needs no branch.
    func adaptiveMoments(_ normalized: MLXArray) -> MLXArray {
        tiledMoments(normalized)
    }

    /// `encode_temporal` — the multi-frame path, as moments.
    ///
    /// Three steps, and each one is a place a port silently disagrees:
    ///
    /// 1. **Pad by repeating the last frame** up to a multiple of `clipLength`.
    ///    Zero-padding or edge-reflecting instead keeps every shape correct.
    /// 2. **Encode each clip independently.** The clips do not overlap and are
    ///    not blended — unlike the spatial tiles, and unlike `decode_temporal`,
    ///    which does overlap.
    /// 3. **Drop `tokenDrop` tokens off the tail** after the concatenation, not
    ///    per clip. Dropping per clip gives the same count only when there is
    ///    exactly one clip, which is why the single-clip fixture cannot catch it.
    ///
    /// Input is already normalized; the caller owns the pixel statistics.
    ///
    /// `taps` carries the intermediates a golden can localise against — the
    /// padded input and each clip's moments before the concatenation.
    public struct TemporalTaps {
        public var padded: MLXArray?
        public var clips: [MLXArray] = []
        public init() {}
    }

    public func temporalMoments(_ normalized: MLXArray,
                                taps: inout TemporalTaps) -> MLXArray {
        let frames = normalized.dim(2)
        var x = normalized
        let pad = (Self.clipLength - frames % Self.clipLength) % Self.clipLength
        if pad > 0 {
            let last = VaeTiling.sliceDim(x, dim: 2, start: frames - 1, end: frames)
            x = concatenated([x] + Array(repeating: last, count: pad), axis: 2)
        }
        taps.padded = x

        let chunks = x.dim(2) / Self.clipLength
        var z: [MLXArray] = []
        z.reserveCapacity(chunks)
        for i in 0 ..< chunks {
            let clip = VaeTiling.sliceDim(x, dim: 2,
                                          start: i * Self.clipLength,
                                          end: (i + 1) * Self.clipLength)
            z.append(adaptiveMoments(clip))
        }
        taps.clips = z
        var out = concatenated(z, axis: 2)
        if Self.tokenDrop > 0 {
            out = VaeTiling.sliceDim(out, dim: 2, start: 0, end: out.dim(2) - Self.tokenDrop)
        }
        return out
    }

    public func temporalMoments(_ normalized: MLXArray) -> MLXArray {
        var t = TemporalTaps()
        return temporalMoments(normalized, taps: &t)
    }

    /// `[-1, 1]` pixels -> the normalized-pixel tensor the clips actually see.
    /// Public so a parity check can compare the pad without reimplementing it.
    public func normalized(_ pixels: MLXArray) -> MLXArray { normalizePixels(pixels) }

    /// Moments -> normalized latents: take the posterior mean, standardise.
    func normalizeMoments(_ m: MLXArray) -> MLXArray {
        let mean = m[0..., 0 ..< Self.zChannels, 0..., 0..., 0...]
        return (mean - latentsMean.reshaped([1, Self.zChannels, 1, 1, 1]))
             / latentsStd.reshaped([1, Self.zChannels, 1, 1, 1])
    }

    /// Pixels in `[-1, 1]` -> normalized latents, for any frame count.
    ///
    /// One frame keeps the last latent frame (the causal front pad manufactures
    /// leading frames that carry no information); more than one goes through
    /// ``temporalMoments(_:)``. Tiling is the reference's normal path, not a
    /// low-memory fallback, so both branches tile.
    public func encode(_ pixels: MLXArray) -> MLXArray {
        let normalized = normalizePixels(pixels)
        if pixels.dim(2) == 1 {
            let z = normalizeMoments(adaptiveMoments(normalized))
            return z[0..., 0..., (z.dim(2) - 1) ..< z.dim(2), 0..., 0...]
        }
        return normalizeMoments(temporalMoments(normalized))
    }
}
