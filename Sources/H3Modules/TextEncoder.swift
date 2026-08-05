import Foundation
import MLX
import MLXNN
import MLXFast
import H3Foundation

/// Qwen3-VL-32B, the conditioning encoder — language path only.
///
/// Three things about this checkpoint are unusual and all three are load-bearing:
///
///  1. **It is truncated to 50 of 64 layers** (`h3_language_layers: 50` in the
///     file's own metadata), because H3 consumes the layer-50 hidden state.
///  2. **There is no final norm and no lm_head** — not "we skip them", they are
///     absent from the file. So `text_cond` is the raw hidden state out of the
///     last layer, unnormalized. `layer_norm_hidden_state=False` in the
///     reference is a description of the checkpoint, not a choice.
///  3. **Attention is causal.** It is a decoder being used as an encoder.
///
/// The vision tower ships in the same file (27 blocks, 1152-dim) and is not
/// ported here: it is only reachable through image prompts, which the DiT
/// payload path does not support yet either.
package struct TextEncoderConfig: Sendable, Equatable {
    package var hiddenSize = 5120
    package var numLayers = 50
    package var numHeads = 64
    /// Grouped-query attention: 64 query heads share 8 key/value heads.
    package var numKVHeads = 8
    package var headDim = 128
    package var intermediateSize = 25600
    package var vocabSize = 151_936
    package var rmsNormEps: Float = 1e-6
    /// 5e6, not the 1e6 of plain Qwen3 — the VL variant widens it.
    package var ropeTheta: Float = 5_000_000.0
    /// Interleaved-mRoPE band widths for t, h, w. They sum to `headDim / 2`.
    package var ropeDims = [24, 20, 20]
    package init() {}

    package var innerDim: Int { numHeads * headDim }
    package var kvDim: Int { numKVHeads * headDim }
}

package final class TextEncoder {
    package let config: TextEncoderConfig
    package let url: URL

    package struct Layer {
        package let inputNorm: H3RMSNorm
        package let postAttnNorm: H3RMSNorm
        package let q: MLXArray, k: MLXArray, v: MLXArray, o: MLXArray
        package let qNorm: H3RMSNorm, kNorm: H3RMSNorm
        package let gate: MLXArray, up: MLXArray, down: MLXArray
    }

    package let embedTokens: MLXArray
    package let layers: [Layer]

    package enum Error: Swift.Error, CustomStringConvertible {
        case missing(String)
        case unexpected(String)
        package var description: String {
            switch self {
            case .missing(let n): "text encoder checkpoint has no tensor named \(n)"
            case .unexpected(let m): m
            }
        }
    }

    /// Two vendors name the language stack differently. The payload is the
    /// same; only the prefix moves.
    ///
    ///     Comfy-Org      model.layers.N.*            visual.*
    ///     DeepBeepMeep   model.language_model.layers.N.*   model.visual.*
    ///
    /// Detected from the keys rather than assumed, and reported, because a
    /// silent wrong guess here fails as "missing tensor" fifty layers deep.
    package static func languagePrefix(_ names: some Collection<String>) throws -> String {
        let s = Set(names)
        if s.contains("model.layers.0.self_attn.q_proj.weight") { return "model." }
        if s.contains("model.language_model.layers.0.self_attn.q_proj.weight") {
            return "model.language_model."
        }
        throw Error.unexpected("cannot find the language stack under either "
                              + "`model.layers.` or `model.language_model.layers.` — "
                              + "is this a Qwen3-VL conditioning checkpoint?")
    }

    package init(url: URL, config: TextEncoderConfig = TextEncoderConfig()) throws {
        self.url = url
        self.config = config
        let all = try MLX.loadArrays(url: url)
        let p = try Self.languagePrefix(all.keys)

        func w(_ n: String) throws -> MLXArray {
            guard let a = all[n] else { throw Error.missing(n) }
            return a
        }
        // Layer count comes from the file, and disagreeing with the reference
        // is a loud failure: a 64-layer checkpoint would silently produce the
        // wrong hidden state.
        let present = all.keys.compactMap { key -> Int? in
            guard key.hasPrefix(p + "layers.") else { return nil }
            let rest = key.dropFirst((p + "layers.").count)
            guard let dot = rest.firstIndex(of: ".") else { return nil }
            return Int(rest[rest.startIndex ..< dot])
        }
        let found = (present.max() ?? -1) + 1
        guard found == config.numLayers else {
            throw Error.unexpected("checkpoint has \(found) language layers, expected "
                                   + "\(config.numLayers). H3 consumes the layer-50 state; "
                                   + "an untruncated 64-layer encoder is the wrong file.")
        }

        self.embedTokens = try w(p + "embed_tokens.weight")
        var built: [Layer] = []
        built.reserveCapacity(config.numLayers)
        for i in 0 ..< config.numLayers {
            let b = p + "layers.\(i)."
            built.append(Layer(
                inputNorm: H3RMSNorm(weight: try w(b + "input_layernorm.weight"),
                                     eps: config.rmsNormEps),
                postAttnNorm: H3RMSNorm(weight: try w(b + "post_attention_layernorm.weight"),
                                        eps: config.rmsNormEps),
                q: try w(b + "self_attn.q_proj.weight"),
                k: try w(b + "self_attn.k_proj.weight"),
                v: try w(b + "self_attn.v_proj.weight"),
                o: try w(b + "self_attn.o_proj.weight"),
                qNorm: H3RMSNorm(weight: try w(b + "self_attn.q_norm.weight"),
                                 eps: config.rmsNormEps),
                kNorm: H3RMSNorm(weight: try w(b + "self_attn.k_norm.weight"),
                                 eps: config.rmsNormEps),
                gate: try w(b + "mlp.gate_proj.weight"),
                up: try w(b + "mlp.up_proj.weight"),
                down: try w(b + "mlp.down_proj.weight")))
        }
        self.layers = built
    }

    /// `[S, headDim]` cos and sin for positions `0 ..< count`.
    ///
    /// Plain RoPE. A pure-text prompt gets `arange(S)` as a single row, which is
    /// what this computes; image prompts take ``mrope(positionIds:dtype:)``
    /// instead.
    func rope(count: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let half = config.headDim / 2
        let exponent = MLXArray(0 ..< half).asType(.float32) * (2.0 / Float(config.headDim))
        let invFreq = 1.0 / pow(MLXArray(config.ropeTheta), exponent)
        let pos = MLXArray(0 ..< count).asType(.float32).reshaped([count, 1])
        let freqs = pos * invFreq.reshaped([1, half])              // [S, half]
        let emb = concatenated([freqs, freqs], axis: -1)           // [S, headDim]
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }

    /// **Interleaved** mRoPE, for three rows of position ids.
    ///
    /// Qwen3-VL does not give t, h and w contiguous slices of the frequency
    /// band. T is the default everywhere, and h and w then *replace every third
    /// dimension*: h at `1, 4, 7, ...` and w at `2, 5, 8, ...`, both stopping at
    /// `ropeDims[axis] * 3`. With `ropeDims = [24, 20, 20]` that leaves h and w
    /// 20 dimensions each and t the remaining 24 — the same split the
    /// non-interleaved layout would have used, scattered rather than blocked.
    ///
    /// The contiguous `mrope_section` branch in the same reference function is
    /// the Qwen2-VL layout. Both are present; only this one applies here.
    ///
    /// - Parameter positionIds: `[3, S]`, float-valued t/h/w rows.
    func mrope(positionIds: MLXArray, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let half = config.headDim / 2
        let s = positionIds.dim(1)
        let exponent = MLXArray(0 ..< half).asType(.float32) * (2.0 / Float(config.headDim))
        let invFreq = 1.0 / pow(MLXArray(config.ropeTheta), exponent)   // [half]

        // [3, S, half]
        let freqs = positionIds.asType(.float32).reshaped([3, s, 1]) * invFreq.reshaped([1, 1, half])

        // Start from t, then overwrite the h and w positions.
        var lane = [Int32](repeating: 0, count: half)
        for (axis, offset) in [(1, 1), (2, 2)] {
            var i = offset
            while i < config.ropeDims[axis] * 3 && i < half {
                lane[i] = Int32(axis)
                i += 3
            }
        }
        let laneIdx = broadcast(MLXArray(lane).reshaped([1, 1, half]), to: [1, s, half])
        let inter = takeAlong(freqs, laneIdx, axis: 0).squeezed(axis: 0)  // [S, half]
        let emb = concatenated([inter, inter], axis: -1)                // [S, headDim]
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }

    /// Split-half rotation on `[1, heads, S, headDim]`.
    static func applyRoPE(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let lo = x[.ellipsis, 0 ..< half]
        let hi = x[.ellipsis, half ..< (2 * half)]
        let cLo = c[0..., 0 ..< half], sLo = s[0..., 0 ..< half]
        return concatenated([lo * cLo - hi * sLo, hi * cLo + lo * sLo], axis: -1)
    }

    /// Additive causal mask. The reference uses `finfo(dtype).min / 4` rather
    /// than -inf; both underflow to zero through softmax, and staying finite
    /// avoids NaN if a row were ever fully masked.
    static func causalMask(_ n: Int, dtype: DType) -> MLXArray {
        let idx = MLXArray(0 ..< n)
        let rows = idx.reshaped([n, 1]), cols = idx.reshaped([1, n])
        let blocked = cols .> rows
        return MLX.where(blocked, MLXArray(-1e30 as Float), MLXArray(0 as Float)).asType(dtype)
    }

    /// Token ids -> `[1, S, hiddenSize]` embeddings, the encoder's true input.
    ///
    /// The embedding table is bf16 in the checkpoint and the stack runs fp32,
    /// so this upcasts once here rather than per layer.
    package func embed(ids: [Int]) -> MLXArray {
        let idx = MLXArray(ids.map { Int32($0) })
        return embedTokens[idx].expandedDimensions(axis: 0).asType(.float32)
    }

    /// Text in, conditioning out — the whole encode span.
    ///
    /// Returns the tags alongside, because the packed layout needs a modality
    /// per text token and a pure-text prompt is not the only possible case.
    package func encode(_ text: String, tokenizer: Qwen2Tokenizer,
                       computeDType: DType = .float32)
        -> (cond: MLXArray, ids: [Int], tags: [Int]) {
        let ids = tokenizer.encodePrompt(text)
        var taps = Taps()
        let cond = callAsFunction(embeds: embed(ids: ids),
                                  computeDType: computeDType, taps: &taps)
        return (cond, ids, tokenizer.textTags(count: ids.count))
    }

    package struct Taps {
        /// Hidden state after each recorded layer index.
        package var layers: [Int: MLXArray] = [:]
        package init() {}
    }

    /// Runs the language stack over pre-computed token embeddings.
    ///
    /// Taking embeddings rather than token ids is deliberate: it is the same
    /// boundary the parity contract uses (`te.*.embed_tokens` is a golden
    /// input), so this is verifiable before any tokenizer exists.
    ///
    /// **Compute in fp32.** The default is not a safety margin, it is what the
    /// reference does: against the same golden, fp32 lands at cos 1.000000000 /
    /// rel_rms 1.8e-06 while bf16 lands at 1.1e-02 with a systematic 1% deficit
    /// in magnitude. If the reference had run bf16, our bf16 would be the closer
    /// of the two; it is worse by four orders of magnitude. The mechanism inside
    /// ComfyUI is not traced — the measurement is the fact.
    ///
    /// It is also nearly free here: weights stay bf16 in memory and are upcast
    /// per op, so peak residency moves 48.8 -> 51.8 GB.
    ///
    /// - Parameter embeds: `[1, S, hiddenSize]`
    /// - Returns: `[1, S, hiddenSize]` — the **unnormalized** layer-50 state.
    package func callAsFunction(embeds: MLXArray,
                               computeDType: DType = .float32,
                               recordLayers: Set<Int> = [],
                               positionIds: MLXArray? = nil,
                               visualSpans: [(start: Int, count: Int)] = [],
                               deepstack: [MLXArray] = [],
                               taps: inout Taps) -> MLXArray {
        let s = embeds.dim(1)
        var h = embeds.asType(computeDType)
        // Three rows of position ids means an image is present; one row, or
        // none, is the text path.
        let (rc, rs) = positionIds.map { mrope(positionIds: $0, dtype: computeDType) }
                    ?? rope(count: s, dtype: computeDType)
        let mask = Self.causalMask(s, dtype: computeDType)
        let scale = 1.0 / Float(config.headDim).squareRoot()

        for (i, l) in layers.enumerated() {
            // attention
            let x = l.inputNorm(h)[0]                              // [S, hidden]
            var q = matmul(x, l.q.T).reshaped([s, config.numHeads, config.headDim])
            var k = matmul(x, l.k.T).reshaped([s, config.numKVHeads, config.headDim])
            let v = matmul(x, l.v.T).reshaped([s, config.numKVHeads, config.headDim])
            q = l.qNorm(q)
            k = l.kNorm(k)
            let qh = Self.applyRoPE(q.transposed(1, 0, 2).expandedDimensions(axis: 0),
                                    cos: rc, sin: rs)
            let kh = Self.applyRoPE(k.transposed(1, 0, 2).expandedDimensions(axis: 0),
                                    cos: rc, sin: rs)
            let vh = v.transposed(1, 0, 2).expandedDimensions(axis: 0)
            // MLX's SDPA handles the 64:8 grouping itself.
            let o = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh,
                                                      scale: scale, mask: mask)
            let merged = o.squeezed(axis: 0).transposed(1, 0, 2)
                          .reshaped([s, config.innerDim])
            h = h + matmul(merged, l.o.T).expandedDimensions(axis: 0)

            // SwiGLU MLP
            let y = l.postAttnNorm(h)[0]
            let g = matmul(y, l.gate.T), u = matmul(y, l.up.T)
            h = h + matmul(silu(g) * u, l.down.T).expandedDimensions(axis: 0)

            // Tapped BEFORE the deepstack injection, because that is where the
            // reference's hook sits: it fires on the layer module's return, and
            // the injection happens in the enclosing loop afterwards. Recording
            // it after makes layers 0-2 disagree with the golden by exactly the
            // injected term while every later layer matches — which reads like
            // an error that heals, and errors do not heal.
            if recordLayers.contains(i) { taps.layers[i] = h }

            // DeepStack: the vision tower's layer-8/16/24 features are *added*
            // into the first three language layers, at the image's token
            // positions only. Prefill only, which is all H3 ever does.
            //
            // The reference writes this as a boolean-mask scatter. Here the
            // spans are contiguous by construction, so rebuilding the row from
            // slices does the same job without a scatter kernel.
            if i < deepstack.count, !visualSpans.isEmpty {
                let row = h[0]
                var parts: [MLXArray] = []
                var prev = 0, off = 0
                for sp in visualSpans {
                    if sp.start > prev { parts.append(row[prev ..< sp.start]) }
                    parts.append(row[sp.start ..< (sp.start + sp.count)]
                                 + deepstack[i][off ..< (off + sp.count)].asType(computeDType))
                    prev = sp.start + sp.count
                    off += sp.count
                }
                if prev < s { parts.append(row[prev ..< s]) }
                h = concatenated(parts, axis: 0).expandedDimensions(axis: 0)
            }
            if i % 10 == 0 { eval(h) }
        }
        // No final norm: the checkpoint does not carry one.
        return h
    }
}
