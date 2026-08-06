// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXNN
import MLXFast
import H3Foundation
import H3Attention

/// AdaLN projection: `chunk(linear(silu(t_emb)), expand)`.
///
/// `t_emb` is `[M, tDim]` for M distinct timesteps; the output is `expand`
/// tensors of `[M * modalities, hidden]`. The reshape interleaves modalities
/// **within** each timestep, which is what makes the row index
/// `timestepRow * modalities + modalityTag`.
package struct AdalnProj {
    package let weight: MLXArray      // [expand * hidden * modalities, tDim]
    package let bias: MLXArray?
    package let expand: Int
    package let modalities: Int
    package let hidden: Int
    package let applySiLU: Bool
    /// Compute the projection in fp32 and round the result back to the weight's
    /// dtype.
    ///
    /// This op is ill-conditioned: `silu(t_emb)` is tiny (t_emb std ~9e-03) and
    /// the K=2688 reduction cancels heavily, so the reference's OWN bf16-vs-fp32
    /// spread here is 1.7e-03 — three orders of magnitude looser than any other
    /// matmul in the block. Any difference in accumulation order lands at that
    /// level, and in bf16 MLX sits 1.55x outside it. In fp32 we land on the
    /// reference's fp32 result to 4 significant figures.
    ///
    /// The cost is real but small: AdaLN weights are 26 of the checkpoint's
    /// 66 GB, so upcasting per call adds roughly 50 GB of transient traffic per
    /// forward — about 0.2% of a 61 s production pass. Measured, not assumed.
    package let computeFP32: Bool
    /// Optional persistent fp32 copy for a render that has enough unified
    /// memory to trade ~48 GiB of stable residency for avoiding the repeated
    /// bf16-to-fp32 conversion of the 50 block AdaLN matrices.
    ///
    /// This is opt-in: capacity is workload-dependent, and the bf16 source
    /// remains the canonical checkpoint representation.
    package let residentFP32Weight: MLXArray?

    package init(weight: MLXArray, bias: MLXArray?, expand: Int, modalities: Int,
                hidden: Int, applySiLU: Bool = true, computeFP32: Bool = true,
                keepFP32Resident: Bool = false) {
        self.weight = weight
        self.bias = bias
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.applySiLU = applySiLU
        self.computeFP32 = computeFP32
        self.residentFP32Weight = keepFP32Resident && computeFP32 ? weight.asType(.float32) : nil
    }

    /// Returns `expand` tensors of `[M * modalities, hidden]`, in the weight's
    /// dtype whatever the internal precision.
    package func callAsFunction(_ tEmb: MLXArray) -> [MLXArray] {
        let out = weight.dtype
        let dt: DType = computeFP32 ? .float32 : out
        let input = applySiLU ? silu(tEmb.asType(dt)) : tEmb.asType(dt)
        let projectionWeight = dt == .float32 ? (residentFP32Weight ?? weight.asType(dt)) : weight
        var x = matmul(input, projectionWeight.T)
        if let bias { x = x + bias.asType(dt) }
        x = x.reshaped([x.dim(0) * modalities, expand * hidden]).asType(out)
        return (0 ..< expand).map { x[0..., ($0 * hidden) ..< (($0 + 1) * hidden)] }
    }
}

/// Split-half rotary embedding.
///
/// The rotation table is `[1, S, 1, rot/2, 2, 2]` holding `[[c, -s], [s, c]]`,
/// and pairs are `(i, i + rot/2)` — **split-half, not interleaved**. Choosing
/// interleaved is the single most common RoPE porting error and produces output
/// that looks structured but is wrong.
///
/// Only the first `rot` channels rotate; the tail passes through untouched.
package enum SplitHalfRoPE {
    /// `x` is `[S, heads, headDim]` or `[B, S, heads, headDim]`; `table` is the reference's rotation table.
    package static func apply(_ x: MLXArray, table: MLXArray) -> MLXArray {
        let half = table.dim(-3)
        let rot = half * 2
        let headDim = x.dim(-1)
        precondition(rot <= headDim, "rot \(rot) exceeds headDim \(headDim)")

        // [1,S,1,half,2,2] -> [S,1,half] so it broadcasts over heads.
        let t = table.reshaped([table.dim(1), half, 2, 2])
        let c = t[0..., 0..., 0, 0].expandedDimensions(axis: 1)
        let negS = t[0..., 0..., 0, 1].expandedDimensions(axis: 1)
        let s = t[0..., 0..., 1, 0].expandedDimensions(axis: 1)
        let c2 = t[0..., 0..., 1, 1].expandedDimensions(axis: 1)

        let parts = x.split(indices: [half, rot], axis: -1)
        let a = parts[0]
        let b = parts[1]
        let ra = c * a + negS * b
        let rb = s * a + c2 * b
        if rot == headDim { return concatenated([ra, rb], axis: -1) }
        return concatenated([ra, rb, parts[2]], axis: -1)
    }
}

/// Attention over the packed sequence.
///
/// The reference runs a fused in-place kernel (`ck.rms_rope_split_half_`), but
/// the oracle measured a hand-written unfused path as **bit-identical**
/// (cos 1.000000000000, rel 0.0 at blocks 0/24/49). So this is the readable
/// form deliberately: match the math, not the fusion.
///
/// Named `AttentionLayer` rather than `H3Attention` because `H3Attention` is the
/// module that owns the backend protocol, and a type that shadows its own
/// module's name reads as a mistake even when it compiles.
package struct AttentionLayer {
    package let qkvWeight: MLXArray     // [3 * inner, hidden], no bias
    package let outWeight: MLXArray     // [hidden, inner], no bias
    package let qNorm: H3RMSNorm
    package let kNorm: H3RMSNorm
    package let heads: Int
    package let headDim: Int
    /// Run the attention op itself in fp32 while the rest of the block stays
    /// bf16. Diagnostic: it isolates whether a deviation comes from the
    /// attention kernel's accumulation or from everything else.
    package let fp32Attention: Bool
    /// Resolved once at model build and held, rather than looked up per call.
    ///
    /// Defaults to dense, so every existing caller — the oracles, the refiner,
    /// anything that does not supply an `AttentionContext` — keeps the numerics
    /// the 225 parity taps were measured on.
    package let backend: any H3AttentionBackend

    package init(qkvWeight: MLXArray, outWeight: MLXArray,
                qNormWeight: MLXArray, kNormWeight: MLXArray,
                heads: Int, headDim: Int, eps: Float, fp32Attention: Bool = false,
                backend: any H3AttentionBackend = SDPABackend()) {
        self.qkvWeight = qkvWeight
        self.outWeight = outWeight
        self.qNorm = H3RMSNorm(weight: qNormWeight, eps: eps)
        self.kNorm = H3RMSNorm(weight: kNormWeight, eps: eps)
        self.heads = heads
        self.headDim = headDim
        self.fp32Attention = fp32Attention
        self.backend = backend
    }

    /// Collects the shape of the proxy-score distribution while a render runs.
    ///
    /// Sol-Attn's single knob works because standardized proxy logits are
    /// "consistently near-Gaussian within each model", which is what makes
    /// `density = 1 - Phi(beta)` true. **H3 is not one of the models the paper
    /// measured that on.** If H3's distribution is skewed or heavy-tailed, beta
    /// does not mean what the published settings assume, and every tau value in
    /// circulation is calibrated against a different distribution.
    ///
    /// This is three floats per call against 226 MB of tensors, so it answers
    /// the question locally rather than shipping q/k/v to a rented GPU.
    package final class ProxyProbe: @unchecked Sendable {
        package var enabled = false
        package var samples: [(skew: Double, kurt: Double, tail1: Double, tail2: Double)] = []
        package init() {}
    }
    package static let proxyProbe = ProxyProbe()

    /// Writes real attention inputs to disk, for characterising a sparse kernel
    /// against the distribution it will actually meet.
    ///
    /// **Random tensors are not a substitute, and the difference is not
    /// subtle.** Run against Gaussian q/k/v at this model's shape, Sol-Attn
    /// lands at rel_rms 0.17 even at beta = 0 — because unstructured attention
    /// has no dominant blocks to keep, so there is nothing for a sparse method
    /// to exploit. Any equivalence class measured that way describes the worst
    /// possible input rather than this model.
    ///
    /// Keyed on the **attention context**, not on a global call ordinal.
    ///
    /// An earlier version numbered captures by `step * numLayers + block`,
    /// derived from a counter incremented on every call. That only holds while
    /// the DiT block stack is the sole caller of `sdpa`, and it fails silently
    /// when it is not: the refiner's batched text path goes through the same
    /// function, so one extra call shifts every subsequent ordinal by one and
    /// the capture is then labelled with a block it did not come from. Nothing
    /// about the tensors would look wrong.
    ///
    /// `AttentionContext` already carries `blockIndex` and `scheduleProgress`,
    /// which is what the label actually wants, so it is read from there. The
    /// batched path passes no context and is therefore skipped for free.
    ///
    /// Driven by environment variables rather than a CLI flag, deliberately:
    /// this is a diagnostic for characterising a sparse backend, not a feature,
    /// and it should not add surface to `h3 render` or a dependency edge from
    /// the CLI target to this one.
    ///
    ///     H3_CAPTURE_QKV=/some/dir H3_CAPTURE_BLOCKS=0,24,49 H3_CAPTURE_AFTER=0.25
    package final class QKVCapture: @unchecked Sendable {
        package var directory: String?
        /// Block indices to capture, once each.
        package var wantedBlocks: Set<Int> = []
        /// Capture only at or after this point in the schedule. The early steps
        /// are the dense warm-up and are not what a sparse backend will meet.
        package var afterProgress: Double = 0.25
        package var captured: [Int: Int] = [:]
        /// How many times to capture each wanted block, at successive steps.
        ///
        /// More than one is what makes the *churn* question answerable: whether
        /// the router picks the same blocks at step k and step k+1. A selection
        /// that changes between adjacent steps changes the operator mid
        /// trajectory, which is a candidate explanation for the temporal
        /// artifact and cannot be tested from a single step's tensors.
        package var repeats: Int = 1
        package var written: [String] = []

        package init() {
            let env = ProcessInfo.processInfo.environment
            guard let dir = env["H3_CAPTURE_QKV"], !dir.isEmpty else { return }
            directory = dir
            wantedBlocks = Set((env["H3_CAPTURE_BLOCKS"] ?? "0,24,49")
                .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
            if let a = env["H3_CAPTURE_AFTER"], let d = Double(a) { afterProgress = d }
            if let r = env["H3_CAPTURE_REPEAT"], let n = Int(r) { repeats = max(1, n) }
        }
    }
    package static let qkvCapture = QKVCapture()

    static func maybeCapture(_ qh: MLXArray, _ kh: MLXArray, _ vh: MLXArray,
                             _ context: AttentionContext?) {
        guard let dir = qkvCapture.directory, let context else { return }
        let seen = qkvCapture.captured[context.blockIndex, default: 0]
        guard qkvCapture.wantedBlocks.contains(context.blockIndex),
              seen < qkvCapture.repeats,
              context.scheduleProgress >= qkvCapture.afterProgress else { return }
        qkvCapture.captured[context.blockIndex] = seen + 1

        // [1, heads, S, d] -> [S, heads, d], the layout the Triton kernel takes
        // as BTHD with the batch axis added back on the other side. Kept
        // identical to the previous format so anything that read the old
        // captures still reads these.
        func pack(_ x: MLXArray) -> MLXArray { x[0].transposed(1, 0, 2) }
        let path = "\(dir)/qkv_block\(String(format: "%02d", context.blockIndex))"
                 + "_p\(String(format: "%.2f", context.scheduleProgress)).safetensors"
        do {
            try MLX.save(arrays: ["q": pack(qh), "k": pack(kh), "v": pack(vh)],
                         url: URL(fileURLWithPath: path))
            qkvCapture.written.append(path)
            let note = "  captured block \(context.blockIndex) at progress "
                     + String(format: "%.2f", context.scheduleProgress)
                     + ", S=\(context.sequenceLength) -> \(path)\n"
            FileHandle.standardError.write(Data(note.utf8))
        } catch {
            let note = "  capture failed at block \(context.blockIndex): \(error)\n"
            FileHandle.standardError.write(Data(note.utf8))
        }
    }

    /// Per-query-block skewness, excess kurtosis, and the *measured* densities
    /// a threshold of `mu + beta*sigma` would keep at beta = 1 and 2 — against
    /// the Gaussian predictions of 15.87% and 2.28%.
    static func recordProxy(_ qh: MLXArray, _ kh: MLXArray) {
        let block = 64
        let s = qh.dim(2)
        let n = s / block
        guard n >= 8 else { return }
        let q = qh[0][0..., 0 ..< (n * block), 0...]
            .reshaped([qh.dim(1), n, block, qh.dim(3)]).mean(axis: 2).asType(.float32)
        let k = kh[0][0..., 0 ..< (n * block), 0...]
            .reshaped([kh.dim(1), n, block, kh.dim(3)]).mean(axis: 2).asType(.float32)
        let scores = matmul(q, k.transposed(0, 2, 1))
        let mu = scores.mean(axis: -1, keepDims: true)
        let d = scores - mu
        let sd = MLX.sqrt((d * d).mean(axis: -1, keepDims: true) + 1e-12)
        let z = d / sd
        let skew = Double((z * z * z).mean().item(Float.self))
        let kurt = Double((z * z * z * z).mean().item(Float.self)) - 3.0
        let t1 = Double((z .> MLXArray(Float(1.0))).asType(.float32).mean().item(Float.self))
        let t2 = Double((z .> MLXArray(Float(2.0))).asType(.float32).mean().item(Float.self))
        proxyProbe.samples.append((skew, kurt, t1, t2))
    }

    /// Scaled dot-product attention over `[S, heads, headDim]` or `[B, S, heads, headDim]` inputs, exposed
    /// so the op-level oracle can tap it on its own.
    ///
    /// **The backend dispatch lives here, inside the function the oracle taps,
    /// and not in the caller.** Putting it a level up would mean the oracle
    /// measures dense attention while production runs something else — the
    /// harness validating code production does not execute, which is the exact
    /// failure this arrangement exists to prevent. A backend that returns nil
    /// declines the call and the dense path below runs unchanged, so schedule
    /// windows and dense-block exclusions cost nothing to express.
    ///
    /// With `backend` and `context` omitted this is bit-for-bit what it was
    /// before the seam existed.
    package static func sdpa(q: MLXArray, k: MLXArray, v: MLXArray,
                            headDim: Int, fp32: Bool = false,
                            backend: (any H3AttentionBackend)? = nil,
                            context: AttentionContext? = nil) -> MLXArray {
        let at: DType = fp32 ? .float32 : q.dtype
        let hasBatch = q.ndim == 4
        let qh = hasBatch ? q.transposed(0, 2, 1, 3).asType(at) : q.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(at)
        let kh = hasBatch ? k.transposed(0, 2, 1, 3).asType(at) : k.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(at)
        let vh = hasBatch ? v.transposed(0, 2, 1, 3).asType(at) : v.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(at)
        
        if proxyProbe.enabled { Self.recordProxy(qh, kh) }
        if qkvCapture.directory != nil { Self.maybeCapture(qh, kh, vh, context) }

        let scale = 1.0 / Float(headDim).squareRoot()

        // The backend protocol has no batch axis because H3 has no batch axis;
        // the refiner's batched text path is therefore always dense, which costs
        // nothing worth measuring — it is a few hundred rows against 15,749.
        var o: MLXArray?
        if let backend, let context, !hasBatch {
            o = backend.attend(queries: qh[0], keys: kh[0], values: vh[0],
                               scale: scale, mask: nil, context: context)?
                .expandedDimensions(axis: 0)
        }
        if o == nil {
            o = MLXFast.scaledDotProductAttention(
                queries: qh, keys: kh, values: vh, scale: scale, mask: nil)
        }
        let out = o!

        if hasBatch {
            return out.transposed(0, 2, 1, 3).asType(q.dtype).reshaped([q.dim(0), q.dim(1), q.dim(2) * headDim])
        } else {
            return out.squeezed(axis: 0).transposed(1, 0, 2).asType(q.dtype).reshaped([q.dim(0), q.dim(1) * headDim])
        }
    }

    /// `x` is `[S, hidden]` or `[B, S, hidden]`.
    ///
    /// - Parameter context: where in the render this call sits. Nil means dense:
    ///   a sparse backend that does not know which block it is in, or how far
    ///   through the schedule, cannot honour its own dense warm-up or its
    ///   first/last-block exclusions, and guessing is worse than declining.
    package func callAsFunction(_ x: MLXArray, ropeTable: MLXArray?,
                               context: AttentionContext? = nil) -> MLXArray {
        let qkv = matmul(x, qkvWeight.T)
        let qkvParts = qkv.split(parts: 3, axis: -1)
        
        let targetShape = qkvParts[0].shape.dropLast() + [heads, headDim]
        var q = qkvParts[0].reshaped(targetShape)
        var k = qkvParts[1].reshaped(targetShape)
        let v = qkvParts[2].reshaped(targetShape)

        // RMSNorm is applied per head BEFORE rope, as the fused kernel does.
        q = qNorm(q)
        k = kNorm(k)
        if let ropeTable {
            q = SplitHalfRoPE.apply(q, table: ropeTable)
            k = SplitHalfRoPE.apply(k, table: ropeTable)
        }

        // Same call the oracle taps. Inlining a second copy here would mean the
        // oracle validates code production does not run, which is the exact
        // failure this harness exists to prevent.
        let merged = Self.sdpa(q: q, k: k, v: v, headDim: headDim, fp32: fp32Attention,
                               backend: backend, context: context)
        return matmul(merged, outWeight.T)
    }
}

/// `fc2(silu(gate) * up)` where `fc1` emits `2 * ffn` and
/// `gate, up = chunk(2, dim: -1)` — **gate is the FIRST half**
/// (`_swiglu_eager` in `comfy/ops.py`). Swapping them is a coin flip a port
/// loses half the time, and the output stays plausible.
package struct H3MLP {
    package let fc1: MLXArray   // [2 * ffn, hidden]
    package let fc2: MLXArray   // [hidden, ffn]

    package init(fc1: MLXArray, fc2: MLXArray) {
        self.fc1 = fc1
        self.fc2 = fc2
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = matmul(x, fc1.T)
        let parts = h.split(parts: 2, axis: -1)
        let gate = parts[0]
        let up = parts[1]
        return matmul(silu(gate) * up, fc2.T)
    }
}

/// One transformer block.
///
///     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln(t_emb)
///     h = modScaleShift(norm1(x), shift_msa, scale_msa)
///     x = modGate(x, gate_msa, attn(h))
///     h = modScaleShift(norm2(x), shift_mlp, scale_mlp)
///     x = modGate(x, gate_mlp, mlp(h))
///
/// The reference mutates `x` in place and returns the same object. We return a
/// new array — functionally identical, and MLX has no in-place residual to
/// preserve.
package struct DiTBlock {
    package let norm1: H3RMSNorm
    package let norm2: H3RMSNorm
    package let attn: AttentionLayer
    package let mlp: H3MLP
    package let adaln: AdalnProj

    package init(norm1: H3RMSNorm, norm2: H3RMSNorm, attn: AttentionLayer,
                mlp: H3MLP, adaln: AdalnProj) {
        self.norm1 = norm1
        self.norm2 = norm2
        self.attn = attn
        self.mlp = mlp
        self.adaln = adaln
    }

    package func callAsFunction(_ x: MLXArray, tEmb: MLXArray, index: ModulationIndex,
                               ropeTable: MLXArray?,
                               context: AttentionContext? = nil) -> MLXArray {
        let m = adaln(tEmb)
        precondition(m.count == 6, "DiTBlock AdaLN must expand to 6, got \(m.count)")

        // Each of these four is a fused kernel with the readable expression as
        // its fallback, and the fallback is not decorative: `FusedModulation`
        // returns nil for the refiner's batched shape, for any dtype mix it was
        // not measured on, and whenever `H3_FUSED_MODULATION=0`. The two paths
        // are held to bit-identity by `FusedModulationTests`, so which one ran
        // is a performance question and never a numerical one.
        func norm(_ v: MLXArray, _ n: H3RMSNorm, _ shift: MLXArray,
                  _ scale: MLXArray) -> MLXArray {
            FusedModulation.modulatedRMSNorm(v, weight: n.weight, eps: n.eps,
                                             shift: shift, scale: scale, index: index)
                ?? modScaleShift(n(v), shift: shift, scale: scale, index: index)
        }
        func gated(_ v: MLXArray, _ gate: MLXArray, _ other: MLXArray) -> MLXArray {
            FusedModulation.gatedResidual(v, gate: gate, other: other, index: index)
                ?? modGate(v, gate: gate, other: other, index: index)
        }

        let h1 = norm(x, norm1, m[0], m[1])
        let x1 = gated(x, m[2], attn(h1, ropeTable: ropeTable, context: context))
        let h2 = norm(x1, norm2, m[3], m[4])
        return gated(x1, m[5], mlp(h2))
    }
}
