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

    /// Q, K, V after the projection, per-head RMSNorm and RoPE. GPU attention
    /// consumes this; the output projection is a separate call so a second
    /// CFG branch can occupy the engine while this one attends.
    /// Submits `qkv` to the engine and returns without waiting for it.
    ///
    /// Everything after the projection — the split, the per-head norms, RoPE —
    /// needs the numbers, so it lives in ``finishQKV``. The gap between the two
    /// is where the other CFG branch gets the GPU.
    /// `engine: false` keeps the projection on the GPU even with the backend
    /// on. The MLP island owns both dies for the whole block, and a `qkv`
    /// queued beside it is a third and fourth concurrent evaluation on a
    /// private runtime that hard-locked the machine under exactly that load.
    package func beginQKV(_ x: MLXArray, engine: Bool = true)
        -> ANELinearBackend.Pending {
        DiTBlock.saturationProbe.record("qkv", x: x, weight: qkvWeight)
        return ANELinearBackend.begin(x: x, weight: qkvWeight, label: "qkv",
                                      engine: engine)
    }

    package func finishQKV(_ pending: ANELinearBackend.Pending, ropeTable: MLXArray?)
        -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        shapeQKV(pending.value(), ropeTable: ropeTable)
    }

    package func projectQKV(_ x: MLXArray, ropeTable: MLXArray?)
        -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        DiTBlock.saturationProbe.record("qkv", x: x, weight: qkvWeight)
        let qkv = ANELinearBackend.isEnabled
            ? ANELinearBackend.project(x: x, weight: qkvWeight, label: "qkv")
            : matmul(x, qkvWeight.T)
        return shapeQKV(qkv, ropeTable: ropeTable)
    }

    private func shapeQKV(_ qkv: MLXArray, ropeTable: MLXArray?)
        -> (q: MLXArray, k: MLXArray, v: MLXArray) {
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
        return (q, k, v)
    }

    package func attend(q: MLXArray, k: MLXArray, v: MLXArray,
                        context: AttentionContext?) -> MLXArray {
        // Same call the oracle taps. Inlining a second copy here would mean the
        // oracle validates code production does not run, which is the exact
        // failure this harness exists to prevent.
        Self.sdpa(q: q, k: k, v: v, headDim: headDim, fp32: fp32Attention,
                  backend: backend, context: context)
    }

    /// `attn out` contracts over `inner` rather than `hidden`, so it builds
    /// its own compiled program. Its interior partials peak at 3,925 at
    /// block 49 — 8.3x under the 2^15 cliff unscaled, 134x with the operand
    /// scale this path applies — so it needs no bound beyond what is here.
    package func projectOut(_ merged: MLXArray) -> MLXArray {
        DiTBlock.saturationProbe.record("attn out", x: merged, weight: outWeight)
        return ANELinearBackend.isEnabled
            ? ANELinearBackend.project(x: merged, weight: outWeight, label: "attn out")
            : matmul(merged, outWeight.T)
    }

    package func beginOut(_ merged: MLXArray, engine: Bool = true)
        -> ANELinearBackend.Pending {
        DiTBlock.saturationProbe.record("attn out", x: merged, weight: outWeight)
        return ANELinearBackend.begin(x: merged, weight: outWeight, label: "attn out",
                                      engine: engine)
    }

    /// `x` is `[S, hidden]` or `[B, S, hidden]`.
    ///
    /// - Parameter context: where in the render this call sits. Nil means dense:
    ///   a sparse backend that does not know which block it is in, or how far
    ///   through the schedule, cannot honour its own dense warm-up or its
    ///   first/last-block exclusions, and guessing is worse than declining.
    package func callAsFunction(_ x: MLXArray, ropeTable: MLXArray?,
                               context: AttentionContext? = nil) -> MLXArray {
        let p = projectQKV(x, ropeTable: ropeTable)
        return projectOut(attend(q: p.q, k: p.k, v: p.v, context: context))
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

    /// Submits `fc1` and returns; `finish` completes the gate, the product and
    /// `fc2`. `fc2` is GPU work either way, so it is not worth splitting.
    package func begin(_ x: MLXArray) -> ANELinearBackend.Pending {
        DiTBlock.saturationProbe.record("fc1", x: x, weight: fc1)
        return ANELinearBackend.begin(x: x, weight: fc1, label: "fc1")
    }

    /// Submits the whole MLP to the island and returns without waiting.
    ///
    /// Nil means the island declined before submitting anything — the wrong
    /// shape, no block index, the backend off — and the caller takes its
    /// established path. A returned value always settles to a complete MLP.
    package func beginIsland(_ x: MLXArray, blockIndex: Int?)
        -> ANEMLPIslandBackend.Pending? {
        guard let blockIndex else { return nil }
        return ANEMLPIslandBackend.beginRoute(x: x, fc1: fc1, fc2: fc2,
                                              blockIndex: blockIndex)
    }

    package func finish(_ pending: ANELinearBackend.Pending) -> MLXArray {
        let gated = pending.swiGLUValue()
        DiTBlock.saturationProbe.recordMLPIslandFC2(x: gated, weight: fc2)
        return matmul(gated, fc2.T)
    }

    private func gateAndProject(_ h: MLXArray) -> MLXArray {
        let parts = h.split(parts: 2, axis: -1)
        let gated = silu(parts[0]) * parts[1]
        // `fc2` is the projection the bound exists to rule on, and this is the
        // only place its real input exists.
        DiTBlock.saturationProbe.record("fc2", x: gated, weight: fc2)
        DiTBlock.saturationProbe.recordMLPIslandFC2(x: gated, weight: fc2)
        return matmul(gated, fc2.T)
    }

    package func callAsFunction(_ x: MLXArray, blockIndex: Int? = nil) -> MLXArray {
        // `fc1` is the largest GEMM in a block — 262 ms against qkv's 197 at
        // production width — and its interior partials peak at 72 against the
        // engine's 2^15 cliff, so it is the cheapest projection to route and
        // the one with the most to give.
        //
        // `fc2` is deliberately not routed. It breaches saturation at block 49
        // (34,649) and the failure is silent zeros, so it waits on a measured
        // per-block bound rather than on an operand scale that is merely
        // probably enough.
        if let blockIndex,
           let island = ANEMLPIslandBackend.project(
               x: x, fc1: fc1, fc2: fc2, blockIndex: blockIndex
           ) {
            return island
        }
        if ANELinearBackend.isEnabled {
            return finish(begin(x))
        }
        // **Also here, not only in `begin`.** A bound run has the engine off by
        // design — the activations must come from the unrouted path — so
        // instrumenting only the routed entry point measures `fc1` never.
        DiTBlock.saturationProbe.record("fc1", x: x, weight: fc1)
        return gateAndProject(matmul(x, fc1.T))
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

/// The order-free saturation bound, computed **during a render** instead of
/// from captured tensors.
///
/// `Tools/ANE/saturation_bound.py` answers the same question offline, and it is
/// the better instrument when the captures exist: it reads the reference's own
/// taps. But the captures cover three blocks of fifty and 6.6% of sequence
/// positions, and `fc2` was refused on exactly that — its bound moves 95x
/// across the three blocks that were measured, so twenty-four unmeasured blocks
/// is not a gap that argument survives.
///
/// Nothing has to be captured to close it. What the bound needs is
/// `max over (row, channel) of sum_k |a_k| |w_k|`, which is one GEMM on the
/// magnitudes, and every activation it wants is in hand at the moment the
/// projection runs. Computing it inline covers **every block and every row of
/// every step**, which is strictly more than the captures ever offered.
///
/// Two things this is not. It is not free — the magnitude GEMM is the same
/// shape as the projection, so a bound run costs about double. And the
/// activations are *ours*, from the unrouted bf16 path, not the reference's;
/// that is the arithmetic the conformance suite pins the reference against, and
/// it is stated here rather than glossed because a bound is only as good as the
/// data under it.
///
///     H3_ANE_BOUND=/tmp/bound.json h3 render ...   (run without H3_ANE)
///
/// Add `H3_ANE_BOUND_MLP_ISLAND=1` to also measure the two candidate ANE
/// `fc2` neuron ranges. It is separate because those bounds add two more
/// shard-sized magnitude GEMMs per block.
/// `H3_ANE_BOUND_MLP_ISLAND_ONLY=1` suppresses the older whole-projection
/// measurements during calibration; it implies the island measurement and
/// cuts work that cannot affect the per-block scale table.
package final class SaturationProbe: @unchecked Sendable {
    package let path: String?
    /// Optional, one-shot capture of the real SwiGLU activation consumed by
    /// fc2. This closes the precision half of the scale calibration without
    /// requiring the CUDA reference environment on the render machine.
    private let captureDirectory: String?
    private let captureBlocks: Set<Int>
    private let captureAfter: Double
    private let captureRows: Int
    private var capturedBlocks: Set<Int> = []
    /// Opt-in because the candidate island adds two shard-sized magnitude
    /// GEMMs to every fc2 in an already expensive faithful bound render.
    package let mlpIslandEnabled: Bool
    package let mlpIslandOnly: Bool
    private let mlpIslandNeuronsPerDie: Int?
    /// Set by the block loop, which is the only place that knows where it is.
    package var block = -1
    package var progress = 0.0
    private let lock = NSLock()
    /// Worst bound seen per projection, and where it was seen.
    private var worst: [String: (bound: Double, block: Int, progress: Double,
                                 operandScale: Double, lower: Int, upper: Int)] = [:]

    package init() {
        let env = ProcessInfo.processInfo.environment
        path = (env["H3_ANE_BOUND"]?.isEmpty == false) ? env["H3_ANE_BOUND"] : nil
        captureDirectory = (env["H3_CAPTURE_MLP_ISLAND"]?.isEmpty == false)
            ? env["H3_CAPTURE_MLP_ISLAND"] : nil
        captureBlocks = Set((env["H3_CAPTURE_MLP_BLOCKS"] ?? "39")
            .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
        captureAfter = Double(env["H3_CAPTURE_MLP_AFTER"] ?? "0.8") ?? 0.8
        captureRows = max(1, Int(env["H3_CAPTURE_MLP_ROWS"] ?? "1024") ?? 1024)
        mlpIslandOnly = env["H3_ANE_BOUND_MLP_ISLAND_ONLY"] == "1"
        mlpIslandEnabled = mlpIslandOnly || env["H3_ANE_BOUND_MLP_ISLAND"] == "1"
        mlpIslandNeuronsPerDie = env["H3_ANE_MLP_NEURONS_PER_DIE"].flatMap(Int.init)
    }

    package var enabled: Bool { path != nil }
    package var needsContext: Bool { enabled || captureDirectory != nil }

    /// Pieces the contraction is cut into for the split bound, alongside the
    /// whole one. Both are reported: the whole-`k` figure is what the current
    /// unsplit path must clear, and the per-piece figure is what a split path
    /// must clear, because a split projection never accumulates across a piece.
    package var splits = 8

    /// `x` is `[s, k]`, `weight` is `[n, k]` — the projection's own operands.
    package func record(_ label: String, x: MLXArray, weight: MLXArray) {
        guard enabled, !mlpIslandOnly, block >= 0 else { return }
        measure(label, x: x, weight: weight, pieces: 1)
        let k = x.dim(1)
        if splits > 1, k % splits == 0 {
            measure(label + " split\(splits)", x: x, weight: weight, pieces: splits)
        }
    }

    /// Bounds the first persistent-MLP candidate: GPU owns the first half of
    /// the SwiGLU neurons and each ANE consumes one quarter. At H3 width an ANE
    /// quarter is 3,584 neurons; four pieces keep every private accumulation at
    /// the measured 896-wide throughput and safety sweet spot.
    package func recordMLPIslandFC2(x: MLXArray, weight: MLXArray) {
        maybeCaptureMLPIsland(x)
        guard enabled, mlpIslandEnabled, block >= 0 else { return }
        let k = x.dim(1)
        let perDie = mlpIslandNeuronsPerDie ?? (k / 4)
        guard perDie > 0, perDie % 4 == 0, 2 * perDie <= k else { return }
        let gpu = k - 2 * perDie
        let prefix = String(format: "fc2 island b%02d", block)
        measure(prefix + " ane0 split4", x: x, weight: weight, pieces: 4,
                contraction: gpu ..< (gpu + perDie), operandScale: 1.0 / 256.0)
        measure(prefix + " ane1 split4", x: x, weight: weight, pieces: 4,
                contraction: (gpu + perDie) ..< k, operandScale: 1.0 / 256.0)
    }

    /// Save evenly spaced rows rather than a prefix: packed H3 sequences are
    /// modality ordered, so a prefix silently over-samples text/audio and can
    /// miss the video rows that dominate a production render. The capture is
    /// fp32 because `Tools/ANE/underflow.py` consumes the reference-oracle
    /// format and compares arithmetic rather than storage rounding.
    private func maybeCaptureMLPIsland(_ x: MLXArray) {
        guard let directory = captureDirectory, block >= 0,
              captureBlocks.contains(block), progress >= captureAfter else { return }
        lock.lock()
        let shouldCapture = !capturedBlocks.contains(block)
        if shouldCapture { capturedBlocks.insert(block) }
        lock.unlock()
        guard shouldCapture else { return }

        let count = min(captureRows, x.dim(0))
        let indices: [Int32] = (0 ..< count).map {
            count == 1 ? 0 : Int32(($0 * (x.dim(0) - 1)) / (count - 1))
        }
        let rows = MLXArray(indices)
        let sample = x.take(rows, axis: 0).asType(.float32)
        let name = "mlp_block\(String(format: "%02d", block))"
                 + "_p\(String(format: "%.2f", progress)).safetensors"
        let path = URL(fileURLWithPath: directory).appendingPathComponent(name)
        do {
            try FileManager.default.createDirectory(at: URL(fileURLWithPath: directory),
                                                    withIntermediateDirectories: true)
            try MLX.save(arrays: ["ref.mlp.swiglu": sample, "in.rows": rows], url: path)
            let note = "  captured fc2 input block \(block) at progress "
                     + String(format: "%.2f", progress)
                     + ", \(count)/\(x.dim(0)) rows -> \(path.path)\n"
            FileHandle.standardError.write(Data(note.utf8))
        } catch {
            lock.lock(); capturedBlocks.remove(block); lock.unlock()
            let note = "  fc2 input capture failed at block \(block): \(error)\n"
            FileHandle.standardError.write(Data(note.utf8))
        }
    }

    private func measure(_ label: String, x: MLXArray, weight: MLXArray, pieces: Int,
                         contraction: Range<Int>? = nil,
                         operandScale: Double = 0.0625) {
        // fp32 throughout: this is a safety bound, and bf16's three digits are
        // not enough to argue a factor-of-two margin with.
        let span = contraction ?? (0 ..< x.dim(1))
        guard span.count > 0, span.count % pieces == 0 else { return }
        let piece = span.count / pieces
        var pieceMax: [MLXArray] = []
        for lo in stride(from: span.lowerBound, to: span.upperBound, by: piece) {
            let hi = min(lo + piece, span.upperBound)
            let a = MLX.abs(x[0..., lo ..< hi].asType(.float32))
            // Chunked over output channels so `[15731, 28672]` never exists
            // whole, but reduced lazily and read **once**: an `.item()` per
            // chunk is a GPU sync per chunk, seven per `fc1`, fifty blocks a step.
            let n = weight.dim(0), step = 4096
            var chunkPeaks: [MLXArray] = []
            for start in stride(from: 0, to: n, by: step) {
                let w = MLX.abs(
                    weight[start ..< min(start + step, n), lo ..< hi].asType(.float32)
                )
                chunkPeaks.append(MLX.matmul(a, w.transposed()).max())
            }
            pieceMax.append(MLX.stacked(chunkPeaks).max())
        }
        let peak = Double(MLX.stacked(pieceMax).max().item(Float.self))
        lock.lock()
        if peak > (worst[label]?.bound ?? 0) {
            worst[label] = (peak, block, progress, operandScale,
                            span.lowerBound, span.upperBound)
        }
        lock.unlock()
        write()
    }

    /// Rewritten on every improvement rather than at exit: a bound run is long
    /// and the answer should survive it being interrupted.
    private func write() {
        guard let path else { return }
        lock.lock(); let snapshot = worst; lock.unlock()
        // The persistent MLP island deliberately uses 1/256 for fc2. A global
        // 1/16 threshold would report a false failure for arithmetic the
        // candidate never executes, so every observation carries its scale.
        let defaultThreshold = 32768.0 / 0.0625
        var rows: [String] = []
        for (label, v) in snapshot.sorted(by: { $0.key < $1.key }) {
            let threshold = 32768.0 / v.operandScale
            var requiredScale = 1.0
            // Factor-of-two margin, powers of two only. This is the largest
            // candidate scale the block may use without changing fp16 values
            // through an inexact multiplier.
            while v.bound * requiredScale >= 16384.0 && requiredScale > 1.0 / 65536.0 {
                requiredScale /= 2.0
            }
            rows.append("""
                  "\(label)": { "bound": \(v.bound), "block": \(v.block),             "progress": \(v.progress), "operandScale": \(v.operandScale),             "contraction": [\(v.lower), \(v.upper)], "requiredScale2x": \(requiredScale),             "threshold": \(threshold), "headroom": \(threshold / v.bound),             "proven": \(v.bound < threshold) }
            """)
        }
        let json = "{\n  \"threshold\": \(defaultThreshold),\n\(rows.joined(separator: ",\n"))\n}\n"
        try? json.write(toFile: path, atomically: true, encoding: .utf8)
    }
}

package struct DiTBlock {
    package static let saturationProbe = SaturationProbe()

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

    /// Residual, AdaLN tables, and Q/K/V ready for GPU attention.
    ///
    /// Split from ``callAsFunction`` so classifier-free guidance can run one
    /// branch's attention on the GPU while the other branch occupies the
    /// Neural Engine with its QKV projection. The three stages compose to the
    /// original block; they exist to be scheduled, not to change the math.
    package struct AttentionPrep {
        package let x: MLXArray
        package let m: [MLXArray]
        package let q: MLXArray
        package let k: MLXArray
        package let v: MLXArray
    }

    package func prepareAttention(_ x: MLXArray, tEmb: MLXArray, index: ModulationIndex,
                                 ropeTable: MLXArray?) -> AttentionPrep {
        let m = adaln(tEmb)
        precondition(m.count == 6, "DiTBlock AdaLN must expand to 6, got \(m.count)")
        let h1 = modScaleShift(norm1(x), shift: m[0], scale: m[1], index: index)
        let p = attn.projectQKV(h1, ropeTable: ropeTable)
        return AttentionPrep(x: x, m: m, q: p.q, k: p.k, v: p.v)
    }

    /// A block whose `qkv` is on the engine and not yet collected.
    package struct AttentionStart {
        package let x: MLXArray
        package let m: [MLXArray]
        package let qkv: ANELinearBackend.Pending
    }

    /// AdaLN, the pre-norm, and `qkv` submitted — but not waited for.
    ///
    /// ``prepareAttention`` is this plus ``finishAttention`` back to back. They
    /// are separate so a caller with two independent CFG branches can submit
    /// both projections before collecting either, which is the only way the
    /// engine has anything queued while the GPU attends.
    package func beginAttention(_ x: MLXArray, tEmb: MLXArray,
                                index: ModulationIndex,
                                engine: Bool = true) -> AttentionStart {
        let m = adaln(tEmb)
        precondition(m.count == 6, "DiTBlock AdaLN must expand to 6, got \(m.count)")
        let h1 = modScaleShift(norm1(x), shift: m[0], scale: m[1], index: index)
        return AttentionStart(x: x, m: m, qkv: attn.beginQKV(h1, engine: engine))
    }

    package func finishAttention(_ start: AttentionStart,
                                 ropeTable: MLXArray?) -> AttentionPrep {
        let p = attn.finishQKV(start.qkv, ropeTable: ropeTable)
        return AttentionPrep(x: start.x, m: start.m, q: p.q, k: p.k, v: p.v)
    }

    /// The output projection submitted, with the residual it will feed.
    package struct PostStart {
        package let prep: AttentionPrep
        package let out: ANELinearBackend.Pending
    }

    package func beginPost(_ prep: AttentionPrep, merged: MLXArray,
                           engine: Bool = true) -> PostStart {
        PostStart(prep: prep, out: attn.beginOut(merged, engine: engine))
    }

    /// The first residual and gate, then the MLP submitted.
    ///
    /// Which submission it is depends on what accepted the work: the island
    /// takes the whole MLP across both dies and the GPU, `fc1` alone goes to
    /// the plain backend. Either way nothing has been waited for yet, which is
    /// the property the schedule is built on.
    package struct MLPStart {
        package let x1: MLXArray
        package let m: [MLXArray]
        let work: Work

        enum Work {
            case linear(ANELinearBackend.Pending)
            case island(ANEMLPIslandBackend.Pending)
        }
    }

    /// - Parameter blockIndex: which block this is, for the island's per-block
    ///   `fc2` activation scale. Nil declines the island: its scales are
    ///   calibrated per block and per die, and guessing one is a silent
    ///   saturation, not a rounding error.
    /// - Parameter island: whether to offer the MLP to the island at all.
    ///   Defaults to the environment opt-in; a benchmark passes it explicitly
    ///   so both schedules can be measured interleaved in one process.
    package func beginMLP(_ post: PostStart, index: ModulationIndex,
                          blockIndex: Int? = nil,
                          island: Bool = ANEMLPIslandBackend.isEnabled) -> MLPStart {
        let m = post.prep.m
        let x1 = modGate(post.prep.x, gate: m[2], other: post.out.value(), index: index)
        let h2 = modScaleShift(norm2(x1), shift: m[3], scale: m[4], index: index)
        if island, let island = mlp.beginIsland(h2, blockIndex: blockIndex) {
            return MLPStart(x1: x1, m: m, work: .island(island))
        }
        return MLPStart(x1: x1, m: m, work: .linear(mlp.begin(h2)))
    }

    package func finishBlock(_ start: MLPStart, index: ModulationIndex) -> MLXArray {
        let out: MLXArray
        switch start.work {
        case .linear(let pending): out = mlp.finish(pending)
        case .island(let pending): out = pending.value()
        }
        return modGate(start.x1, gate: start.m[5], other: out, index: index)
    }

    package func attend(_ prep: AttentionPrep, context: AttentionContext?) -> MLXArray {
        attn.attend(q: prep.q, k: prep.k, v: prep.v, context: context)
    }

    package func postAttention(_ prep: AttentionPrep, merged: MLXArray,
                              index: ModulationIndex, blockIndex: Int? = nil) -> MLXArray {
        let x1 = modGate(prep.x, gate: prep.m[2], other: attn.projectOut(merged), index: index)
        let h2 = modScaleShift(norm2(x1), shift: prep.m[3], scale: prep.m[4], index: index)
        return modGate(x1, gate: prep.m[5], other: mlp(h2, blockIndex: blockIndex), index: index)
    }

    package func callAsFunction(_ x: MLXArray, tEmb: MLXArray, index: ModulationIndex,
                               ropeTable: MLXArray?,
                               context: AttentionContext? = nil) -> MLXArray {
        let prep = prepareAttention(x, tEmb: tEmb, index: index, ropeTable: ropeTable)
        return postAttention(prep, merged: attend(prep, context: context), index: index,
                             blockIndex: context?.blockIndex)
    }
}
