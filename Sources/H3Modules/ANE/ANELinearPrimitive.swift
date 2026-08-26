// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXFast
import H3ANEBridge

/// Runs part of a linear projection on the Neural Engine while MLX runs the
/// rest on the GPU, so the two add up instead of taking turns.
///
/// This is augmentation, not substitution. The engine is slower than the GPU
/// — about 3.8 TFLOP/s a die against 16 — so routing a projection *to* it
/// loses. Routing a slice of the output channels to it while the GPU works on
/// the remainder wins, provided the slice is sized so both finish together and
/// the two dies actually run at the same time.
///
/// Three things had to be measured before any of that was safe, and all three
/// are in `docs/`:
///
///  * **Precision.** fp16 with the engine's wide accumulator is *more* accurate
///    than the bf16 GPU path on real captured tensors, 7e-5 to 5e-4 against
///    1.66e-3 (`ANE_PRECISION_RESULTS`).
///  * **Saturation.** A running partial that reaches 2^15 makes the engine
///    return zero, silently. qkv's worst measured partial is 485, so it has
///    67x headroom unscaled — but a power-of-two operand scale is exact in
///    fp16 and costs nothing, so it is applied anyway and the projections that
///    do breach (fc2 at block 49) stay off this path until bounded.
///  * **Placement.** `kANEFAneInstanceHint` does not choose a die
///    (`ANE_REVERSE_ENGINEERING`). Two dies run only when two evaluations are in
///    flight at once, which is why the shards go through `h3_ane_run_pair`.
///
/// Opt in with `H3_ANE=experimental`. Everything falls back to a plain `matmul`
/// when the engine is unavailable, the shape is one the engine refuses, or any
/// step fails — so the failure mode is "no speedup", not "wrong numbers".
public enum ANELinearBackend {

    /// Read once. This used to be a computed property, which rebuilt the whole
    /// environment dictionary on every attention call — a thousand times a
    /// render — and let the routing change underneath a run in progress.
    public static let isEnabled: Bool = {
        guard let env = ProcessInfo.processInfo.environment["H3_ANE"],
              env.lowercased() == "experimental"
        else { return false }
        return h3_ane_is_available()
    }()

    /// Operand scale. Exact in fp16 because it is a power of two, so it moves
    /// the partial-sum envelope without touching the arithmetic.
    private static let operandScale: Float = 0.0625      // 1/16

    /// Share of output channels routed to the engine.
    ///
    /// Both sides should finish together, and the balance point was swept
    /// rather than derived — the GPU sustains ~18.6 TFLOP/s on this shape, not
    /// the 16 the roadmap quotes, so a share computed from 16 gives the engine
    /// too much and leaves the GPU idle waiting for it. At S=15744, K=5376,
    /// N=21504 against a 196.7 ms plain matmul:
    ///
    ///     per die   GPU cols   GPU ms   ANE ms   combined   speedup
    ///        2560      16384    144.9    112.4      144.3    1.363x
    ///        2816      15872    139.0    124.6      138.3    1.422x
    ///     >> 3072      15360    135.3    134.6      134.8    1.459x
    ///        3200      15104    132.8    141.2      140.5    1.400x
    ///        3456      14592    128.5    151.4      150.9    1.303x
    ///        3712      14080    123.5    163.9      163.2    1.205x
    ///
    /// The curve is sharp on the far side: every column past the balance point
    /// costs more on the engine than it saves on the GPU. 0.286 lands on 3072.
    private static let aneShare = 0.286

    // MARK: - Shard plan

    /// Which output channels go where. Boundaries land on `headDim` so a shard
    /// never splits a head, and on 32 elements so every IOSurface row is a
    /// whole 64-byte stride.
    struct ShardPlan {
        let gpu: Range<Int>
        let ane0: Range<Int>
        let ane1: Range<Int>
        var perDie: Int { ane0.count }
    }

    static func plan(n: Int, headDim: Int = 128) -> ShardPlan? {
        guard n >= 8 * headDim, n % headDim == 0 else { return nil }
        let perDie = Int((Double(n) * aneShare / 2 / Double(headDim)).rounded()) * headDim
        guard perDie > 0, 2 * perDie < n else { return nil }
        let gpuEnd = n - 2 * perDie
        return ShardPlan(gpu: 0 ..< gpuEnd,
                         ane0: gpuEnd ..< (gpuEnd + perDie),
                         ane1: (gpuEnd + perDie) ..< n)
    }

    // MARK: - Compiled programs and their activation/output surfaces

    /// One compiled program per die plus the surfaces they read and write.
    ///
    /// A program is fixed at one `(s, k, n)` and compiling costs real time, so
    /// these are cached for the life of the process. Sequence length is
    /// constant across a render, so in practice exactly one session is built.
    ///
    /// Both dies read the *same* activation surface — the shards differ only in
    /// which output channels they produce — so the activation is uploaded once
    /// per call rather than once per die.
    /// Sequence lengths the engine runs at full rate.
    ///
    /// Throughput depends on the activation's minor extent: 3.87 TFLOP/s a die
    /// at s=14336 and s=16384, but 2.45 at s=15731 — the production length,
    /// which is prime. Rounding up to a multiple of 64 and computing a few
    /// surplus columns is worth 1.217x against 0.789x unpadded.
    static func paddedSequence(_ s: Int) -> Int { (s + 63) & ~63 }

    private final class Session {
        /// The compiled length, which is `paddedSequence(requested)`.
        let s: Int, k: Int, perDie: Int
        let program0: OpaquePointer, program1: OpaquePointer
        let x: OpaquePointer
        let y0: OpaquePointer, y1: OpaquePointer
        /// The surfaces are mutable. Calls sharing a compiled shape must own
        /// them from activation upload through the MLX copy-out.
        let executionLock = NSLock()

        init?(s requested: Int, k: Int, perDie: Int) {
            let s = ANELinearBackend.paddedSequence(requested)
            guard let p0 = h3_ane_program_create(Int32(s), Int32(k), Int32(perDie)) else { return nil }
            guard let p1 = h3_ane_program_create(Int32(s), Int32(k), Int32(perDie)) else {
                h3_ane_program_free(p0); return nil
            }
            // The engine reads the activation with sequence as the minor axis,
            // so this surface is [k, s], not [s, k].
            guard let xt = h3_ane_tensor_create(Int32(k), Int32(s)) else {
                h3_ane_program_free(p0); h3_ane_program_free(p1); return nil
            }
            guard let y0t = h3_ane_tensor_create(Int32(s), Int32(perDie)) else {
                h3_ane_program_free(p0); h3_ane_program_free(p1); h3_ane_tensor_free(xt); return nil
            }
            guard let y1t = h3_ane_tensor_create(Int32(s), Int32(perDie)) else {
                h3_ane_program_free(p0); h3_ane_program_free(p1)
                h3_ane_tensor_free(xt); h3_ane_tensor_free(y0t); return nil
            }
            // Outputs are aliased into MLX, so padding is not representable.
            guard h3_ane_tensor_is_dense(y0t), h3_ane_tensor_is_dense(y1t) else {
                h3_ane_program_free(p0); h3_ane_program_free(p1)
                h3_ane_tensor_free(xt); h3_ane_tensor_free(y0t); h3_ane_tensor_free(y1t)
                return nil
            }
            self.s = s; self.k = k; self.perDie = perDie
            self.program0 = p0; self.program1 = p1
            self.x = xt; self.y0 = y0t; self.y1 = y1t
        }

        deinit {
            h3_ane_program_free(program0); h3_ane_program_free(program1)
            h3_ane_tensor_free(x); h3_ane_tensor_free(y0); h3_ane_tensor_free(y1)
        }
    }

    /// The two weight surfaces for one projection, uploaded once.
    ///
    /// Weights do not change during a render. The first version of this
    /// converted and copied both shards on every call — 66 GB of memcpy per
    /// render for bytes that were already there.
    private final class WeightSet {
        let w0: OpaquePointer, w1: OpaquePointer
        weak var source: MLXArray?
        let shape: [Int]
        init(w0: OpaquePointer, w1: OpaquePointer, source: MLXArray) {
            self.w0 = w0; self.w1 = w1; self.source = source; self.shape = source.shape
        }
        deinit { h3_ane_tensor_free(w0); h3_ane_tensor_free(w1) }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var sessions: [String: Session] = [:]
    nonisolated(unsafe) private static var weights: [ObjectIdentifier: WeightSet] = [:]
    /// Shapes the engine has already refused, so a fallback costs one dictionary
    /// lookup rather than a failed compile per call.
    nonisolated(unsafe) private static var refused: Set<String> = []

    private static func session(s: Int, k: Int, perDie: Int) -> Session? {
        let key = "\(paddedSequence(s))x\(k)x\(perDie)"
        lock.lock(); defer { lock.unlock() }
        if let cached = sessions[key] { return cached }
        if refused.contains(key) { return nil }
        guard let made = Session(s: s, k: k, perDie: perDie) else {
            refused.insert(key)
            return nil
        }
        sessions[key] = made
        return made
    }

    /// Uploads both weight shards once, keyed by the identity of the weight
    /// array. `MLXArray` is a reference type and a layer holds the same one for
    /// the life of the process, so this is stable across steps and distinct
    /// across blocks.
    private static func weightSet(for qkvWeight: MLXArray, plan: ShardPlan,
                                  k: Int) -> WeightSet? {
        let key = ObjectIdentifier(qkvWeight)
        lock.lock()
        // Drop entries whose model has gone away, both to bound retained ANE
        // surfaces and to prevent ObjectIdentifier reuse selecting old weights.
        weights = weights.filter { $0.value.source != nil }
        if let cached = weights[key], cached.source === qkvWeight,
           cached.shape == qkvWeight.shape {
            lock.unlock(); return cached
        }
        weights.removeValue(forKey: key)
        lock.unlock()

        // The program contracts over the last axis of the activation and the
        // first of the weight, so the engine wants `[k, n]` while the
        // checkpoint holds `[n, k]`. The transpose happens here, once, which
        // is the entire reason weights have their own lifetime: per call it
        // would cost more than the projection.
        guard let w0 = h3_ane_tensor_create(Int32(k), Int32(plan.perDie)),
              let w1 = h3_ane_tensor_create(Int32(k), Int32(plan.perDie))
        else { return nil }

        func upload(_ rows: Range<Int>, into tensor: OpaquePointer) -> Bool {
            // Contiguous for the same reason as the activation: this is a
            // transposed view, and handing one to `asData` costs a gather.
            let shard = MLX.contiguous(qkvWeight[rows].transposed().asType(.float16))
            shard.eval()
            return shard.asData(access: .noCopyIfContiguous).data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return false }
                return h3_ane_tensor_write(tensor, base, Int32(k), Int32(plan.perDie))
            }
        }

        guard upload(plan.ane0, into: w0), upload(plan.ane1, into: w1) else {
            h3_ane_tensor_free(w0); h3_ane_tensor_free(w1)
            return nil
        }

        let set = WeightSet(w0: w0, w1: w1, source: qkvWeight)
        lock.lock(); weights[key] = set; lock.unlock()
        return set
    }

    // MARK: - The projection

    /// `x[S, K] @ weight[N, K]^T`, with `aneShare` of the output channels
    /// computed on the engine and the rest on the GPU.
    ///
    /// Nothing here is specific to qkv — it is an output-channel split of an
    /// ordinary linear, and the shard plan is derived from `N`. What decides
    /// whether a given projection may use it is not this function but the
    /// saturation bound: `fc2` drives interior partials to 34,649 at block 49
    /// against a 2^15 cliff that returns zero silently, so it stays on the GPU
    /// until that is bounded per block. `qkv` peaks at 485 and `fc1` at 72.
    ///
    /// Falls back to a plain `matmul` — same result, no speedup — whenever the
    /// engine cannot take the shape or a step fails.
    /// - Parameter label: which projection this is, recorded on the render
    ///   receipt when the engine actually takes it. A receipt that cannot tell
    ///   you which arithmetic produced a video cannot support the
    ///   reproducibility claim it is written to make, and routing changes the
    ///   arithmetic: the same seed and checkpoint give a different sample.
    public static func project(x: MLXArray, weight: MLXArray, label: String) -> MLXArray {
        guard isEnabled else { return MLX.matmul(x, weight.transposed()) }
        guard let routed = route(x: x, qkvWeight: weight) else {
            // A decline is silent to the caller by design — same numbers, no
            // speedup — but it must not be silent to the receipt, or a render
            // that fell back looks like one that did not.
            note(label, routed: false)
            return MLX.matmul(x, weight.transposed())
        }
        note(label, routed: true)
        return routed
    }

    nonisolated(unsafe) private static var routed: Set<String> = []
    nonisolated(unsafe) private static var declined: Set<String> = []

    private static func note(_ label: String, routed didRoute: Bool) {
        lock.lock(); defer { lock.unlock() }
        if didRoute { routed.insert(label) } else { declined.insert(label) }
    }

    /// Projections the engine actually computed during this process, and any
    /// that were offered to it and fell back.
    ///
    /// Observed rather than declared. A list of what *should* be routed drifts
    /// from the call sites; this is what ran.
    public static var routedProjections: [String] {
        lock.lock(); defer { lock.unlock() }
        return routed.sorted()
    }

    public static var declinedProjections: [String] {
        lock.lock(); defer { lock.unlock() }
        return declined.subtracting(routed).sorted()
    }

    /// The routed path itself, with the opt-in check left out.
    ///
    /// Kept separate so routing failures can return nil to the production entry
    /// point, which owns the unconditional MLX fallback.
    ///
    /// Returns nil when the engine cannot take this shape or a step fails, so
    /// every caller decides its own fallback.
    static func route(x: MLXArray, qkvWeight: MLXArray) -> MLXArray? {
        guard h3_ane_is_available(), x.ndim == 2, qkvWeight.ndim == 2 else { return nil }

        let s = x.shape[0], k = x.shape[1], n = qkvWeight.shape[0]
        guard qkvWeight.shape[1] == k, let plan = plan(n: n) else { return nil }
        guard let session = session(s: s, k: k, perDie: plan.perDie),
              let weights = weightSet(for: qkvWeight, plan: plan, k: k)
        else { return nil }

        // A Session owns one mutable activation and two mutable outputs. Hold
        // it until the ANE results have been copied into MLX-owned storage.
        session.executionLock.lock()
        defer { session.executionLock.unlock() }

        // Order matters more than anything else in this function.
        //
        // The activation has to be materialised before the engine can read it,
        // and materialising it drains the GPU queue. So it goes FIRST, while
        // that queue is still empty. Submitting the GPU shard before this point
        // — which is what an earlier version did — means the drain waits for
        // the whole 14592-column matmul, the engine starts only after the GPU
        // has finished, and the two costs add instead of overlapping: 39.9 ms
        // serial against 20.2 ms overlapped, for work that is 19.2 and 19.8 ms
        // on its own.
        //
        // `MLX.contiguous` is not decoration either. `transposed()` returns a
        // strided view and `asType` preserves the strides, so without it
        // `asData(access: .noCopyIfContiguous)` finds a non-contiguous array
        // and silently falls back to `asDataCopy()` — an element-wise gather
        // over 11M elements that measured 8792 ms against 0.4 ms for the
        // memcpy it replaces.
        let scaled = MLX.contiguous((x * operandScale).transposed().asType(.float16))
        scaled.eval()
        let uploaded = scaled.asData(access: .noCopyIfContiguous).data.withUnsafeBytes { raw -> Bool in
            guard let base = raw.baseAddress else { return false }
            return h3_ane_tensor_write_prefix(session.x, base, Int32(k), Int32(s))
        }

        // Now submit the GPU shard. `asyncEval` queues the Metal work and
        // returns, so the GPU is busy with it for the whole time the calling
        // thread sits blocked inside the engine below. Its own stream keeps it
        // from being dragged into any later synchronisation on the default one.
        let gpuOut = Stream.withNewDefaultStream(device: .gpu) {
            let output = MLX.matmul(x, qkvWeight[plan.gpu].transposed())
            MLX.asyncEval(output)
            return output
        }
        guard uploaded else { return nil }

        guard h3_ane_run_pair(session.program0, session.x, weights.w0, session.y0,
                              session.program1, session.x, weights.w1, session.y1)
        else { return nil }

        // Read the results by aliasing the output surfaces rather than copying
        // them home: `MLXArray(rawPointer:)` transfers a managed external
        // allocation into MLX without copying, per mlx-swift's initializer
        // contract. The finalizer deliberately frees nothing — the Session owns
        // these surfaces and reuses them next call.
        // The surfaces are the compiled (padded) length; the surplus rows hold
        // results computed from the zero columns and are dropped here. Slicing
        // a row prefix of a row-major array stays contiguous, so this costs
        // nothing beyond the scale-and-convert that follows.
        func adopt(_ tensor: OpaquePointer) -> MLXArray? {
            guard let ptr = h3_ane_tensor_ptr(tensor) else { return nil }
            let full = MLXArray(rawPointer: ptr, [session.s, plan.perDie], dtype: .float16) { }
            return session.s == s ? full : full[0 ..< s]
        }
        guard let raw0 = adopt(session.y0), let raw1 = adopt(session.y1) else { return nil }

        // Undo the operand scale and join. The `eval` is what copies the engine's
        // output out of the shared surfaces, and it has to happen before the next
        // call overwrites them — returning a lazy graph over live surfaces would
        // read whichever block ran most recently.
        let ane0 = (raw0 * (1 / operandScale)).asType(.bfloat16)
        let ane1 = (raw1 * (1 / operandScale)).asType(.bfloat16)
        MLX.eval(ane0, ane1)

        return MLX.concatenated([gpuOut, ane0, ane1], axis: -1)
    }
}
