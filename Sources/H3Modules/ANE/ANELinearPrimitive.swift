// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Metal
import MLX
import MLXFast
import MLXNN
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

    /// The native IOSurface pack/merge path is a separately gated research
    /// arm. It changes the synchronization seam, not the model decomposition,
    /// and remains opt-in until its production block clears the 1.15x gate.
    ///
    /// A `var` so a benchmark can interleave it against the CPU seam in one
    /// process. Cross-process comparison drifts about 4% here — more than the
    /// effect being measured — and a control that moves more than the treatment
    /// cannot decide anything. The two paths are bit-identical, so alternating
    /// them changes only which seam is timed.
    nonisolated(unsafe) static var nativeIOEnabled =
        ProcessInfo.processInfo.environment["H3_ANE_NATIVE_IO"] == "1"

    /// The fused nonlinear merge is kept behind a narrower research switch.
    /// Its kernel is useful for measuring the ceiling, but random production
    /// values have not yet matched MLX's fused bf16 SiLU bit-for-bit. Native
    /// pack and linear merge remain exact without enabling this switch.
    static var nativeFusedSwiGLUEnabled: Bool {
        nativeIOEnabled
            && ProcessInfo.processInfo.environment["H3_ANE_NATIVE_FUSED_SWIGLU"] == "1"
    }

    private static let metalDevice = MTLCreateSystemDefaultDevice()

    /// True after a guided step overlapped GPU attention on one branch with
    /// engine linears on the other. Observed for the receipt, like routing.
    nonisolated(unsafe) package static var overlappedCFG = false
    nonisolated(unsafe) package static var usedNativeIO = false
    nonisolated(unsafe) package static var queryTilesUsed = 0

    /// Whether a guided step overlaps the two CFG branches.
    ///
    /// **Off, and measured off.** The schedule works — it is bit-identical and
    /// it recovers real overlap, 1.205x against a serial block at the same
    /// engine share — but no share makes it beat the plain routed path in
    /// absolute time, because every column moved to the engine costs more in
    /// routing overhead than the overlap gives back. Per CFG pair at production
    /// width, depth 4:
    ///
    /// | engine share | serial | pipelined |
    /// |---|---:|---:|
    /// | unrouted | 2339.0 ms | 2339.5 ms |
    /// | 0.286 | **2219.8 ms** | 2226.8 ms |
    /// | 0.33 | 2318.8 ms | 2244.4 ms |
    /// | 0.45 | 2617.2 ms | 2341.0 ms |
    /// | 0.60 | 3069.7 ms | 2546.6 ms |
    ///
    /// The minimum of that table is the configuration that already shipped.
    /// `H3_ANE_CFG_OVERLAP=1` turns the schedule on for research; see
    /// `docs/ANE_STATUS.md` for why the ceiling is where it is.
    static let cfgOverlapEnabled: Bool =
        ProcessInfo.processInfo.environment["H3_ANE_CFG_OVERLAP"] == "1"

    package static func markCFGOverlap() {
        lock.lock(); defer { lock.unlock() }
        overlappedCFG = true
    }

    package static func markNativeIO() {
        lock.lock(); defer { lock.unlock() }
        usedNativeIO = true
    }

    package static func markQueryTiling(_ count: Int) {
        lock.lock(); defer { lock.unlock() }
        queryTilesUsed = max(queryTilesUsed, count)
    }

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
    /// **0.286 is the balance point for a projection running alone, and it is
    /// the wrong number when two CFG branches overlap.** Alone, the GPU and the
    /// engine race on the same GEMM and the share that makes them finish
    /// together wins. Overlapped, the engine has the whole block-pair to work
    /// in — the GPU is busy with the other branch's attention — so it should
    /// take much more than half, and the balance moves to where the *engine's*
    /// total equals the *GPU's* total across both branches.
    ///
    /// Solving that balance on the measured numbers — 2347.7 ms of GPU work a
    /// pair, of which 1050.4 is routable, against an engine at 0.43 of the
    /// GPU's rate — puts the overlapped share at 0.67. `H3_ANE_SHARE` overrides
    /// it for sweeps.
    /// **The two windows want different shares, so there are two.**
    ///
    /// `qkv` runs before attention: nothing else in the block is available to
    /// cover it, so the engine and the GPU race on the same GEMM and 0.286 —
    /// the point where they finish together — is still right.
    ///
    /// `attn out` and `fc1` run *after* attention, and under query tiling that
    /// is a different situation entirely: tile `i-1`'s post-attention chain has
    /// the GPU's attention on tile `i` to hide behind. The engine should take
    /// as much of those columns as fits the window, which is more than half of
    /// what the racing answer would give it.
    ///
    /// `H3_ANE_SHARE_QKV` and `H3_ANE_SHARE_POST` sweep them independently;
    /// `H3_ANE_SHARE` overrides both, for comparison against the single-share
    /// measurements this replaced.
    private static let racingShare = 0.286

    private static func envShare(_ name: String) -> Double? {
        guard let raw = ProcessInfo.processInfo.environment[name],
              let value = Double(raw), value > 0, value < 1 else { return nil }
        return value
    }

    nonisolated(unsafe) static var share: Double =
        envShare("H3_ANE_SHARE") ?? racingShare

    nonisolated(unsafe) static var postShare: Double =
        envShare("H3_ANE_SHARE") ?? envShare("H3_ANE_SHARE_POST") ?? 0.45

    /// Which window this projection sits in. `qkv` gates attention; everything
    /// else in a block comes after it.
    static func share(for label: String) -> Double {
        label == "qkv" ? share : postShare
    }

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

    static func plan(n: Int, headDim: Int = 128, share: Double? = nil) -> ShardPlan? {
        guard n >= 8 * headDim, n % headDim == 0 else { return nil }
        let columns = share ?? Self.share
        let perDie = Int((Double(n) * columns / 2 / Double(headDim)).rounded()) * headDim
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

    /// The engine, driven by a thread that never touches MLX.
    ///
    /// `h3_ane_run_pair` blocks for as long as the shard takes — 135 ms at the
    /// production shard — and mlx-swift serialises **every** `eval` in the
    /// process behind one global recursive lock. A thread that blocks on the
    /// engine while holding that lock stops the GPU dead, which is why the
    /// first overlap schedule measured 0.987x: the two halves took turns
    /// through `evalLock` instead of running side by side.
    ///
    /// So the engine gets its own thread. The MLX thread hands it work and
    /// collects the result later, and in between it is free to keep the GPU
    /// fed with the other CFG branch.
    final class Engine: @unchecked Sendable {
        static let shared = Engine()
        private let queue = DispatchQueue(label: "h3.ane.engine", qos: .userInitiated)

        /// One submitted evaluation. `wait` is the only synchronisation point,
        /// and it is deliberately not called until the caller needs the numbers.
        final class Job: @unchecked Sendable {
            private let done = DispatchSemaphore(value: 0)
            private var ok = false
            func settle(_ value: Bool) { ok = value; done.signal() }
            func wait() -> Bool { done.wait(); return ok }
        }

        func submit(_ work: @escaping @Sendable () -> Bool) -> Job {
            let job = Job()
            queue.async { job.settle(work()) }
            return job
        }
    }

    /// The surfaces one evaluation owns from activation upload to copy-out.
    ///
    /// The engine reads and writes these in place, so a second evaluation
    /// cannot start on them until the first has been copied into MLX-owned
    /// storage. When the engine ran synchronously one set per shape sufficed;
    /// with work queued ahead of the GPU there must be one set per job in
    /// flight, or the schedule that makes this fast also makes it wrong.
    final class Slot: @unchecked Sendable {
        let x: OpaquePointer
        let y0: OpaquePointer, y1: OpaquePointer

        init?(s: Int, k: Int, perDie: Int) {
            // The engine reads the activation with sequence as the minor axis,
            // so this surface is [k, s], not [s, k].
            guard let xt = h3_ane_tensor_create(Int32(k), Int32(s)) else { return nil }
            guard let y0t = h3_ane_tensor_create(Int32(s), Int32(perDie)) else {
                h3_ane_tensor_free(xt); return nil
            }
            guard let y1t = h3_ane_tensor_create(Int32(s), Int32(perDie)) else {
                h3_ane_tensor_free(xt); h3_ane_tensor_free(y0t); return nil
            }
            // Outputs are aliased into MLX, so padding is not representable.
            guard h3_ane_tensor_is_dense(y0t), h3_ane_tensor_is_dense(y1t) else {
                h3_ane_tensor_free(xt); h3_ane_tensor_free(y0t); h3_ane_tensor_free(y1t)
                return nil
            }
            self.x = xt; self.y0 = y0t; self.y1 = y1t
        }

        deinit {
            h3_ane_tensor_free(x); h3_ane_tensor_free(y0); h3_ane_tensor_free(y1)
        }
    }

    /// One compiled program per die plus a pool of surfaces they read and write.
    ///
    /// A program is fixed at one `(s, k, n)` and compiling costs real time, so
    /// these are cached for the life of the process. Sequence length is
    /// constant across a render, so in practice exactly one session is built
    /// per projection width.
    ///
    /// Both dies read the *same* activation surface — the shards differ only in
    /// which output channels they produce — so the activation is uploaded once
    /// per call rather than once per die.
    final class Session {
        /// The compiled length, which is `paddedSequence(requested)`.
        let s: Int, k: Int, perDie: Int
        let program0: OpaquePointer, program1: OpaquePointer

        /// Three. Two covers the CFG schedule, which has at most two branches
        /// of one projection outstanding; query tiling wants one more, because
        /// it submits the next tile's job before waiting on the current one and
        /// collects the tile behind that. At tile size the surfaces are small —
        /// a `fc1` slot at T=8 is 46 MB against 427 at full length — so the
        /// third is nearly free where it is actually used.
        static let slotCount = 3
        private let lock = NSLock()
        private var idle: [Slot]
        private let available: DispatchSemaphore
        private var poisoned = false

        var isUsable: Bool {
            lock.lock(); defer { lock.unlock() }
            return !poisoned
        }

        init?(s requested: Int, k: Int, perDie: Int) {
            let s = ANELinearBackend.paddedSequence(requested)
            guard let p0 = h3_ane_program_create(Int32(s), Int32(k), Int32(perDie)) else { return nil }
            guard let p1 = h3_ane_program_create(Int32(s), Int32(k), Int32(perDie)) else {
                h3_ane_program_free(p0); return nil
            }
            var slots: [Slot] = []
            for _ in 0 ..< Self.slotCount {
                guard let slot = Slot(s: s, k: k, perDie: perDie) else {
                    h3_ane_program_free(p0); h3_ane_program_free(p1); return nil
                }
                slots.append(slot)
            }
            self.s = s; self.k = k; self.perDie = perDie
            self.program0 = p0; self.program1 = p1
            self.idle = slots
            self.available = DispatchSemaphore(value: slots.count)
        }

        /// Blocks when every slot is in flight, which is the backpressure that
        /// keeps the queue from growing without bound.
        func take() -> Slot? {
            guard isUsable else { return nil }
            available.wait()
            lock.lock(); defer { lock.unlock() }
            guard !poisoned else {
                available.signal()
                return nil
            }
            return idle.removeLast()
        }

        func give(_ slot: Slot) {
            lock.lock(); idle.append(slot); lock.unlock()
            available.signal()
        }

        /// `evaluateWithQoS:` has no cancellation operation. After a failed
        /// pair the private runtime may still own these mutable surfaces, so
        /// retire the whole session and deliberately keep this slot out of its
        /// free list rather than racing a fallback request against stale ANE
        /// writes.
        func quarantine(_ slot: Slot) {
            lock.lock(); poisoned = true; lock.unlock()
            _ = slot
        }

        deinit {
            h3_ane_program_free(program0); h3_ane_program_free(program1)
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
        /// **Which columns these two surfaces hold.** The shards are a slice of
        /// the weight, so a set uploaded for one shard plan is the wrong data
        /// for another — and the only thing that changes the plan is the engine
        /// share, which is a knob. Keyed on the weight alone, a share sweep
        /// silently reuses the previous split's columns and reports numbers for
        /// a projection nobody ran.
        var perDie: Int = 0
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
        if let cached = sessions[key] {
            if cached.isUsable { return cached }
            refused.insert(key)
            return nil
        }
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
           cached.shape == qkvWeight.shape, cached.perDie == plan.perDie {
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
        set.perDie = plan.perDie
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
        guard let routed = route(x: x, qkvWeight: weight, share: share(for: label)) else {
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

    /// One projection whose engine half is in flight.
    ///
    /// The point of the split is the gap between `begin` and `value`: the
    /// caller submits the engine's shards, gets this back immediately, spends
    /// the interval doing GPU work for the *other* CFG branch, and only then
    /// asks for the numbers. The engine and the GPU are busy for the whole
    /// interval, which is the entire reason any of this is faster.
    ///
    /// A class rather than a struct so a `Pending` that is dropped without
    /// being collected still returns its slot; leaking one would deadlock the
    /// next block on `Session.take`.
    package final class Pending {
        fileprivate enum Body {
            /// The engine declined, or routing is off. Same numbers, no engine.
            case plain(MLXArray)
            /// `x` and `weight` are carried so a failure after submission can
            /// still produce the right answer. The engine can decline before
            /// the GPU shard is queued — `start` returns nil and the caller
            /// falls back — but it can also fail *after*, and at that point the
            /// only thing in hand is a partial projection over a fraction of
            /// the output columns. Returning that would be a silently wrong
            /// block, which is the one failure mode this whole path is built to
            /// avoid.
            case routed(gpu: MLXArray, job: Engine.Job,
                        session: Session, slot: Slot, rows: Int, perDie: Int,
                        x: MLXArray, weight: MLXArray)
        }
        fileprivate let body: Body
        private var collected: MLXArray?
        private var swigluCollected: MLXArray?
        fileprivate init(_ body: Body) { self.body = body }

        /// Waits for the engine, adopts its output, and joins the two halves.
        ///
        /// Call this as late as the data dependency allows. Calling it straight
        /// after `begin` is exactly the synchronous path, and measures like it.
        package func value() -> MLXArray {
            if let collected { return collected }
            let result: MLXArray
            switch body {
            case .plain(let v):
                result = v
            case .routed(let gpu, let job, let session, let slot, let rows, let perDie,
                         let x, let weight):
                guard job.wait() else {
                    // The engine failed after the GPU shard was queued. `gpu`
                    // holds only the columns the GPU was given, so the whole
                    // projection is recomputed rather than returned in part.
                    // Slower than the plain path and correct, which is the only
                    // acceptable direction for this to fail in.
                    session.quarantine(slot)
                    ANELinearBackend.note("engine failed mid-flight", routed: false)
                    result = MLX.matmul(x, weight.transposed())
                    collected = result
                    return result
                }
                result = ANELinearBackend.join(gpu: gpu, slot: slot, session: session,
                                               rows: rows, perDie: perDie)
                session.give(slot)
            }
            collected = result
            return result
        }

        /// Collects an `fc1` projection directly as `SiLU(gate) * up` when the
        /// native merge path is active. This avoids materialising and joining
        /// the full 28,672-column tensor only to split it again immediately.
        package func swiGLUValue() -> MLXArray {
            if let swigluCollected { return swigluCollected }
            if let collected {
                let parts = collected.split(parts: 2, axis: -1)
                let result = silu(parts[0]) * parts[1]
                swigluCollected = result
                return result
            }

            let result: MLXArray
            switch body {
            case .plain(let v):
                let parts = v.split(parts: 2, axis: -1)
                result = silu(parts[0]) * parts[1]
            case .routed(let gpu, let job, let session, let slot, let rows, let perDie,
                         let x, let weight):
                guard job.wait() else {
                    session.quarantine(slot)
                    ANELinearBackend.note("engine failed mid-flight", routed: false)
                    let full = MLX.matmul(x, weight.transposed())
                    let parts = full.split(parts: 2, axis: -1)
                    result = silu(parts[0]) * parts[1]
                    swigluCollected = result
                    return result
                }
                if let fused = ANELinearBackend.joinSwiGLU(
                    gpu: gpu, slot: slot, session: session,
                    rows: rows, perDie: perDie
                ) {
                    result = fused
                } else {
                    let full = ANELinearBackend.join(gpu: gpu, slot: slot, session: session,
                                                     rows: rows, perDie: perDie)
                    let parts = full.split(parts: 2, axis: -1)
                    result = silu(parts[0]) * parts[1]
                }
                session.give(slot)
            }
            swigluCollected = result
            return result
        }

        deinit {
            guard collected == nil, swigluCollected == nil,
                  case .routed(_, let job, let session, let slot, _, _, _, _) = body
            else { return }
            if job.wait() { session.give(slot) }
            else { session.quarantine(slot) }
        }
    }

    /// The pointers one queued evaluation needs, in a parcel the queue accepts.
    ///
    /// `OpaquePointer` is not `Sendable` and should not be — but these are
    /// surfaces the slot pool has already made exclusive to this job, which is
    /// the invariant the compiler cannot see.
    fileprivate struct ShardHandles: @unchecked Sendable {
        let x: OpaquePointer
        let y0: OpaquePointer, y1: OpaquePointer
        let p0: OpaquePointer, p1: OpaquePointer
        let w0: OpaquePointer, w1: OpaquePointer
    }

    /// Submits both shards to the engine and returns without waiting.
    ///
    /// Everything before the submission is GPU work on the calling thread and
    /// has to happen here: the activation must be materialised and copied into
    /// the engine's surface before the engine can read it.
    package static func begin(x: MLXArray, weight: MLXArray, label: String) -> Pending {
        // Without the opt-in nothing is offered, so nothing is recorded. A
        // decline noted here would put entries on the receipt of a render that
        // never went near the engine, and the receipt's whole job is to say
        // which arithmetic produced the output.
        guard isEnabled else { return Pending(.plain(MLX.matmul(x, weight.transposed()))) }
        guard let body = start(x: x, weight: weight, share: share(for: label)) else {
            note(label, routed: false)
            return Pending(.plain(MLX.matmul(x, weight.transposed())))
        }
        note(label, routed: true)
        return Pending(body)
    }

    /// The routed path itself, with the opt-in check left out.
    ///
    /// Kept separate so routing failures can return nil to the production entry
    /// point, which owns the unconditional MLX fallback.
    ///
    /// Returns nil when the engine cannot take this shape or a step fails, so
    /// every caller decides its own fallback.
    static func route(x: MLXArray, qkvWeight: MLXArray, share: Double? = nil) -> MLXArray? {
        guard let body = start(x: x, weight: qkvWeight, share: share) else { return nil }
        return Pending(body).value()
    }

    fileprivate static func start(x: MLXArray, weight: MLXArray,
                                  share: Double? = nil) -> Pending.Body? {
        guard h3_ane_is_available(), x.ndim == 2, weight.ndim == 2 else { return nil }

        let s = x.shape[0], k = x.shape[1], n = weight.shape[0]
        guard weight.shape[1] == k, let plan = plan(n: n, share: share) else { return nil }
        guard let session = session(s: s, k: k, perDie: plan.perDie),
              let weights = weightSet(for: weight, plan: plan, k: k)
        else { return nil }

        // A slot owns one mutable activation and two mutable outputs from here
        // until `Pending.value` has copied the results into MLX.
        guard let slot = session.take() else { return nil }

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
        // **On its own stream, and that is the whole schedule.** The default
        // stream is where attention was just queued with `asyncEval` — 880 ms
        // of it for two CFG branches — and Metal runs a stream's commands in
        // order. Materialising the activation there means the engine cannot be
        // fed until the GPU has drained the very work it was supposed to run
        // beside, which is a serial block wearing a schedule's clothes.
        //
        // On a separate stream this waits only for `x` itself, through the
        // cross-stream dependency MLX inserts, and the attention keeps running.
        let uploaded: Bool
        if nativeIOEnabled, let device = metalDevice {
            let source = Stream.withNewDefaultStream(device: .gpu) { () -> MLXArray in
                let v = MLX.contiguous(x)
                v.eval()
                return v
            }
            if let buffer = source.asMTLBuffer(device: device, noCopy: true) {
                uploaded = h3_ane_pack_bf16_to_fp16_transpose(
                    buffer.contents(), slot.x, Int32(s), Int32(k), nil)
            } else {
                uploaded = false
            }
        } else {
            let scaled = Stream.withNewDefaultStream(device: .gpu) { () -> MLXArray in
                let v = MLX.contiguous((x * operandScale).transposed().asType(.float16))
                v.eval()
                return v
            }
            uploaded = scaled.asData(access: .noCopyIfContiguous).data.withUnsafeBytes { raw -> Bool in
                guard let base = raw.baseAddress else { return false }
                return h3_ane_tensor_write_prefix(slot.x, base, Int32(k), Int32(s))
            }
        }
        guard uploaded else { session.give(slot); return nil }
        if nativeIOEnabled { markNativeIO() }

        // The engine goes to its own thread and this one returns immediately.
        // Nothing below here blocks, so the caller keeps the GPU busy while the
        // shards run.
        let shards = ShardHandles(x: slot.x, y0: slot.y0, y1: slot.y1,
                                  p0: session.program0, p1: session.program1,
                                  w0: weights.w0, w1: weights.w1)
        let job = Engine.shared.submit {
            h3_ane_run_pair(shards.p0, shards.x, shards.w0, shards.y0,
                            shards.p1, shards.x, shards.w1, shards.y1)
        }

        // `asyncEval` queues the Metal work and returns, so the GPU is busy with
        // it while the engine runs. Its own stream keeps it from being dragged
        // into any later synchronisation on the default one.
        let gpuOut = Stream.withNewDefaultStream(device: .gpu) { () -> MLXArray in
            let output = MLX.matmul(x, weight[plan.gpu].transposed())
            MLX.asyncEval(output)
            return output
        }

        return .routed(gpu: gpuOut, job: job, session: session, slot: slot,
                       rows: s, perDie: plan.perDie, x: x, weight: weight)
    }

    /// Adopts the engine's two output surfaces and joins them to the GPU shard.
    fileprivate static func join(gpu: MLXArray, slot: Slot, session: Session,
                                 rows s: Int, perDie: Int) -> MLXArray {
        if nativeIOEnabled, let device = metalDevice {
            let gpu = MLX.contiguous(gpu)
            let total = gpu.dim(-1) + 2 * perDie
            let output = MLXArray.zeros([s, total], dtype: .bfloat16)
            if let gpuBuffer = gpu.asMTLBuffer(device: device, noCopy: true),
               let outputBuffer = output.asMTLBuffer(device: device, noCopy: true),
               h3_ane_merge_attn_out(
                   gpuBuffer.contents(), slot.y0, slot.y1, outputBuffer.contents(),
                   Int32(s), Int32(gpu.dim(-1)), Int32(perDie), Int32(perDie), nil
               ) {
                return output
            }
        }

        // Read the results by aliasing the output surfaces rather than copying
        // them home: `MLXArray(rawPointer:)` transfers a managed external
        // allocation into MLX without copying, per mlx-swift's initializer
        // contract. The finalizer deliberately frees nothing — the Slot owns
        // these surfaces and reuses them next call.
        // The surfaces are the compiled (padded) length; the surplus rows hold
        // results computed from the zero columns and are dropped here. Slicing
        // a row prefix of a row-major array stays contiguous, so this costs
        // nothing beyond the scale-and-convert that follows.
        func adopt(_ tensor: OpaquePointer) -> MLXArray {
            let ptr = h3_ane_tensor_ptr(tensor)!
            let full = MLXArray(rawPointer: ptr, [session.s, perDie], dtype: .float16) { }
            return session.s == s ? full : full[0 ..< s]
        }

        // Undo the operand scale and join. The `eval` is what copies the engine's
        // output out of the shared surfaces, and it has to happen before the slot
        // is reused — returning a lazy graph over live surfaces would read
        // whichever block ran most recently.
        let ane0 = (adopt(slot.y0) * (1 / operandScale)).asType(.bfloat16)
        let ane1 = (adopt(slot.y1) * (1 / operandScale)).asType(.bfloat16)
        MLX.eval(ane0, ane1)

        return MLX.concatenated([gpu, ane0, ane1], axis: -1)
    }

    /// Native `fc1` merge that emits the gated FFN activation directly.
    fileprivate static func joinSwiGLU(gpu: MLXArray, slot: Slot, session: Session,
                                       rows s: Int, perDie: Int) -> MLXArray? {
        guard nativeFusedSwiGLUEnabled, let device = metalDevice else { return nil }
        let gpu = MLX.contiguous(gpu)
        let total = gpu.dim(-1) + 2 * perDie
        guard total.isMultiple(of: 2) else { return nil }
        let output = MLXArray.zeros([s, total / 2], dtype: .bfloat16)
        guard let gpuBuffer = gpu.asMTLBuffer(device: device, noCopy: true),
              let outputBuffer = output.asMTLBuffer(device: device, noCopy: true),
              h3_ane_merge_fc1_swiglu(
                  gpuBuffer.contents(), slot.y0, slot.y1, outputBuffer.contents(),
                  Int32(s), Int32(gpu.dim(-1)), Int32(perDie), Int32(perDie), nil
              )
        else { return nil }
        return output
    }
}
