// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXFast
import H3Foundation
import H3Hardware
import H3ANEBridge

/// Dense attention split by complete heads across Metal and both ANE dies.
///
/// Four heads per ANE is deliberate. At the production sequence (15,731), a
/// four-head graph takes about the same 256 ms as one head, while eight heads
/// crosses a compiler/runtime tiling boundary and takes roughly 20 seconds.
/// The remaining heads stay in one MLX SDPA, and all three devices run at once.
/// No score plane crosses a device boundary; only Q/K/V and completed head
/// outputs do.
package struct ANEAttentionBackend: H3AttentionBackend {
    package static let identifier = "ane"
    package static let equivalenceClass: Float = 0.003
    package static let materialisesScores = false
    package static let prefersMortonOrder = false

    /// Four heads per die is the measured default; `H3_ANE_ATTENTION_HEADS`
    /// re-sweeps it against a capture when the shape or the OS changes.
    private static let headsPerDie: Int = {
        let requested = ProcessInfo.processInfo.environment["H3_ANE_ATTENTION_HEADS"]
            .flatMap(Int.init)
        guard let requested, requested > 0 else { return 4 }
        return requested
    }()
    private static var engineHeads: Int { 2 * headsPerDie }

    /// Gated on `H3_ANE_ATTENTION` alone, deliberately. This route shares no
    /// code with the linear primitive behind `H3_ANE`, and requiring both
    /// meant enabling attention also enabled linear routing — which on 27.0
    /// fails closed with `Request cancelled` under render load and poisoned
    /// the measurement it was supposed to isolate.
    package static func isAvailable(on machine: Machine) -> Bool {
        ProcessInfo.processInfo.environment["H3_ANE_ATTENTION"] == "1"
            && h3_ane_is_available()
    }

    package init() {}

    private final class Slot: @unchecked Sendable {
        let q0: OpaquePointer, k0: OpaquePointer, v0: OpaquePointer, y0: OpaquePointer
        let q1: OpaquePointer, k1: OpaquePointer, v1: OpaquePointer, y1: OpaquePointer

        init?(sequence: Int, dimension: Int) {
            let rows = ANEAttentionBackend.headsPerDie * sequence
            var made: [OpaquePointer] = []
            func tensor() -> OpaquePointer? {
                guard let value = h3_ane_tensor_create(Int32(rows), Int32(dimension))
                else { return nil }
                made.append(value)
                return value
            }
            guard let q0 = tensor(), let k0 = tensor(), let v0 = tensor(), let y0 = tensor(),
                  let q1 = tensor(), let k1 = tensor(), let v1 = tensor(), let y1 = tensor()
            else {
                for value in made { h3_ane_tensor_free(value) }
                return nil
            }
            self.q0 = q0; self.k0 = k0; self.v0 = v0; self.y0 = y0
            self.q1 = q1; self.k1 = k1; self.v1 = v1; self.y1 = y1
        }

        deinit {
            for value in [q0, k0, v0, y0, q1, k1, v1, y1] {
                h3_ane_tensor_free(value)
            }
        }
    }

    private final class Session: @unchecked Sendable {
        let sequence: Int, dimension: Int
        let p0: OpaquePointer, p1: OpaquePointer
        let slot: Slot
        let available = DispatchSemaphore(value: 1)
        private let lock = NSLock()
        private var poisoned = false

        init?(sequence: Int, dimension: Int, heads: Int) {
            let perDie = Int32(ANEAttentionBackend.headsPerDie)
            guard let p0 = h3_ane_attention_program_create(
                perDie, Int32(sequence), Int32(dimension)
            ) else { return nil }
            guard let p1 = h3_ane_attention_program_create(
                perDie, Int32(sequence), Int32(dimension)
            ) else {
                h3_ane_attention_program_free(p0)
                return nil
            }
            guard let slot = Slot(sequence: sequence, dimension: dimension) else {
                h3_ane_attention_program_free(p0)
                h3_ane_attention_program_free(p1)
                return nil
            }
            guard ANEAttentionBackend.worthTaking(
                p0: p0, p1: p1, slot: slot,
                sequence: sequence, dimension: dimension, heads: heads
            ) else {
                h3_ane_attention_program_free(p0)
                h3_ane_attention_program_free(p1)
                return nil
            }
            self.sequence = sequence; self.dimension = dimension
            self.p0 = p0; self.p1 = p1; self.slot = slot
        }

        func take() -> Bool {
            lock.lock(); let usable = !poisoned; lock.unlock()
            guard usable else { return false }
            return available.wait(timeout: .now() + .seconds(10)) == .success
        }

        func give(success: Bool) {
            if !success { lock.lock(); poisoned = true; lock.unlock() }
            available.signal()
        }

        deinit {
            h3_ane_attention_program_free(p0)
            h3_ane_attention_program_free(p1)
        }
    }

    private final class Job: @unchecked Sendable {
        private let done = DispatchSemaphore(value: 0)
        private var result = false
        func settle(_ value: Bool) { result = value; done.signal() }
        func wait() -> Bool { done.wait(); return result }
    }

    /// A silent `nil` from `attend` is indistinguishable from a fast render:
    /// the caller falls back to dense SDPA. `H3_ANE_ATTENTION_TRACE=1` reports
    /// the first decline of each kind and the routed count, so a measurement
    /// cannot be attributed to a route that never ran.
    nonisolated(unsafe) private static var traced: Set<String> = []
    nonisolated(unsafe) private static var routedCalls = 0
    private static let tracing =
        ProcessInfo.processInfo.environment["H3_ANE_ATTENTION_TRACE"] == "1"

    private static func decline(_ reason: @autoclosure () -> String) -> MLXArray? {
        guard tracing else { return nil }
        let text = reason()
        lock.lock()
        let fresh = traced.insert(text).inserted
        lock.unlock()
        if fresh { FileHandle.standardError.write(Data("[ane-attn] declined: \(text)\n".utf8)) }
        return nil
    }

    private static func countRouted() {
        guard tracing else { return }
        lock.lock(); routedCalls += 1; let n = routedCalls; lock.unlock()
        if n == 1 || n % 100 == 0 {
            FileHandle.standardError.write(Data("[ane-attn] routed \(n) calls\n".utf8))
        }
    }

    /// The decision the calibration makes, separated from the timing so it can
    /// be tested on numbers instead of provoked out of a shape.
    ///
    /// Routing costs `max(engine, retained)`, because the two devices run
    /// concurrently, and it is worth doing only if that beats `unrouted` — the
    /// whole thing on the GPU. The margin covers the upload and copy back,
    /// which the race excludes.
    package static func shouldRoute(engine: Double, retained: Double,
                                    unrouted: Double) -> Bool {
        max(engine, retained) < unrouted * 0.97
    }

    /// Decides, by measurement, whether this shape belongs on the dies at all.
    ///
    /// Every ANE constant in this file is shape-specific in a way nothing
    /// predicts. Four heads a graph runs at 1.90 TFLOP/s and five at 0.20.
    /// S=15,744 runs at 1.71 and S=15,731 — thirteen elements away — at 0.41.
    /// Neither is alignment, capacity, or anything else with a rule behind it,
    /// and a slow shape is not an error: it computes the right answer four
    /// times too slowly, so it would ship as a silent regression.
    ///
    /// So rather than threshold on a TFLOP/s figure that would itself be a
    /// guess, race the two devices on the actual question. The split is worth
    /// taking only if the dies finish their `2 * headsPerDie` heads before the
    /// GPU finishes the `heads - 2 * headsPerDie` it keeps; past that the dies
    /// become the critical path and the route is a loss. Timing both needs no
    /// constant for either device's rate, and it catches every cliff at once —
    /// including ones not yet found.
    ///
    /// Costs about a second, once per shape, against a 12 s compile.
    private static func worthTaking(p0: OpaquePointer, p1: OpaquePointer,
                                    slot: Slot, sequence: Int, dimension: Int,
                                    heads: Int) -> Bool {
        let gpuHeads = heads - engineHeads
        guard gpuHeads > 0 else { return false }

        // Conformance tests route deliberately uneconomic shapes — a few heads
        // at S=64 — because they are checking arithmetic, not speed, and the
        // guard would rightly refuse them. A test escape; nothing should set it
        // in a render.
        if forceRoute { return true }

        func best(_ n: Int, _ body: () -> Void) -> Double {
            var best = Double.infinity
            for _ in 0 ..< n {
                let began = Date()
                body()
                best = min(best, Date().timeIntervalSince(began))
            }
            return best
        }

        // The surfaces hold zeros here. These kernels are dense and their cost
        // does not depend on the values, and zero carries no denormal penalty.
        let engine = best(3) {
            _ = h3_ane_attention_run_pair(p0, slot.q0, slot.k0, slot.v0, slot.y0,
                                          p1, slot.q1, slot.k1, slot.v1, slot.y1)
        }

        let scale = 1 / Float(dimension).squareRoot()
        func sdpa(_ n: Int) -> Double {
            let shape = [n, sequence, dimension]
            let q = MLXArray.zeros(shape, dtype: .bfloat16)
            let k = MLXArray.zeros(shape, dtype: .bfloat16)
            let v = MLXArray.zeros(shape, dtype: .bfloat16)
            MLX.eval(q, k, v)
            return best(3) {
                let out = MLXFast.scaledDotProductAttention(
                    queries: q.expandedDimensions(axis: 0),
                    keys: k.expandedDimensions(axis: 0),
                    values: v.expandedDimensions(axis: 0),
                    scale: scale, mask: nil)
                MLX.eval(out)
            }
        }
        let retained = sdpa(gpuHeads)
        let unrouted = sdpa(heads)

        // Simulate the decision rather than approximate it. Routing costs
        // `max(engine, retained)` because the two devices run concurrently, and
        // it is worth doing only if that beats the whole thing on the GPU. The
        // margin covers the upload and copy back, which the race excludes.
        //
        // Comparing the engine against the *retained* heads instead — which an
        // earlier version did — asks whether the engine beats the GPU per head,
        // which is the wrong question and refused the production shape: 309 ms
        // of engine against 336 of retained work looked like a loss, when the
        // real alternative was 392 ms of unrouted GPU.
        let usable = shouldRoute(engine: engine, retained: retained, unrouted: unrouted)
        let routed = max(engine, retained)
        if tracing {
            FileHandle.standardError.write(Data(String(
                format: "[ane-attn] calibrate S=%d D=%d H=%d: engine(%d) %.0f ms, gpu(%d) %.0f ms, routed %.0f vs unrouted %.0f -> %@\n",
                sequence, dimension, heads, engineHeads, engine * 1e3,
                gpuHeads, retained * 1e3, routed * 1e3, unrouted * 1e3,
                usable ? "route" : "DECLINE").utf8))
        }
        return usable
    }

    /// Bypasses the calibration guard. A `var` so the guard's own test can turn
    /// it off for one shape without a second process.
    nonisolated(unsafe) package static var forceRoute: Bool =
        ProcessInfo.processInfo.environment["H3_ANE_ATTENTION_FORCE"] == "1"

    private static let engineQueue = DispatchQueue(
        label: "h3.ane.attention", qos: .userInitiated)
    private static let lock = NSLock()
    nonisolated(unsafe) private static var sessions: [String: Session] = [:]
    nonisolated(unsafe) private static var refused: Set<String> = []

    private static func session(sequence: Int, dimension: Int, heads: Int) -> Session? {
        let key = "\(sequence)x\(dimension)x\(heads)"
        lock.lock(); defer { lock.unlock() }
        if let existing = sessions[key] { return existing }
        if refused.contains(key) { return nil }
        guard let made = Session(sequence: sequence, dimension: dimension, heads: heads) else {
            refused.insert(key)
            return nil
        }
        sessions[key] = made
        return made
    }

    private static func upload(_ array: MLXArray, to tensor: OpaquePointer,
                               rows: Int, width: Int) -> Bool {
        array.asData(access: .noCopyIfContiguous).data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return false }
            return h3_ane_tensor_write(tensor, base, Int32(rows), Int32(width))
        }
    }

    package func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                        scale: Float, mask: MLXArray?,
                        context: AttentionContext) -> MLXArray? {
        guard mask == nil else { return Self.decline("a mask is present") }
        guard queries.ndim == 3 else { return Self.decline("ndim \(queries.ndim)") }
        guard keys.shape == queries.shape, values.shape == queries.shape else {
            return Self.decline("q/k/v shapes differ")
        }
        guard queries.dim(0) > Self.engineHeads else {
            return Self.decline("only \(queries.dim(0)) heads")
        }
        let heads = queries.dim(0), sequence = queries.dim(1), dimension = queries.dim(2)
        guard dimension == 128 else { return Self.decline("head dim \(dimension)") }
        guard let session = Self.session(sequence: sequence, dimension: dimension,
                                         heads: heads) else {
            return Self.decline("no usable session for \(sequence)x\(dimension)x\(heads)")
        }
        guard session.take() else { return Self.decline("session busy or poisoned") }

        let split = heads - Self.engineHeads
        let r0 = split ..< (split + Self.headsPerDie)
        let r1 = r0.upperBound ..< heads
        let q0 = MLX.contiguous((queries[r0] * scale).asType(.float16))
        let k0 = MLX.contiguous(keys[r0].asType(.float16))
        let v0 = MLX.contiguous(values[r0].asType(.float16))
        let q1 = MLX.contiguous((queries[r1] * scale).asType(.float16))
        let k1 = MLX.contiguous(keys[r1].asType(.float16))
        let v1 = MLX.contiguous(values[r1].asType(.float16))
        MLX.eval([q0, k0, v0, q1, k1, v1])

        let rows = Self.headsPerDie * sequence
        let slot = session.slot
        guard Self.upload(q0, to: slot.q0, rows: rows, width: dimension),
              Self.upload(k0, to: slot.k0, rows: rows, width: dimension),
              Self.upload(v0, to: slot.v0, rows: rows, width: dimension),
              Self.upload(q1, to: slot.q1, rows: rows, width: dimension),
              Self.upload(k1, to: slot.k1, rows: rows, width: dimension),
              Self.upload(v1, to: slot.v1, rows: rows, width: dimension)
        else {
            session.give(success: true)
            return Self.decline("upload failed")
        }

        let job = Job()
        Self.engineQueue.async {
            job.settle(h3_ane_attention_run_pair(
                session.p0, slot.q0, slot.k0, slot.v0, slot.y0,
                session.p1, slot.q1, slot.k1, slot.v1, slot.y1))
        }

        // Queue the remaining complete heads only after the ANE inputs are
        // materialised. This preserves true GPU/ANE overlap: no MLX lock is
        // held by the thread blocked in the private runtime.
        let gpu = Stream.withNewDefaultStream(device: .gpu) {
            let output = MLXFast.scaledDotProductAttention(
                queries: queries[0 ..< split].expandedDimensions(axis: 0),
                keys: keys[0 ..< split].expandedDimensions(axis: 0),
                values: values[0 ..< split].expandedDimensions(axis: 0),
                scale: scale, mask: nil).squeezed(axis: 0)
            MLX.asyncEval(output)
            return output
        }

        let success = job.wait()
        guard success else {
            session.give(success: false)
            return Self.decline("evaluate failed")
        }

        func adopt(_ tensor: OpaquePointer) -> MLXArray {
            MLXArray(rawPointer: h3_ane_tensor_ptr(tensor)!,
                     [Self.headsPerDie, sequence, dimension], dtype: .float16) { }
                .asType(queries.dtype)
        }
        let a0 = adopt(slot.y0), a1 = adopt(slot.y1)
        MLX.eval(a0, a1)
        session.give(success: true)
        Self.countRouted()
        return MLX.concatenated([gpu, a0, a1], axis: 0)
    }
}
