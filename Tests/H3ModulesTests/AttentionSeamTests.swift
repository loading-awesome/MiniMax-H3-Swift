// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import H3Attention
import H3Hardware
@testable import H3Modules

/// Records what the block stack handed it, and refuses the call.
///
/// Refusing is what makes it safe to assert on: the numerics stay dense, so any
/// difference between a run with this backend and a run without one is the seam
/// misbehaving rather than an approximation doing its job.
private final class Recorder: @unchecked Sendable {
    var calls: [(shape: [Int], context: AttentionContext)] = []
}

private struct RecordingBackend: H3AttentionBackend {
    static let identifier = "recording"
    static let equivalenceClass: Float = 0
    static let materialisesScores = false
    static let prefersMortonOrder = false
    static func isAvailable(on machine: Machine) -> Bool { true }

    /// Per instance, not static. swift-testing runs these concurrently, and a
    /// shared recorder made two tests observe each other's calls — the count
    /// assertion below read 2.
    let log: Recorder
    init() { self.log = Recorder() }
    init(log: Recorder) { self.log = log }

    func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                scale: Float, mask: MLXArray?, context: AttentionContext) -> MLXArray? {
        log.calls.append((queries.shape, context))
        return nil
    }
}

/// Answers with a value no dense path could produce, so "was the backend
/// actually used" is a question about output rather than about a call count.
private struct ConstantBackend: H3AttentionBackend {
    static let identifier = "constant"
    static let equivalenceClass: Float = 99
    static let materialisesScores = false
    static let prefersMortonOrder = false
    static func isAvailable(on machine: Machine) -> Bool { true }
    init() {}

    func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                scale: Float, mask: MLXArray?, context: AttentionContext) -> MLXArray? {
        MLXArray.ones(queries.shape, dtype: queries.dtype) * MLXArray(Float(7))
    }
}

@Suite("attention seam")
struct AttentionSeamTests {

    static let heads = 2
    static let headDim = 4
    static let hidden = 8          // heads * headDim
    static let tokens = 16

    /// Deterministic, and not constant: a backend that ignores its inputs would
    /// pass against a constant tensor.
    static func ramp(_ shape: [Int], scale: Float = 0.01) -> MLXArray {
        let n = shape.reduce(1, *)
        let v = (0 ..< n).map { Float($0 % 17) * scale - 0.08 }
        return MLXArray(v, shape)
    }

    static func layer(backend: any H3AttentionBackend = SDPABackend()) -> AttentionLayer {
        AttentionLayer(qkvWeight: ramp([3 * hidden, hidden], scale: 0.013),
                       outWeight: ramp([hidden, hidden], scale: 0.021),
                       qNormWeight: MLXArray.ones([headDim]),
                       kNormWeight: MLXArray.ones([headDim]),
                       heads: heads, headDim: headDim, eps: 1e-6,
                       fp32Attention: false, backend: backend)
    }

    static let context = AttentionContext(blockIndex: 12, blockCount: 50,
                                          scheduleProgress: 0.4,
                                          sequenceLength: tokens, videoSpan: 4 ..< tokens)

    @Test("a declining backend gives bit-identical output to no backend at all")
    func declineIsExactlyDense() {
        // The fallback carries the whole parity claim: every schedule window and
        // dense-block exclusion is expressed as a decline, so if declining were
        // merely *close* to dense, those exclusions would themselves be an
        // approximation.
        let x = Self.ramp([Self.tokens, Self.hidden], scale: 0.03)
        let dense = Self.layer()(x, ropeTable: nil)
        let declined = Self.layer(backend: RecordingBackend())(x, ropeTable: nil,
                                                               context: Self.context)
        let maxDiff = MLX.abs(dense - declined).max().item(Float.self)
        #expect(maxDiff == 0)
    }

    @Test("the backend is actually reached, with the block's own context")
    func dispatchHappens() {
        // Without this, every other test here passes on a seam that is never
        // called — which is precisely the state this repo was in: an
        // AttentionContext designed, documented, and wired to nothing.
        let log = Recorder()
        let x = Self.ramp([Self.tokens, Self.hidden], scale: 0.03)
        _ = Self.layer(backend: RecordingBackend(log: log))(x, ropeTable: nil,
                                                            context: Self.context)

        #expect(log.calls.count == 1)
        let call = log.calls[0]
        // [heads, S, headDim] — no batch axis, as the protocol states.
        #expect(call.shape == [Self.heads, Self.tokens, Self.headDim])
        #expect(call.context.blockIndex == 12)
        #expect(call.context.videoSpan == 4 ..< Self.tokens)
    }

    @Test("a backend that answers replaces the dense result")
    func answerIsUsed() {
        let x = Self.ramp([Self.tokens, Self.hidden], scale: 0.03)
        let dense = Self.layer()(x, ropeTable: nil)
        let constant = Self.layer(backend: ConstantBackend())(x, ropeTable: nil,
                                                              context: Self.context)
        #expect(MLX.abs(dense - constant).max().item(Float.self) > 0.1)
    }

    @Test("without a context the backend is never consulted")
    func noContextMeansDense() {
        // The oracles and any single forward pass call with no context. A sparse
        // backend that does not know its block index or its schedule position
        // cannot honour its own dense warm-up, and guessing is worse than
        // declining.
        let log = Recorder()
        let x = Self.ramp([Self.tokens, Self.hidden], scale: 0.03)
        _ = Self.layer(backend: RecordingBackend(log: log))(x, ropeTable: nil)
        #expect(log.calls.isEmpty)
    }

    @Test("the batched text path stays dense")
    func batchedPathIsDense() {
        // The token refiner runs `[B, S, hidden]` over a few hundred text rows.
        // The protocol has no batch axis — H3 is batch-1 by construction — and
        // a few hundred rows against 15,749 is not where sparsity pays.
        let log = Recorder()
        let x = Self.ramp([1, Self.tokens, Self.hidden], scale: 0.03)
        let out = Self.layer(backend: RecordingBackend(log: log))(x, ropeTable: nil,
                                                                  context: Self.context)
        #expect(log.calls.isEmpty)
        #expect(out.shape == [1, Self.tokens, Self.hidden])
    }
}
