// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import H3Hardware
import H3Foundation
@testable import H3Attention

/// A second backend, which is the whole point: with only SDPA compiled in, a
/// resolver that ignores its argument and returns `SDPABackend()` looks correct.
///
/// It declines every call, so a Selection that carries it is trivially
/// distinguishable from one carrying SDPA — by behaviour, not by a label the
/// resolver could copy across while handing back the wrong object.
private struct DecliningBackend: H3AttentionBackend {
    static let identifier = "declining"
    static let equivalenceClass: Float = 0.25
    static let materialisesScores = false
    static let prefersMortonOrder = true
    static func isAvailable(on machine: Machine) -> Bool { true }
    init() {}
    func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                scale: Float, mask: MLXArray?, context: AttentionContext) -> MLXArray? { nil }
}

private struct UnavailableBackend: H3AttentionBackend {
    static let identifier = "unavailable"
    static let equivalenceClass: Float = 0.5
    static let materialisesScores = false
    static let prefersMortonOrder = false
    static func isAvailable(on machine: Machine) -> Bool { false }
    init() {}
    func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                scale: Float, mask: MLXArray?, context: AttentionContext) -> MLXArray? { nil }
}

@Suite("attention registry")
struct AttentionRegistryTests {

    static let machine = Machine(model: "Mac15,14", chip: "Apple M3 Ultra",
                                 memoryBytes: 275 * 1_000_000_000, cores: 28,
                                 isPortable: false)

    /// `MLXArray` defines `==` elementwise, returning an array rather than a
    /// Bool, so the usual `#expect(x == nil)` does not mean what it reads like.
    private static func declined(_ x: MLXArray?) -> Bool {
        if case .none = x { return true }
        return false
    }

    @Test("the resolved backend is the one that was named")
    func backendIsWhatWasNamed() throws {
        // The regression this exists for: `make` reported `t.identifier` and
        // `t.equivalenceClass` while constructing `SDPABackend()` unconditionally.
        // Every field of the Selection was right and the object was wrong.
        let s = try AttentionRegistry.resolve(
            requested: "declining", machine: Self.machine, ordering: nil,
            backends: [SDPABackend.self, DecliningBackend.self])

        #expect(s.identifier == "declining")
        #expect(s.equivalenceClass == 0.25)
        #expect(s.materialisesScores == false)
        // Behaviour, not metadata: SDPA never declines, this one always does.
        let z = MLXArray.zeros([2, 4, 8], dtype: .float32)
        let ctx = AttentionContext(blockIndex: 0, blockCount: 50, scheduleProgress: 0.5,
                                   sequenceLength: 4, videoSpan: nil)
        #expect(Self.declined(s.backend.attend(queries: z, keys: z, values: z,
                                               scale: 1, mask: nil, context: ctx)))
    }

    @Test("a backend that prefers Morton order gets it, and one that does not does not")
    func orderingFollowsTheBackend() throws {
        // Ordering is a pipeline transform chosen once from the backend's stated
        // preference, so it travels with the backend rather than with a flag.
        let sparse = try AttentionRegistry.resolve(
            requested: "declining", machine: Self.machine, ordering: nil,
            backends: [DecliningBackend.self])
        #expect(sparse.ordering == TokenOrdering.mortonPerFrame)

        let dense = try AttentionRegistry.resolve(
            requested: "sdpa", machine: Self.machine, ordering: nil,
            backends: [SDPABackend.self])
        #expect(dense.ordering == .none)

        // An explicit request still wins over the preference.
        let forced = try AttentionRegistry.resolve(
            requested: "declining", machine: Self.machine, ordering: TokenOrdering.none,
            backends: [DecliningBackend.self])
        #expect(forced.ordering == .none)
    }

    @Test("auto takes the first available backend, not the first registered")
    func autoSkipsUnavailable() throws {
        let s = try AttentionRegistry.resolve(
            requested: "auto", machine: Self.machine, ordering: nil,
            backends: [UnavailableBackend.self, DecliningBackend.self])
        #expect(s.identifier == "declining")
    }

    @Test("an unknown or unavailable backend is refused, never silently downgraded")
    func refusesRatherThanFallsBack() {
        // A silent fallback would mean numerics the caller did not ask for, with
        // no record of the substitution — the failure this whole seam is shaped
        // against.
        #expect(throws: H3Error.self) {
            try AttentionRegistry.resolve(requested: "sol", machine: Self.machine,
                                          ordering: nil, backends: [SDPABackend.self])
        }
        #expect(throws: H3Error.self) {
            try AttentionRegistry.resolve(requested: "unavailable", machine: Self.machine,
                                          ordering: nil,
                                          backends: [UnavailableBackend.self])
        }
    }

    @Test("SDPA answers every call")
    func sdpaNeverDeclines() {
        // The fallback must be total: every decline path in the block loop ends
        // here, so a nil from SDPA would have nowhere left to go.
        let q = MLXArray.zeros([2, 16, 8], dtype: .float32)
        let ctx = AttentionContext(blockIndex: 0, blockCount: 50, scheduleProgress: 0,
                                   sequenceLength: 16, videoSpan: 4 ..< 16)
        let out = SDPABackend().attend(queries: q, keys: q, values: q,
                                       scale: 0.35, mask: nil, context: ctx)
        #expect(out != nil)
        #expect(out?.shape == [2, 16, 8])
    }
}

@Suite("metal library")
struct MetalLibraryTests {

    @Test("MLX's Metal kernels are where MLX will look for them")
    func metallibIsPresent() throws {
        // Not a tautology: `swift build` does not produce this file, and without
        // `Scripts/bootstrap-metal.sh` every MLX-linked test in this package
        // dies on its first GPU op with an untyped C++ error carrying no path.
        // This turns that into one legible failure. See H3Hardware.MetalLibrary.
        try MetalLibrary.preflight()
        #expect(MetalLibrary.locate() != nil)
    }

    @Test("and it is found beside the binary, not via the working directory")
    func foundIndependentlyOfCwd() throws {
        // The experimental tree ran for months on the cwd fallback — a metallib
        // committed at the repo root, every render launched from there. That
        // works until somebody installs the binary and runs it from home.
        try MetalLibrary.preflight()
        #expect(MetalLibrary.locatedOnlyViaWorkingDirectory() == false)
    }
}
