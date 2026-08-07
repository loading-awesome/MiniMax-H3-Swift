// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import H3Foundation
@testable import MiniMaxH3

@Suite("benchmark emitter")
struct BenchmarkEmitterTests {

    /// Walks up from this source file to the package root.
    private static func packageRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 3 { url.deleteLastPathComponent() }
        return url
    }

    @Test("the pinned MLX version matches Package.resolved")
    func pinnedMLXVersionMatchesPackageResolved() throws {
        // `BenchmarkEmitter.mlxSwiftVersion` is a literal because nothing
        // reports it at runtime, and a literal describing a dependency is a
        // comment that goes stale the first time the dependency moves. Every
        // record written after that would carry a wrong environment field —
        // and wrong provenance is worse than absent provenance, because it
        // reads as an answer.
        let resolved = Self.packageRoot().appendingPathComponent("Package.resolved")
        let text = try String(contentsOf: resolved, encoding: .utf8)
        struct Resolved: Decodable {
            struct Pin: Decodable {
                let identity: String
                struct State: Decodable { let version: String? }
                let state: State
            }
            let pins: [Pin]
        }
        let pins = try JSONDecoder().decode(Resolved.self, from: Data(text.utf8)).pins
        let mlx = try #require(pins.first { $0.identity == "mlx-swift" },
                               "mlx-swift is no longer a direct pin")
        let complaint = "Package.resolved pins mlx-swift \(mlx.state.version ?? "?") but "
            + "BenchmarkEmitter says \(BenchmarkEmitter.mlxSwiftVersion) — update the literal"
        #expect(mlx.state.version == BenchmarkEmitter.mlxSwiftVersion,
                Comment(rawValue: complaint))
    }

    @Test("the arm name says what was on")
    func armNameDescribesConfiguration() {
        // Derived rather than sequential: `cached-0.100-sdpa` can be read back
        // into a configuration, `run-3` cannot. The probe no longer appears in
        // the name because it is no longer a choice — the whole-sequence arm
        // lost its A/B and the flag went with it.
        #expect(BenchmarkEmitter.armName(request: Self.request(cache: 0),
                                         result: Self.result(backend: "sdpa")) == "dense-sdpa")
        #expect(BenchmarkEmitter.armName(request: Self.request(cache: 0.1),
                                         result: Self.result(backend: "sdpa"))
                == "cached-0.100-sdpa")
    }

    @Test("H3_ overrides are captured but the arm label is not one of them")
    func overridesExcludeTheLabel() {
        // `H3_BENCH_ARM` names the run; it is not part of what ran. Leaving it
        // in `overrides` would make two identical configurations compare as
        // different arms of an experiment.
        let overrides = BenchmarkEmitter.environmentOverrides()
        #expect(overrides["H3_BENCH_ARM"] == nil)
        #expect(overrides.allSatisfy { $0.key.hasPrefix("H3_") })
    }


    // MARK: fixtures

    private static func request(cache: Double) -> RenderRequest {
        var r = RenderRequest(prompt: "a red kite", videoOutput: URL(fileURLWithPath: "/tmp/x.mp4"))
        r.cacheThreshold = cache
        return r
    }

    private static func result(backend: String) -> RenderResult {
        var r = RenderResult(video: URL(fileURLWithPath: "/tmp/x.mp4"), audio: nil,
                             frameCount: 1, width: 864, height: 480, seconds: 5,
                             timings: .init(), cacheSummary: nil)
        r.attentionBackend = backend
        return r
    }
}
