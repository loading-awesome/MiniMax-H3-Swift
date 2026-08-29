// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
import H3Foundation
import H3Recipes
@testable import H3Pipeline

/// `--megapixels` names the same rungs `--recipe` does, spelled the way the
/// choice is usually thought about. These pin that the two agree, and that a
/// size input which loses precedence is never discarded in silence.
@Suite("megapixels resolves to the ladder")
struct MegapixelShapeTests {

    private func base(_ edit: (inout RenderRequest) -> Void = { _ in }) -> RenderRequest {
        var r = RenderRequest(prompt: "p", videoOutput: URL(fileURLWithPath: "/tmp/o.mp4"))
        edit(&r)
        return r
    }

    @Test("a rung and its recipe name the same shape, both ways up")
    func agreesWithRecipe() throws {
        for rung in H3Ladder.rungs {
            let land = try RenderRequest.ladderShape(rung.megapixels, .r16x9)
            #expect(land == (rung.width, rung.height), "\(rung.megapixels) MP landscape")
            let port = try RenderRequest.ladderShape(rung.megapixels, .r9x16)
            #expect(port == (rung.height, rung.width), "\(rung.megapixels) MP portrait")
        }
    }

    @Test("a value between rungs is refused, not rounded")
    func refusesBetweenRungs() {
        // Rounding would hand back a shape the caller did not ask for and
        // could not see they had not got.
        #expect(throws: H3Error.self) { try RenderRequest.ladderShape(0.45, .r16x9) }
    }

    @Test("a ratio with no ladder is refused")
    func refusesRatioWithoutLadder() {
        #expect(throws: H3Error.self) { try RenderRequest.ladderShape(0.4, .r1x1) }
    }

    @Test("a losing size input that disagrees is refused, never ignored")
    func refusesConflicts() {
        #expect(throws: H3Error.self) {
            try base { $0.megapixels = 2.0; $0.recipe = .h3_16x9_0p4mp }.validate()
        }
        #expect(throws: H3Error.self) {
            try base { $0.megapixels = 2.0; $0.width = 832; $0.height = 448 }.validate()
        }
    }

    @Test("inputs that agree are accepted")
    func acceptsAgreement() throws {
        try base { $0.megapixels = 0.4; $0.recipe = .h3_16x9_0p4mp }.validate()
        try base { $0.megapixels = 0.4; $0.width = 832; $0.height = 448 }.validate()
    }

    @Test("the 768p fallback is the ladder's 0.98 rung, not a shape of its own")
    func fallbackTracksTheLadder() throws {
        // This drifted once already: the ladder moved 0.98 MP from 1344x768 to
        // 1376x768 and the fallback stayed, so the default shape was one no
        // rung named.
        let rung = try #require(H3Ladder.rungs.first { $0.megapixels == 0.98 })
        #expect(try base().dimensions() == (rung.width, rung.height))
        #expect(try base { $0.aspectRatio = .r9x16 }.dimensions() == (rung.height, rung.width))
    }
}
