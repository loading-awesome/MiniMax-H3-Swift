// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import H3Foundation
@testable import H3Recipes

/// `--width`/`--height` stays open, because the only verified configuration
/// this port has — 864x480 — is no longer a rung and must keep working. The
/// guard is therefore advice, and these pin that it advises when it should and
/// stays quiet when it should not.
@Suite("off-ladder shapes are advised, not refused")
struct TileAdviceTests {

    @Test("the tiling rule has one definition")
    func ruleIsShared() {
        // Three callers need this rule and each used to carry a copy. These are
        // the boundaries: the largest length a count covers, and one past it.
        for n in 1...6 {
            let top = H3Tiling.largestLength(tiles: n)
            #expect(H3Tiling.tiles(length: top) == n, "\(top) should need \(n)")
            #expect(H3Tiling.tiles(length: top + 1) == n + 1, "\(top + 1) should need \(n + 1)")
        }
    }

    @Test("the verified shape gets the advice this exists for")
    func verifiedShapeIsAdvised() throws {
        // 864x480 against 832x448 is the motivating case, and it sits at 89.9%
        // of the pixels — a 90% floor would have missed it.
        let advice = H3Ladder.tileAdvice(width: 864, height: 480)
        #expect(advice.tiles == 15)
        let better = try #require(advice.better)
        #expect((better.width, better.height) == (832, 448))
        #expect(better.tiles == 8)
    }

    @Test("advice is offered in portrait too")
    func portraitIsAdvised() throws {
        let advice = H3Ladder.tileAdvice(width: 480, height: 864)
        let better = try #require(advice.better)
        #expect((better.width, better.height) == (448, 832))
    }

    @Test("a rung is never told to be something else")
    func rungsAreQuiet() {
        // Every rung is tile-minimal for its size by construction, so any advice
        // here would mean the ladder and this function disagree.
        for rung in H3Ladder.rungs {
            #expect(H3Ladder.tileAdvice(width: rung.width, height: rung.height).better == nil,
                    "\(rung.megapixels) MP landscape")
            #expect(H3Ladder.tileAdvice(width: rung.height, height: rung.width).better == nil,
                    "\(rung.megapixels) MP portrait")
        }
    }

    @Test("a much smaller rung is not offered as an improvement")
    func doesNotSuggestShrinking() {
        // "Smaller is cheaper" is not advice. A shape already near the bottom
        // of the ladder has nothing to be told.
        #expect(H3Ladder.tileAdvice(width: 608, height: 352).better == nil)
    }
}
