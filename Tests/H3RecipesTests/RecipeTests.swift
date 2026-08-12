// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import H3Foundation
@testable import H3Recipes

@Suite("recipes")
struct RecipeTests {

    @Test("every registered recipe is on the 32-pixel grid")
    func recipesAreOnGrid() {
        // The VAE downsamples by 16 and the DiT patchifies by 2. A recipe that
        // is off the grid would be refused at render time by a rule the recipe
        // itself was supposed to satisfy, which makes it dead weight in a menu.
        for (id, recipe) in H3RecipeRegistry.all {
            #expect(recipe.targetWidth % 32 == 0, "\(id.rawValue) width")
            #expect(recipe.targetHeight % 32 == 0, "\(id.rawValue) height")
        }
    }

    @Test("2K recipes are upscale targets, not base renders")
    func twoKIsTiered() {
        // 2K is the output of In-Context Regeneration, which is unreleased and
        // unported. Tiering these rather than deleting them keeps them visible
        // with a status, so "unavailable" is something you can read rather than
        // something you discover.
        for (id, recipe) in H3RecipeRegistry.all where id.rawValue.contains("2k") {
            #expect(recipe.tier == .upscaleTarget, "\(id.rawValue)")
        }
        // Everything that is not 2K is a size the DiT samples directly. Written
        // as "not 2k" rather than as a name match on "768p": the ladder ids
        // carry neither substring, so a name match would have silently stopped
        // checking the 28 entries that make up most of the menu.
        for (id, recipe) in H3RecipeRegistry.all where !id.rawValue.contains("2k") {
            #expect(recipe.tier == .baseRender, "\(id.rawValue)")
        }
    }

    @Test("the ladder is the recommended-size table, unedited")
    func ladderMatchesTheTable() {
        // Pinned literally, because the ladder's whole value is that it is a
        // table someone else measured. A rung that drifted would still be on
        // the 32-grid, still render, and no longer be a recommended size.
        let expected: [(Double, Int, Int)] = [
            (0.2, 608, 352), (0.3, 736, 416), (0.4, 864, 480), (0.5, 960, 544),
            (0.6, 1056, 608), (0.7, 1152, 640), (0.8, 1216, 672), (0.9, 1280, 736),
            (0.98, 1344, 768), (1.0, 1376, 768), (1.2, 1504, 832), (1.5, 1664, 928),
            (1.8, 1824, 1024), (2.0, 1920, 1088),
        ]
        #expect(H3Ladder.rungs.count == expected.count)
        for (rung, want) in zip(H3Ladder.rungs, expected) {
            #expect(rung.megapixels == want.0)
            #expect(rung.width == want.1, "\(rung.landscape.rawValue) width")
            #expect(rung.height == want.2, "\(rung.landscape.rawValue) height")
        }
    }

    @Test("every rung is registered, in both orientations, as a transpose")
    func ladderIsRegisteredBothWays() throws {
        for rung in H3Ladder.rungs {
            let land = try #require(H3RecipeRegistry.all[rung.landscape])
            let port = try #require(H3RecipeRegistry.all[rung.portrait])
            #expect(land.targetWidth == rung.width)
            #expect(land.targetHeight == rung.height)
            // The transpose, not a second table: a portrait entry that drifted
            // from its landscape rung would be a size with no source behind it.
            #expect(port.targetWidth == land.targetHeight, "\(rung.portrait.rawValue)")
            #expect(port.targetHeight == land.targetWidth, "\(rung.portrait.rawValue)")
        }
    }

    @Test("the ladder climbs, and the verified shape is on it")
    func ladderClimbsThroughTheVerifiedShape() {
        // Strictly increasing in pixels catches a transposed or mistyped rung
        // that the 32-grid check and the per-row pin would both accept.
        for (a, b) in zip(H3Ladder.rungs, H3Ladder.rungs.dropFirst()) {
            #expect(a.width * a.height < b.width * b.height,
                    "\(a.landscape.rawValue) -> \(b.landscape.rawValue)")
        }
        // The one shape with a measured tolerance behind it has to be nameable.
        // Before the ladder it was not: the tiers could not express 864x480 and
        // it was reachable only through --width/--height.
        #expect(H3Ladder.verified.width == H3RenderPolicy.verifiedWidth)
        #expect(H3Ladder.verified.height == H3RenderPolicy.verifiedHeight)
        #expect(H3Ladder.verified.landscape == .h3_16x9_0p4mp)
    }

    @Test("exactly the top two rungs are past the render-time ceiling")
    func ceilingFallsBetweenTheTopTwoRungs() {
        // Registered-but-refused is a deliberate state, so where the boundary
        // sits is part of the contract rather than an accident of the table.
        // If a throughput or config change moves it, the menu's top should be
        // reconsidered rather than silently growing or shrinking.
        let steps = H3RenderPolicy.verifiedSteps
        let frames = H3RenderPolicy.verifiedFrames
        for rung in H3Ladder.rungs {
            let geometry = LatentGeometry(width: rung.width, height: rung.height, length: frames)
            let tokens = geometry.videoTokens + geometry.audioTokens
            let over = H3RenderPolicy.estimatedSeconds(tokens: tokens, steps: steps)
                     > H3RenderPolicy.maxEstimatedSeconds
            #expect(over == (rung.megapixels >= 1.8),
                    "\(rung.landscape.rawValue) at \(rung.width)x\(rung.height)")
        }
    }

    @Test("an upscale-target recipe refuses to be rendered directly")
    func upscaleTargetsRefuse() throws {
        let twoK = try #require(H3RecipeRegistry.all[.h3_2k_16_9])
        #expect(throws: (any Error).self) {
            try twoK.validate(width: twoK.targetWidth, height: twoK.targetHeight,
                              duration: 5, steps: 20)
        }
    }

    @Test("a base recipe validates at its own shape")
    func baseRecipesValidate() throws {
        let base = try #require(H3RecipeRegistry.all[.h3_16x9_0p4mp])
        #expect(throws: Never.self) {
            try base.validate(width: base.targetWidth, height: base.targetHeight,
                              duration: 5, steps: 20)
        }
    }

    @Test("the catalogue prices every registered recipe, and marks the verified one")
    func cataloguePricesEverything() throws {
        let catalogue = H3RecipeCatalogue()
        #expect(catalogue.rows.count == H3RecipeRegistry.all.count)
        // Exactly one entry can be the verified shape. Two would mean the
        // portrait transpose was being credited with the landscape shape's
        // measurements, which is the kind of thing a listing makes look true.
        let verified = catalogue.rows.filter { $0.status == .verified }
        #expect(verified.count == 1)
        #expect(verified.first?.id == .h3_16x9_0p4mp)
        for row in catalogue.rows where row.id.rawValue.contains("2k") {
            #expect(row.status == .upscaleTarget, "\(row.id.rawValue)")
        }
    }

    @Test("the catalogue quotes the price the policy would refuse at")
    func catalogueAgreesWithThePolicy() {
        // The listing and the refusal must not be able to disagree: a size
        // advertised at one price and refused at another is worse than no
        // listing, because the number came from somewhere trustworthy-looking.
        func refusedOnTime(_ row: H3RecipeCatalogue.Row) -> Bool {
            H3RenderPolicy.check(
                width: row.width, height: row.height,
                frameCount: H3RenderPolicy.verifiedFrames,
                steps: H3RenderPolicy.verifiedSteps, tokens: row.tokens)
                .contains { $0.rule == "estimated render time" }
        }
        let rows = H3RecipeCatalogue().rows
        for row in rows where row.status != .upscaleTarget {
            #expect(refusedOnTime(row) == (row.status == .overCeiling), "\(row.id.rawValue)")
        }
        // Precedence, pinned: every 2K size is *also* past the ceiling, and is
        // still reported as no-route. "Too slow" would be true and misleading —
        // it implies a faster machine or fewer steps would get you there, and
        // nothing about this size is reachable at any speed.
        let twoK = rows.filter { $0.status == .upscaleTarget }
        #expect(twoK.allSatisfy(refusedOnTime))
        #expect(!twoK.isEmpty)
    }

    @Test("a longer duration reprices the menu and moves the ceiling down it")
    func longerRendersPushMoreRungsOverTheCeiling() {
        // The ceiling is not a property of a size. Listing it as one is how a
        // caller ends up trusting a 5 s status while asking for 10 s.
        let short = H3RecipeCatalogue(seconds: 5)
        let long = H3RecipeCatalogue(seconds: 10)
        #expect(long.frameCount > short.frameCount)
        let overShort = short.rows.filter { $0.status == .overCeiling }.count
        let overLong = long.rows.filter { $0.status == .overCeiling }.count
        #expect(overLong > overShort)
    }

    @Test("filtering keeps the comparison to the verified shape")
    func filteredListingKeepsItsBaseline() {
        // `--aspect 2k` filters the verified row out of the listing. The
        // relative column has to survive that, because "how much worse than
        // the shape we trust" is the reason to narrow the list in the first
        // place.
        let catalogue = H3RecipeCatalogue()
        let twoK = catalogue.rows(matching: "2k")
        #expect(twoK.count == 6)
        #expect(!twoK.contains { $0.status == .verified })
        let report = catalogue.report(twoK)
        #expect(report.contains("x"))
        #expect(!report.contains("  -  "))
    }

    @Test("the estimate is quoted before a refusal, not after the render")
    func policyQuotesItsEstimate() {
        // The whole point of the policy is that the cost is knowable in the
        // first second rather than at minute twenty.
        let line = H3RenderPolicy.estimateLine(tokens: 133_200, steps: 20)
        #expect(!line.isEmpty)
        let big = H3RenderPolicy.estimatedSeconds(tokens: 133_200, steps: 20)
        let verified = H3RenderPolicy.estimatedSeconds(tokens: 15_750, steps: 20)
        // Attention is quadratic in sequence length, so 8.5x the tokens must
        // cost far more than 8.5x the time — an estimator that scaled linearly
        // would call 2K a twenty-minute job.
        #expect(big > verified * 8.5)
    }
}
