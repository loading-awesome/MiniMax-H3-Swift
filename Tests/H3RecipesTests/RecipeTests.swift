// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
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
        for (id, recipe) in H3RecipeRegistry.all where id.rawValue.contains("768p") {
            #expect(recipe.tier == .baseRender, "\(id.rawValue)")
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
        let base = try #require(H3RecipeRegistry.all[.h3_768p_16_9])
        #expect(throws: Never.self) {
            try base.validate(width: base.targetWidth, height: base.targetHeight,
                              duration: 5, steps: 20)
        }
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
