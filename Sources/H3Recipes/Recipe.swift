// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import H3Foundation

public enum H3RecipeID: String, Sendable, Codable, CaseIterable {
    case h3_2k_16_9
    case h3_2k_9_16
    case h3_2k_21_9
    case h3_2k_4_3
    case h3_2k_3_4
    case h3_2k_1_1

    // The 16:9 megapixel ladder and its transpose. See `H3Ladder`.
    //
    // These replace `h3_768p_16_9` and `h3_768p_9_16`, which named exactly one
    // rung each — 1344x768 and 768x1344, the 0.98 MP row. The single-size
    // spelling was the reason 864x480, the shape every golden and every
    // measured tolerance uses, could not be named at all and had to be reached
    // through `--width`/`--height`.
    case h3_16x9_0p2mp, h3_9x16_0p2mp
    case h3_16x9_0p3mp, h3_9x16_0p3mp
    case h3_16x9_0p4mp, h3_9x16_0p4mp
    case h3_16x9_0p5mp, h3_9x16_0p5mp
    case h3_16x9_0p6mp, h3_9x16_0p6mp
    case h3_16x9_0p7mp, h3_9x16_0p7mp
    case h3_16x9_0p8mp, h3_9x16_0p8mp
    case h3_16x9_0p9mp, h3_9x16_0p9mp
    case h3_16x9_0p98mp, h3_9x16_0p98mp
    case h3_16x9_1p0mp, h3_9x16_1p0mp
    case h3_16x9_1p2mp, h3_9x16_1p2mp
    case h3_16x9_1p5mp, h3_9x16_1p5mp
    case h3_16x9_1p8mp, h3_9x16_1p8mp
    case h3_16x9_2p0mp, h3_9x16_2p0mp

    case h3_768p_21_9
    case h3_768p_4_3
    case h3_768p_3_4
    case h3_768p_1_1
}

/// One rung of the megapixel ladder: a landscape size and its transpose.
package struct H3LadderRung: Sendable, Equatable {
    /// The nominal target this rung was derived from, not what it delivers.
    /// Snapping to 32 overshoots: the 0.4 rung is 414,720 pixels, and the
    /// 0.98 and 1.0 rungs differ by 32 pixels of width and share a height.
    package let megapixels: Double
    /// Landscape orientation. The portrait recipe is this transposed.
    package let width: Int
    package let height: Int
    package let landscape: H3RecipeID
    package let portrait: H3RecipeID
}

/// The recommended output sizes, as one table.
///
/// **Every rung is a multiple of 32 on both axes and none of them is exactly
/// 16:9.** The VAE downsamples by 16 and the DiT patchifies by 2, so 32 is the
/// grid a render has to land on; forcing 1.778 onto that grid is not possible
/// at most sizes, and the ladder takes the nearest grid point instead. The
/// spread is 1.724 to 1.857.
///
/// **The rungs are chosen for the decoder's tile grid, not only for pixels.**
/// The video VAE cuts fixed 256-pixel tiles with a 64-pixel minimum overlap, so
/// the tile count steps rather than scales and a dimension just past a step
/// pays for a whole row or column of tiles it barely uses. Each rung is the
/// fewest-tile shape within 10% of its megapixel target and inside the aspect
/// band above, subject to being strictly larger than the rung below it. The
/// clearest case is 0.4 MP: 864x480 needed 15 tiles where 832x448 needs 8, so
/// the rung moved and the decode halved for 4% fewer pixels.
///
/// **This retired 864x480, which was the verified shape.** Its parity gate and
/// lip-sync runs do not transfer to 832x448, and `docs/SHAPES.md` records that
/// the harness has to be re-run rather than pretending the verification moved
/// with the name.
///
/// **The ladder is a menu, not a promise about cost.** Attention is quadratic
/// in packed sequence length, so the rungs do not cost what their megapixel
/// numbers suggest — 2.0 MP is 5x the pixels of 0.4 MP and roughly 12x the
/// sampling time. The top two rungs are past `H3RenderPolicy`'s three-hour
/// ceiling at 124 frames and 20 steps and will be refused at render time unless
/// the caller passes `allowSuboptimal`. They are registered rather than
/// omitted so the refusal can carry its measured reason, which is the same
/// argument the 2K entries are kept under.
package enum H3Ladder {
    package static let rungs: [H3LadderRung] = [
        .init(megapixels: 0.2,  width: 608,  height: 352,  landscape: .h3_16x9_0p2mp,    portrait: .h3_9x16_0p2mp),
        .init(megapixels: 0.3,  width: 768,  height: 416,  landscape: .h3_16x9_0p3mp,    portrait: .h3_9x16_0p3mp),
        .init(megapixels: 0.4,  width: 832,  height: 448,  landscape: .h3_16x9_0p4mp,    portrait: .h3_9x16_0p4mp),
        .init(megapixels: 0.5,  width: 992,  height: 544,  landscape: .h3_16x9_0p5mp,    portrait: .h3_9x16_0p5mp),
        .init(megapixels: 0.6,  width: 1024, height: 576,  landscape: .h3_16x9_0p6mp,    portrait: .h3_9x16_0p6mp),
        .init(megapixels: 0.7,  width: 1184, height: 640,  landscape: .h3_16x9_0p7mp,    portrait: .h3_9x16_0p7mp),
        .init(megapixels: 0.8,  width: 1216, height: 704,  landscape: .h3_16x9_0p8mp,    portrait: .h3_9x16_0p8mp),
        .init(megapixels: 0.9,  width: 1344, height: 736,  landscape: .h3_16x9_0p9mp,    portrait: .h3_9x16_0p9mp),
        .init(megapixels: 0.98, width: 1376, height: 768,  landscape: .h3_16x9_0p98mp,   portrait: .h3_9x16_0p98mp),
        .init(megapixels: 1.0,  width: 1408, height: 768,  landscape: .h3_16x9_1p0mp,    portrait: .h3_9x16_1p0mp),
        .init(megapixels: 1.2,  width: 1408, height: 800,  landscape: .h3_16x9_1p2mp,    portrait: .h3_9x16_1p2mp),
        .init(megapixels: 1.5,  width: 1600, height: 928,  landscape: .h3_16x9_1p5mp,    portrait: .h3_9x16_1p5mp),
        .init(megapixels: 1.8,  width: 1792, height: 1024, landscape: .h3_16x9_1p8mp,    portrait: .h3_9x16_1p8mp),
        .init(megapixels: 2.0,  width: 1888, height: 1024, landscape: .h3_16x9_2p0mp,    portrait: .h3_9x16_2p0mp),
    ]

    /// The rung the port has actually verified end to end.
    ///
    /// Pinned against `H3RenderPolicy.verifiedWidth`/`verifiedHeight` by the
    /// tests: if the ladder and the policy ever name different shapes, the menu
    /// no longer contains the one size with a measured tolerance behind it.
    package static var verified: H3LadderRung {
        rungs.first { $0.width == H3RenderPolicy.verifiedWidth
                   && $0.height == H3RenderPolicy.verifiedHeight }!
    }
}

/// What a recipe's resolution actually *is* in this model's pipeline.
///
/// The distinction is not cosmetic and getting it wrong is why `--resolution
/// 2k` used to be the default. 2k is **not a base render size**: it is the
/// output of In-Context Regeneration, a second stage that takes a base render
/// and upscales it. Asking the DiT to sample 2560x1440 directly is not "the
/// high quality setting", it is a shape the model was not trained to sample —
/// and it is 133,200 packed tokens, an estimated 14.6 hours for 20 steps.
///
/// The upscaler is **not released** and not ported (`--in-context-upscale`
/// already refuses as unimplemented), so every 2k recipe is currently a target
/// with no route to it.
package enum H3RecipeTier: String, Sendable, Codable {
    /// A size the DiT samples directly.
    case baseRender
    /// A size only reachable by upscaling a base render. Not sampleable.
    case upscaleTarget
}

package struct H3Recipe: Sendable {
    package let id: H3RecipeID
    package let tier: H3RecipeTier
    package let targetWidth: Int
    package let targetHeight: Int
    package let minDuration: Int
    package let maxDuration: Int
    package let minSteps: Int
    package let maxSteps: Int
    package let requiresAudio: Bool
    
    package init(
        id: H3RecipeID,
        tier: H3RecipeTier,
        targetWidth: Int,
        targetHeight: Int,
        // 4 s snaps to 107 frames, under the model's 124-frame trained floor —
        // the one duration in 4...15 that does. 5 s is the real minimum.
        minDuration: Int = 5,
        maxDuration: Int = 15,
        // 6 steps measured 52 flash events; 20 measured none.
        minSteps: Int = H3RenderPolicy.minSteps,
        maxSteps: Int = 100,
        requiresAudio: Bool = true
    ) {
        self.id = id
        self.tier = tier
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        self.minDuration = minDuration
        self.maxDuration = maxDuration
        self.minSteps = minSteps
        self.maxSteps = maxSteps
        self.requiresAudio = requiresAudio
    }
}

package enum H3RecipeValidationError: Error, CustomStringConvertible {
    case resolutionMismatch(expected: String, got: String, recipeID: H3RecipeID)
    case notABaseRender(recipeID: H3RecipeID, size: String)
    case durationOutOfRange(requested: Int, allowed: ClosedRange<Int>, recipeID: H3RecipeID)
    case stepsOutOfRange(requested: Int, allowed: ClosedRange<Int>, recipeID: H3RecipeID)

    package var description: String {
        switch self {
        case .resolutionMismatch(let expected, let got, let id):
            return "Recipe '\(id.rawValue)' requires resolution \(expected); got \(got)."
        case .notABaseRender(let id, let size):
            return "Recipe '\(id.rawValue)' (\(size)) is an upscale target, not a base "
                 + "render size. 2K is produced by In-Context Regeneration from a base "
                 + "render; the DiT does not sample it directly. That upscaler is not "
                 + "released and not ported — --in-context-upscale refuses as "
                 + "unimplemented — so there is currently no route to this size. "
                 + "Use a 768p recipe."
        case .durationOutOfRange(let requested, let allowed, let id):
            return "Recipe '\(id.rawValue)' requires duration in range \(allowed) seconds; requested \(requested)."
        case .stepsOutOfRange(let requested, let allowed, let id):
            return "Recipe '\(id.rawValue)' requires steps in range \(allowed); requested \(requested)."
        }
    }
}

extension H3Recipe {
    package func validate(
        width: Int,
        height: Int,
        duration: Int,
        steps: Int
    ) throws {
        guard tier == .baseRender else {
            throw H3RecipeValidationError.notABaseRender(
                recipeID: id, size: "\(targetWidth)x\(targetHeight)")
        }
        guard width == targetWidth, height == targetHeight else {
            throw H3RecipeValidationError.resolutionMismatch(
                expected: "\(targetWidth)x\(targetHeight)", got: "\(width)x\(height)", recipeID: id
            )
        }
        guard (minDuration...maxDuration).contains(duration) else {
            throw H3RecipeValidationError.durationOutOfRange(
                requested: duration, allowed: minDuration...maxDuration, recipeID: id
            )
        }
        guard (minSteps...maxSteps).contains(steps) else {
            throw H3RecipeValidationError.stepsOutOfRange(
                requested: steps, allowed: minSteps...maxSteps, recipeID: id
            )
        }
    }
}

package struct H3RecipeRegistry {
    /// 2K entries are kept, not deleted: they are the real output sizes of the
    /// In-Context Regeneration stage and the names callers will ask for. They
    /// are tagged `.upscaleTarget` so asking for one as a base render fails
    /// with an explanation instead of quietly sampling a shape the model was
    /// never trained to sample.
    /// The four aspect ratios the ladder does not cover keep their single 768p
    /// entry. The source table is 16:9 only, and deriving 21:9 or 4:3 rungs
    /// from it would be inventing sizes rather than registering recommended
    /// ones — a menu whose entries have different provenance, with nothing in
    /// the name to say which is which.
    package static let all: [H3RecipeID: H3Recipe] = {
        var out: [H3RecipeID: H3Recipe] = [
            .h3_2k_16_9: H3Recipe(id: .h3_2k_16_9, tier: .upscaleTarget, targetWidth: 2560, targetHeight: 1440),
            .h3_2k_9_16: H3Recipe(id: .h3_2k_9_16, tier: .upscaleTarget, targetWidth: 1440, targetHeight: 2560),
            .h3_2k_21_9: H3Recipe(id: .h3_2k_21_9, tier: .upscaleTarget, targetWidth: 3360, targetHeight: 1440),
            .h3_2k_4_3: H3Recipe(id: .h3_2k_4_3, tier: .upscaleTarget, targetWidth: 1920, targetHeight: 1440),
            .h3_2k_3_4: H3Recipe(id: .h3_2k_3_4, tier: .upscaleTarget, targetWidth: 1440, targetHeight: 1920),
            .h3_2k_1_1: H3Recipe(id: .h3_2k_1_1, tier: .upscaleTarget, targetWidth: 1440, targetHeight: 1440),

            .h3_768p_21_9: H3Recipe(id: .h3_768p_21_9, tier: .baseRender, targetWidth: 1792, targetHeight: 768),
            .h3_768p_4_3: H3Recipe(id: .h3_768p_4_3, tier: .baseRender, targetWidth: 1024, targetHeight: 768),
            .h3_768p_3_4: H3Recipe(id: .h3_768p_3_4, tier: .baseRender, targetWidth: 768, targetHeight: 1024),
            .h3_768p_1_1: H3Recipe(id: .h3_768p_1_1, tier: .baseRender, targetWidth: 768, targetHeight: 768),
        ]
        // Generated from the one table rather than transcribed twice: a
        // portrait entry that disagreed with its landscape rung would be a
        // plausible-looking size with no measurement behind it.
        for rung in H3Ladder.rungs {
            out[rung.landscape] = H3Recipe(id: rung.landscape, tier: .baseRender,
                                           targetWidth: rung.width, targetHeight: rung.height)
            out[rung.portrait] = H3Recipe(id: rung.portrait, tier: .baseRender,
                                          targetWidth: rung.height, targetHeight: rung.width)
        }
        return out
    }()
}
