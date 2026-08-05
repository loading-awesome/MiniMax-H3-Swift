import Foundation
import H3Foundation

public enum H3RecipeID: String, Sendable, Codable, CaseIterable {
    case h3_2k_16_9
    case h3_2k_9_16
    case h3_2k_21_9
    case h3_2k_4_3
    case h3_2k_3_4
    case h3_2k_1_1
    
    case h3_768p_16_9
    case h3_768p_9_16
    case h3_768p_21_9
    case h3_768p_4_3
    case h3_768p_3_4
    case h3_768p_1_1
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
    package static let all: [H3RecipeID: H3Recipe] = [
        .h3_2k_16_9: H3Recipe(id: .h3_2k_16_9, tier: .upscaleTarget, targetWidth: 2560, targetHeight: 1440),
        .h3_2k_9_16: H3Recipe(id: .h3_2k_9_16, tier: .upscaleTarget, targetWidth: 1440, targetHeight: 2560),
        .h3_2k_21_9: H3Recipe(id: .h3_2k_21_9, tier: .upscaleTarget, targetWidth: 3360, targetHeight: 1440),
        .h3_2k_4_3: H3Recipe(id: .h3_2k_4_3, tier: .upscaleTarget, targetWidth: 1920, targetHeight: 1440),
        .h3_2k_3_4: H3Recipe(id: .h3_2k_3_4, tier: .upscaleTarget, targetWidth: 1440, targetHeight: 1920),
        .h3_2k_1_1: H3Recipe(id: .h3_2k_1_1, tier: .upscaleTarget, targetWidth: 1440, targetHeight: 1440),
        
        .h3_768p_16_9: H3Recipe(id: .h3_768p_16_9, tier: .baseRender, targetWidth: 1344, targetHeight: 768),
        .h3_768p_9_16: H3Recipe(id: .h3_768p_9_16, tier: .baseRender, targetWidth: 768, targetHeight: 1344),
        .h3_768p_21_9: H3Recipe(id: .h3_768p_21_9, tier: .baseRender, targetWidth: 1792, targetHeight: 768),
        .h3_768p_4_3: H3Recipe(id: .h3_768p_4_3, tier: .baseRender, targetWidth: 1024, targetHeight: 768),
        .h3_768p_3_4: H3Recipe(id: .h3_768p_3_4, tier: .baseRender, targetWidth: 768, targetHeight: 1024),
        .h3_768p_1_1: H3Recipe(id: .h3_768p_1_1, tier: .baseRender, targetWidth: 768, targetHeight: 768),
    ]
}
