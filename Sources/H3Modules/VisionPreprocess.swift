// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Foundation

/// Image -> `flatten_patches`, the vision tower's actual input.
///
/// `process_qwen2vl_images` in the reference. Three things it does that are
/// easy to get wrong, in the order they bite:
///
///  1. **Resize to a multiple of `patch * merge` = 32**, by rounding the
///     original dimensions rather than by cropping.
///  2. **Normalise with mean/std 0.5** — Qwen3-VL maps to [-1, 1]. Qwen2.5-VL
///     used CLIP statistics, and the two look similar enough to swap by
///     accident and produce plausible garbage.
///  3. **Duplicate the frame across the temporal patch**, then apply a 9-way
///     permutation that interleaves 2x2 merge blocks. That permutation is the
///     reason a token's neighbours in the sequence are its neighbours in the
///     block, which is what the merger's reshape assumes.
package enum VisionPreprocess {
    package static let mean: Float = 0.5
    package static let std: Float = 0.5

    /// Target patch grid for an image, given the reference's rounding rules.
    ///
    /// The min/max pixel clamps exist in the reference and are reproduced here,
    /// but note the H3 path never approaches either: 3136 px is a 56x56 image
    /// and 12.8 Mpx is larger than anything a prompt carries.
    package static func grid(width: Int, height: Int,
                            config: VisionTowerConfig = VisionTowerConfig(),
                            minPixels: Int = 3136, maxPixels: Int = 12_845_056)
        -> (width: Int, height: Int, grid: VisionGrid) {
        let factor = config.patchSize * config.spatialMergeSize
        var hBar = Int((Double(height) / Double(factor)).rounded()) * factor
        var wBar = Int((Double(width) / Double(factor)).rounded()) * factor

        if hBar * wBar > maxPixels {
            let beta = (Double(height) * Double(width) / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            wBar = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (Double(height) * Double(width))).squareRoot()
            hBar = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            wBar = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (wBar, hBar, VisionGrid(h: hBar / config.patchSize, w: wBar / config.patchSize))
    }

    /// Already-resized image `[1, H, W, 3]` in [0, 1] -> `[tokens, 1536]`.
    ///
    /// Resizing is the caller's job because it belongs to whatever decoded the
    /// file; this is the part that has to match the reference bit for bit.
    package static func patches(image: MLXArray, grid: VisionGrid,
                               config: VisionTowerConfig = VisionTowerConfig()) throws -> MLXArray {
        let p = config.patchSize
        let tp = config.temporalPatchSize
        guard image.dim(1) == grid.h * p && image.dim(2) == grid.w * p else {
            throw H3Error.mediaOffCanvas(
                path: "presented image", size: "\(image.dim(2))x\(image.dim(1))",
                remedy: "this grid wants \(grid.w * p)x\(grid.h * p). Resizing belongs to "
                      + "whatever decoded the file; this step has to match the reference bit "
                      + "for bit and so will not resize for you.")
        }

        let norm = (image.asType(.float32).transposed(0, 3, 1, 2) - mean) / std  // [1,3,H,W]
        // The single frame is repeated across the temporal patch: the tower
        // always consumes 2 frames, and a still image is both of them.
        let rep = tiled(norm, repetitions: [tp, 1, 1, 1])                        // [2,3,H,W]
        return pack(rep, grid: grid, config: config)
    }

    /// `process_video_block` — a **frame pair** `[2, H, W, 3]` in [0, 1].
    ///
    /// Identical to ``patches(image:grid:config:)`` except that the temporal
    /// patch is filled by two distinct frames instead of one repeated one. That
    /// is the whole difference, and it is why a reference video is presented to
    /// Qwen at 2 fps in pairs rather than frame by frame: each pair is one
    /// vision block covering half a second.
    ///
    /// `grid.t` stays 1 — the pair fills the temporal patch, it does not add a
    /// temporal grid step.
    package static func patches(framePair: MLXArray, grid: VisionGrid,
                               config: VisionTowerConfig = VisionTowerConfig()) throws -> MLXArray {
        let p = config.patchSize, tp = config.temporalPatchSize
        guard framePair.dim(0) == tp else {
            throw H3Error.invalidRequest(
                rule: "wrong frame count for a video block",
                detail: "a video block wants \(tp) frames, got \(framePair.dim(0))",
                remedy: "present reference video as frame pairs at 2 fps; a pair fills the "
                      + "temporal patch and does not add a temporal grid step.")
        }
        guard framePair.dim(1) == grid.h * p && framePair.dim(2) == grid.w * p else {
            throw H3Error.mediaOffCanvas(
                path: "presented frame pair", size: "\(framePair.dim(2))x\(framePair.dim(1))",
                remedy: "this grid wants \(grid.w * p)x\(grid.h * p).")
        }
        let norm = (framePair.asType(.float32).transposed(0, 3, 1, 2) - mean) / std  // [2,3,H,W]
        return pack(norm, grid: grid, config: config)
    }

    /// The 9-way permutation, shared by the still and the frame-pair paths.
    static func pack(_ norm: MLXArray, grid: VisionGrid,
                     config: VisionTowerConfig) -> MLXArray {
        let p = config.patchSize, m = config.spatialMergeSize
        let tp = config.temporalPatchSize, ch = config.inChannels
        return norm.reshaped([grid.t, tp, ch,
                              grid.h / m, m, p,
                              grid.w / m, m, p])
                   .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
                   .reshaped([grid.tokens, ch * tp * p * p])
    }
}
