// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import H3Foundation

/// The recipe menu, priced.
///
/// **A list of names is not a menu.** The registry holds 38 entries and three
/// of the things a caller most needs to know about one — what it costs, whether
/// it can be rendered at all, and which single entry has a measured tolerance
/// behind it — are nowhere in its name. Two of the 38 are refused at render
/// time by `H3RenderPolicy` and six have no route at all until the upscaler
/// lands. Discovering that by starting a render is the failure this exists to
/// prevent.
///
/// The estimate is `H3RenderPolicy.estimatedSeconds` — the same arithmetic the
/// run banner and the refusal quote, so a size cannot be advertised at one
/// price here and refused at another there.
package struct H3RecipeCatalogue: Sendable {

    package enum Status: Sendable {
        /// Verified end to end: the parity-gated shape, and the one proven to
        /// render an intelligible talking head.
        case verified
        /// A size the DiT samples directly, inside the render-time ceiling.
        case baseRender
        /// Sampleable, but past `maxEstimatedSeconds` at this duration and step
        /// count. Refused without `--allow-suboptimal`.
        case overCeiling
        /// Not a base render size. Needs In-Context Regeneration, which is
        /// neither released nor ported, so there is currently no route to it.
        case upscaleTarget

        package var label: String {
            switch self {
            case .verified: return "verified"
            case .baseRender: return "ok"
            case .overCeiling: return "over ceiling"
            case .upscaleTarget: return "no route"
            }
        }
    }

    package struct Row: Sendable {
        package let id: H3RecipeID
        package let width: Int
        package let height: Int
        package let tokens: Int
        package let estimateSeconds: Double
        package let status: Status

        package var aspect: Double { Double(width) / Double(height) }
    }

    package let rows: [Row]
    package let frameCount: Int
    package let steps: Int

    /// Price every registered recipe at one duration and step count.
    ///
    /// Both default to the verified configuration, because that is the number
    /// every other figure in this project is quoted relative to.
    package init(seconds: Int = 5, steps: Int = H3RenderPolicy.verifiedSteps) {
        let frames = LatentGeometry.alignFrameCount(seconds * H3Video.fps)
        self.frameCount = frames
        self.steps = steps

        func row(_ id: H3RecipeID, _ recipe: H3Recipe) -> Row {
            let geometry = LatentGeometry(width: recipe.targetWidth,
                                          height: recipe.targetHeight,
                                          length: seconds * H3Video.fps)
            // Video plus audio, matching what `render` puts in its banner and
            // feeds to the policy. A listing that counted tokens differently
            // would quote a price the render then disagrees with.
            let tokens = geometry.videoTokens + geometry.audioTokens
            let estimate = H3RenderPolicy.estimatedSeconds(tokens: tokens, steps: steps)

            // Order matters. Every 2K size is also past the ceiling, and
            // reporting it as "too slow" would be true and misleading — it
            // implies fewer steps or a faster machine would get you there,
            // when nothing about the size is reachable at any speed.
            let status: Status
            if recipe.tier == .upscaleTarget {
                status = .upscaleTarget
            } else if estimate > H3RenderPolicy.maxEstimatedSeconds {
                status = .overCeiling
            } else if recipe.targetWidth == H3RenderPolicy.verifiedWidth,
                      recipe.targetHeight == H3RenderPolicy.verifiedHeight {
                status = .verified
            } else {
                status = .baseRender
            }
            return Row(id: id, width: recipe.targetWidth, height: recipe.targetHeight,
                       tokens: tokens, estimateSeconds: estimate, status: status)
        }

        // Ladder order is the table's own order — ascending megapixels — and
        // not sorted by anything computed. The whole point of the ladder is
        // that it is a sequence somebody chose; re-sorting it here would make
        // the printed menu disagree with the source it came from.
        var out: [Row] = []
        for rung in H3Ladder.rungs {
            if let r = H3RecipeRegistry.all[rung.landscape] { out.append(row(rung.landscape, r)) }
        }
        for rung in H3Ladder.rungs {
            if let r = H3RecipeRegistry.all[rung.portrait] { out.append(row(rung.portrait, r)) }
        }
        let laddered = Set(H3Ladder.rungs.flatMap { [$0.landscape, $0.portrait] })
        let rest = H3RecipeRegistry.all
            .filter { !laddered.contains($0.key) }
            .sorted { a, b in
                let (pa, pb) = (a.value.targetWidth * a.value.targetHeight,
                                b.value.targetWidth * b.value.targetHeight)
                // Base renders before upscale targets, then by size: the things
                // you can actually run come first.
                if (a.value.tier == .upscaleTarget) != (b.value.tier == .upscaleTarget) {
                    return b.value.tier == .upscaleTarget
                }
                return pa == pb ? a.key.rawValue < b.key.rawValue : pa < pb
            }
        out += rest.map { row($0.key, $0.value) }
        self.rows = out
    }

    /// Rows in one aspect family, for a filtered listing.
    package func rows(matching filter: String) -> [Row] {
        rows.filter { $0.id.rawValue.contains(filter) }
    }

    // MARK: rendering

    static func duration(_ seconds: Double) -> String {
        seconds < 90 * 60
            ? String(format: "%.0f min", seconds / 60)
            : String(format: "%.1f h", seconds / 3600)
    }

    static func thousands(_ n: Int) -> String {
        let s = String(n)
        guard s.count > 3 else { return s }
        return String(s.dropLast(3)) + "," + String(s.suffix(3))
    }

    // Padded by hand rather than with `%-16@`: Foundation's `String(format:)`
    // silently ignores the field width on `%@`, so every column ran together
    // and the table was unreadable while each individual value was correct.
    static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }

    static func padLeft(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
    }

    package var report: String {
        report(rows)
    }

    package func report(_ rows: [Row]) -> String {
        var out = ""
        let seconds = Double(frameCount) / Double(H3Video.fps)
        out += String(format: "%d frames (%.1f s) at %d steps, %.2f TFLOP/s measured.\n",
                      frameCount, seconds, steps, H3RenderPolicy.measuredTFLOPS)
        // Computed, not asserted. The pixel ratio is fixed but the time ratio
        // moves with duration and step count — at 10 s the same two rungs are
        // 14x apart, not 12x — and a hardcoded sentence would have been a
        // stale number sitting directly above the table that contradicts it.
        // The largest *sampleable* size, not the largest registered one: the
        // 2K entries are bigger and unreachable, and illustrating what cost
        // does with a size nobody can render teaches the wrong lesson.
        if let base = self.rows.first(where: { $0.status == .verified }),
           let top = self.rows
               .filter({ $0.status != .upscaleTarget })
               .max(by: { $0.width * $0.height < $1.width * $1.height }) {
            out += String(format:
                "Attention is quadratic in sequence length, so cost climbs faster than "
                + "pixels do:\n%@ is %.1fx the pixels of %dx%d and %.0fx its sampling "
                + "time.\n\n",
                "\(top.width)x\(top.height)" as NSString,
                Double(top.width * top.height) / Double(base.width * base.height),
                base.width, base.height,
                top.estimateSeconds / base.estimateSeconds)
        }

        out += "  " + Self.pad("recipe", 16) + Self.pad("size", 11)
             + Self.padLeft("aspect", 6) + Self.padLeft("tokens", 9)
             + Self.padLeft("sampling", 10) + Self.padLeft("vs 864", 8)
             + "   status\n"

        // The baseline comes from the whole catalogue, not from the subset
        // being printed: `--aspect 2k` filters out the verified row, and a
        // relative column that emptied itself the moment you narrowed the
        // listing would drop the comparison exactly where it is most wanted.
        let verified = self.rows.first { $0.status == .verified }?.estimateSeconds
        var previousGroup: String?
        for r in rows {
            // A blank line between families, so 28 ladder rows do not read as
            // one undifferentiated wall.
            let group = r.id.rawValue.split(separator: "_").dropFirst().first.map(String.init) ?? ""
            if let p = previousGroup, p != group { out += "\n" }
            previousGroup = group

            let relative = verified.map { String(format: "%.2fx", r.estimateSeconds / $0) } ?? "-"
            out += "  " + Self.pad(r.id.rawValue, 16)
                 + Self.pad("\(r.width)x\(r.height)", 11)
                 + Self.padLeft(String(format: "%.3f", r.aspect), 6)
                 + Self.padLeft(Self.thousands(r.tokens), 9)
                 + Self.padLeft(Self.duration(r.estimateSeconds), 10)
                 + Self.padLeft(relative, 8)
                 + "   " + r.status.label + "\n"
        }

        out += "\n"
        if rows.contains(where: { $0.status == .verified }) {
            out += "verified      864x480 is the parity-gated shape and the only one with a "
                 + "measured tolerance\n              behind it. Every speed figure in this "
                 + "project is quoted against it.\n"
        }
        if rows.contains(where: { $0.status == .overCeiling }) {
            out += String(format:
                "over ceiling  past the %.0f-hour refusal at this duration and step count. "
                + "Pass\n              --allow-suboptimal to run one anyway, or ask for "
                + "fewer steps.\n",
                H3RenderPolicy.maxEstimatedSeconds / 3600)
        }
        if rows.contains(where: { $0.status == .upscaleTarget }) {
            out += "no route      2K is the output of In-Context Regeneration, which is "
                 + "unreleased and\n              unported. The DiT does not sample these "
                 + "sizes directly.\n"
        }
        return out
    }
}
