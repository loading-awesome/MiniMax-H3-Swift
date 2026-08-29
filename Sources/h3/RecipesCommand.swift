// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import ArgumentParser
import H3Recipes

/// Prints the recipe menu with a status column.
///
/// Argument plumbing only — the pricing and the status rules live in
/// `H3RecipeCatalogue`, where they are tested without spawning a process and
/// where they share `H3RenderPolicy`'s arithmetic with the run banner and the
/// refusal. A listing that priced sizes independently would eventually
/// advertise one number and refuse at another.
struct RecipesCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "recipes",
        abstract: "List the output sizes, with cost and status.",
        discussion: """
            Every registered size, priced by the same arithmetic `h3 render`
            quotes in its banner and its refusals.

              h3 recipes
              h3 recipes --seconds 10
              h3 recipes --aspect 9x16

            The 16:9 sizes and their portrait transposes are a megapixel ladder
            snapped to the 32-pixel grid the VAE and patchify require — which is
            why none of them is exactly 16:9 — and then chosen, within that
            grid, for the video decoder's tile layout. The decoder cuts fixed
            256-pixel tiles, so a size just past a tile boundary pays for a
            whole row it barely uses: 0.4 MP is 832x448 and decodes in 8 tiles
            where the 864x480 it replaced needed 15.

            Cost is not linear in pixels. Attention is quadratic in packed
            sequence length, so the 2.0 MP rung is 5x the pixels of the 0.4 MP
            rung and about 12x its sampling time. Two rungs sit past the
            three-hour refusal; they are listed rather than hidden so the cost
            is readable before a render rather than after one.

            The `reference` marker is not `verified`. It names the rung the
            others are priced against and where this port's measurements were
            taken. 864x480 held the verification and is no longer a rung; it is
            still reachable with --width 864 --height 480.
            """
    )

    @Option(help: "duration to price the menu at; snaps up onto the 17k+5 lattice")
    var seconds: Int = 5

    @Option(help: "step count to price the menu at")
    var steps: Int = 20

    @Option(help: "show only ids containing this, e.g. 16x9, 9x16, 2k")
    var aspect: String?

    func run() throws {
        let catalogue = H3RecipeCatalogue(seconds: seconds, steps: steps)
        let rows = aspect.map { catalogue.rows(matching: $0) } ?? catalogue.rows
        guard !rows.isEmpty else {
            print("no recipe id contains '\(aspect ?? "")'. Try 16x9, 9x16, 768p or 2k.")
            return
        }
        print("")
        print(catalogue.report(rows))
    }
}
