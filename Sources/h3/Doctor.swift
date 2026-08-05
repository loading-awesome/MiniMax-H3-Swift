import Foundation
import ArgumentParser
import MiniMaxH3

/// Everything the library decided at startup, and why.
///
/// This exists because the alternative is a support conversation. "It picked
/// the wrong checkpoint", "it says it will not fit", "it is using the slow
/// attention" are all questions with one answer each, and printing all of them
/// at once costs milliseconds — every checkpoint is identified from its header,
/// which is a few hundred kilobytes regardless of a 66 GB body.
///
/// It reports rather than throws. A doctor that stops at the first missing file
/// makes you run it four times to find four problems.
struct Doctor: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "doctor",
        abstract: "Show the detected machine, the resolved checkpoints, and what will fit."
    )

    @Option(help: "configuration file (default: ~/.config/minimax-h3/config.json)")
    var config: String?

    @Option(help: "packed sequence length to plan for; defaults to the verified shape")
    var tokens: Int = 15_750

    func run() throws {
        let machine = Machine.detect()
        let available = Machine.availableBytes()

        print("machine")
        print("  \(machine.summary)")
        print(String(format: "  %.0f GB available right now (free + inactive + speculative)",
                     Double(available) / 1e9))
        if machine.isPortable {
            print("  NOTE: portable chassis. Throughput here was calibrated on a desktop under")
            print("        sustained load; a twenty-minute render will throttle and finish late.")
        }

        // ---- Metal kernels
        //
        // First, because it is the first thing that fails and the least
        // self-explanatory when it does: without them every checkpoint below
        // resolves perfectly and the render dies on its opening GPU op.
        var problems = 0
        print("\nMetal kernels")
        if let lib = MetalLibrary.locate() {
            print("  \(lib.path)")
            if MetalLibrary.locatedOnlyViaWorkingDirectory() {
                problems += 1
                print("  WARNING: found only relative to the current directory, so this binary")
                print("           works from here and nowhere else. Run Scripts/bootstrap-metal.sh")
                print("           to place mlx.metallib beside the executable instead.")
            }
        } else {
            problems += 1
            print("  MISSING — the first GPU operation will fail. Looked in:")
            for p in MetalLibrary.searchPaths { print("    \(p.path)") }
            print("  remedy: run Scripts/bootstrap-metal.sh from the package root.")
        }

        // ---- configuration
        let (cfg, url) = try H3Configuration.load(from: config.map(URL.init(fileURLWithPath:)))
        print("\nconfiguration")
        print("  \(url?.path ?? "none found — using built-in defaults, which point at nothing")")
        if url == nil {
            print("  run `h3 config init` to write a template to \(H3Configuration.defaultURL.path)")
        }
        print("  approximate weights  \(cfg.policy.allowApproximateWeights ? "permitted" : "refused")")
        print("  attention backend    \(cfg.attention.backend)")

        // ---- checkpoints
        let catalog = Catalog(config: cfg)
        let rows = catalog.inventory()
        print("\ncheckpoints (\(rows.count) configured)")
        if rows.isEmpty {
            print("  none configured")
        }
        for (role, path, result) in rows {
            switch result {
            case .success(let id):
                print("  \(role.padding(toLength: 24, withPad: " ", startingAt: 0))\(id.summary)")
                // Only a DiT with no identifiable vendor is a problem; the
                // fused-attention layout is a DiT property and nothing else's.
                if id.kind == .dit && id.vendor == nil { problems += 1 }
            case .failure(let e):
                problems += 1
                print("  \(role.padding(toLength: 24, withPad: " ", startingAt: 0))MISSING  \(path)")
                _ = e
            }
        }

        // ---- what fits
        print("\nmemory plan at \(tokens) packed tokens")
        var anyFits = false
        for precision in MemoryPlan.Precision.allCases {
            let plan = MemoryPlan.plan(precision: precision, packedTokens: tokens,
                                       availableBytes: available,
                                       marginFraction: cfg.policy.memoryMarginFraction)
            let gated = precision.isApproximate && !cfg.policy.allowApproximateWeights
            let verdict: String
            if gated {
                verdict = "refused (approximate weights)"
            } else if plan.fits {
                verdict = String(format: "FITS, %+.0f GB headroom", Double(plan.headroomBytes) / 1e9)
            } else {
                verdict = String(format: "does not fit, short by %.0f GB",
                                 -Double(plan.headroomBytes) / 1e9)
            }
            if plan.fits && !gated { anyFits = true }
            // Say it on the row, not in a footnote. An int8 line showing the
            // same peak as bf16 reads as a bug unless the reason is right
            // there: the weights are dequantised at load, so the file is
            // smaller and the footprint is not.
            let caveat = (!precision.isResidentQuantised
                          && (precision == .int8 || precision == .prunedInt8))
                ? "   [dequantised at load — saves disk, not memory]" : ""
            // `%-12@` does not pad on Darwin's Foundation — width flags are
            // ignored for `%@` — so the columns have to be padded in Swift.
            let name = precision.rawValue.padding(toLength: 12, withPad: " ", startingAt: 0)
            print(String(format: "  %@ peak %6.1f GB   %@%@", name, plan.peakGB, verdict, caveat))
        }

        if let best = MemoryPlan.best(packedTokens: tokens, availableBytes: available,
                                      allowApproximate: cfg.policy.allowApproximateWeights) {
            print("\nselected: \(best.precision.rawValue)")
            print(best.explanation)
        } else if !anyFits {
            print("\nNo configuration fits this machine at \(tokens) packed tokens.")
            print("Activations scale with sequence length and attention is quadratic in it, so a")
            print("smaller checkpoint alone will not help — reduce resolution or duration too.")
        }

        print("")
        print(problems == 0
              ? "no problems found"
              : "\(problems) problem(s) above")
    }
}
