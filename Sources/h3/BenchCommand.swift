// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import ArgumentParser
import H3Foundation

/// Reads the benchmark records renders leave behind and prints the table.
///
/// Argument plumbing only — every rule about what may be compared with what
/// lives in `BenchmarkComparison` and `BenchmarkRecord`, where it can be tested
/// without spawning a process.
struct BenchCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Compare recorded renders.",
        discussion: """
            Every render writes `<output>.h3-bench.json` beside its mp4, plus a
            `<output>.h3-steps.csv` with one row per sampler step. This reads a
            directory of them and prints what changed.

              h3 bench --dir ~/renders

            Arms are named from what was on — `dense-sdpa`, `cached-0.100-perstream-sdpa`
            — or from `H3_BENCH_ARM` when a sweep sets it. The control defaults
            to the first arm with the cache off, which is the only baseline that
            is not itself an approximation.

            Runs made with different prompts, seeds, dimensions, step counts,
            checkpoints, machines or MLX revisions are **not** compared. They are
            listed with the reason instead, because a comparison tool that hides
            what it dropped produces a table that looks complete.
            """
    )

    @Option(help: "directory of renders to read, searched recursively")
    var dir: String

    @Option(help: "arm to treat as the control; defaults to the first with the cache off")
    var control: String?

    @Flag(help: "print the per-step rows as CSV instead of the table")
    var csv = false

    func run() throws {
        let records = try BenchmarkComparison.load(
            directory: URL(fileURLWithPath: dir, isDirectory: true))
        guard !records.isEmpty else {
            print("no .h3-bench.json files under \(dir)")
            return
        }
        if csv {
            print("arm,step,branch,sigma,whole_sequence_change,video_change,audio_change,"
                  + "decision,reason,consecutive_skips_before,step_seconds")
            for r in records {
                for line in r.trace.csv.split(separator: "\n").dropFirst() {
                    print("\(r.arm),\(line)")
                }
            }
            return
        }
        print("")
        print(BenchmarkComparison(records: records, control: control).report)
    }
}
