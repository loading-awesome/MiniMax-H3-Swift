// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// A set of recorded runs, arranged into the table a decision gets made from.
///
/// **The refusals are the feature.** A comparison tool that quietly drops the
/// arms it cannot line up produces a clean table with a hole in it, and a hole
/// in a table is invisible — the reader sees five rows and concludes five arms
/// were tested. Every arm that cannot be compared appears here anyway, with the
/// reason it could not, so that a missing number is as loud as a bad one.
package struct BenchmarkComparison {

    package struct Row {
        package let arm: String
        package let runs: Int
        /// Median across repeats of each run's own median step time.
        package let medianStepSeconds: Double
        /// Spread across repeats, as a fraction of the median.
        ///
        /// **The number that says whether any of this means anything.** A 5%
        /// improvement measured against a control whose own repeats vary by 8%
        /// is not an improvement; it is one draw from a distribution. Reported
        /// beside every speed-up so the two are read together.
        package let repeatSpread: Double
        package let stepsSkippedFraction: Double
        package let peakGB: Double
        /// Nil when this arm could not be compared to the control.
        package let speedup: Double?
        package let refusals: [String]
        /// Configuration keys that differ from the control's and are not part
        /// of the named profile — environment overrides, resolved defaults.
        ///
        /// **Not a refusal.** Configuration is what an experiment varies, so
        /// differing here must not block the comparison. It must be *stated*:
        /// an arm that differs from the control in a knob nobody mentioned is
        /// how a result gets attributed to the wrong cause.
        package let configurationDrift: [String]
    }

    package let controlArm: String
    package let rows: [Row]
    /// Arms present in the input that share no comparable control.
    package let unusable: [String]

    /// - Parameter control: the arm to divide by. When nil, the first arm whose
    ///   cache is off is used — the faithful dense run, which is the only
    ///   baseline that is not itself an approximation.
    package init(records: [BenchmarkRecord], control: String? = nil) {
        var byArm: [String: [BenchmarkRecord]] = [:]
        for r in records { byArm[r.arm, default: []].append(r) }

        let controlName = control
            ?? records.first(where: { $0.configuration.cacheThreshold == 0 })?.arm
            ?? records.first?.arm ?? ""
        controlArm = controlName
        let controlRuns = byArm[controlName] ?? []

        var rows: [Row] = []
        var unusable: [String] = []
        for (arm, runs) in byArm.sorted(by: { $0.key < $1.key }) {
            let medians = runs.map(\.trace.medianStepSeconds).filter(\.isFinite)
            let median = SamplingTrace.median(medians)
            // Full range rather than a standard deviation: with two or three
            // repeats a standard deviation is mostly a statement about the
            // sample size, while the spread between best and worst is exactly
            // the quantity a reader needs to compare a claimed gain against.
            let spread = (medians.count > 1 && median > 0)
                ? ((medians.max() ?? 0) - (medians.min() ?? 0)) / median : 0

            let allSteps = runs.flatMap(\.trace.steps)
            let skipped = allSteps.isEmpty ? 0
                : Double(allSteps.filter { $0.decision == .reuse }.count) / Double(allSteps.count)
            let peak = Double(runs.map(\.memory.mlxPeakBytes).max() ?? 0) / 1e9

            var speedup: Double?
            var refusals: [String] = []
            var drift: [String] = []
            if let base = controlRuns.first, let mine = runs.first, arm != controlName {
                let a = mine.configuration.overrides, b = base.configuration.overrides
                for key in Set(a.keys).union(b.keys).sorted() where a[key] != b[key] {
                    drift.append("\(key): \(a[key] ?? "unset") vs control's \(b[key] ?? "unset")")
                }
            }
            if arm == controlName {
                speedup = 1.0
            } else if let base = controlRuns.first, let mine = runs.first {
                let problems = mine.incomparabilities(with: base)
                if problems.isEmpty {
                    let controlMedian = SamplingTrace.median(
                        controlRuns.map(\.trace.medianStepSeconds))
                    if median > 0, controlMedian.isFinite { speedup = controlMedian / median }
                } else {
                    refusals = problems
                    unusable.append(arm)
                }
            } else {
                refusals = ["no control named \(controlName) in this set"]
                unusable.append(arm)
            }

            rows.append(Row(arm: arm, runs: runs.count, medianStepSeconds: median,
                            repeatSpread: spread, stepsSkippedFraction: skipped,
                            peakGB: peak, speedup: speedup, refusals: refusals,
                            configurationDrift: drift))
        }
        self.rows = rows
        self.unusable = unusable
    }

    /// The table, for a terminal.
    package var report: String {
        var out = "control: \(controlArm)\n\n"
        out += String(format: "  %-34@ %5@ %10@ %8@ %8@ %8@ %8@\n",
                      "arm" as NSString, "runs" as NSString, "s/step" as NSString,
                      "spread" as NSString, "speedup" as NSString, "reused" as NSString,
                      "peak GB" as NSString)
        for r in rows {
            let speed = r.speedup.map { String(format: "%.2fx", $0) } ?? "—"
            out += String(format: "  %-34@ %5d %10.2f %7.1f%% %8@ %7.0f%% %8.1f\n",
                          r.arm as NSString, r.runs, r.medianStepSeconds,
                          r.repeatSpread * 100, speed as NSString,
                          r.stepsSkippedFraction * 100, r.peakGB)
        }
        // Every gain is printed next to the noise floor it has to clear, rather
        // than leaving the reader to find the spread column and do it.
        if let control = rows.first(where: { $0.arm == controlArm }), control.repeatSpread > 0 {
            out += String(format: "\n  the control's own repeats vary by %.1f%%; "
                          + "a gain smaller than that is not a gain\n",
                          control.repeatSpread * 100)
        } else if (rows.first { $0.arm == controlArm })?.runs ?? 0 < 2 {
            out += "\n  the control ran once, so there is no variance estimate — "
                 + "every speed-up below is unqualified\n"
        }
        for r in rows where !r.refusals.isEmpty {
            out += "\n  \(r.arm) could not be compared:\n"
            for reason in r.refusals { out += "    - \(reason)\n" }
        }
        // Printed for every arm that has it, including ones whose speed-up was
        // computed. A number next to an unmentioned configuration difference is
        // a result attributed to the wrong cause.
        for r in rows where !r.configurationDrift.isEmpty {
            out += "\n  \(r.arm) also differs from the control in:\n"
            for d in r.configurationDrift { out += "    - \(d)\n" }
        }
        return out
    }

    /// Loads every `*.h3-bench.json` under a directory, recursively.
    package static func load(directory: URL) throws -> [BenchmarkRecord] {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)
        var out: [BenchmarkRecord] = []
        while let url = enumerator?.nextObject() as? URL {
            guard url.lastPathComponent.hasSuffix("h3-bench.json") else { continue }
            out.append(try decoder.decode(BenchmarkRecord.self, from: Data(contentsOf: url)))
        }
        return out.sorted { $0.recordedAt < $1.recordedAt }
    }
}
