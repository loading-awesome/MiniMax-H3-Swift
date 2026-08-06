// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// What happened at one sampler step, on one conditioning branch.
///
/// **The unit of evidence for every acceleration claim in this tree.** A render
/// reports one wall-clock total and one skip percentage, and neither can settle
/// an argument: two configurations that finish in the same time can have got
/// there by skipping different steps at different points on the schedule, and a
/// change that helps the quiet middle while wrecking the first two steps looks
/// like a modest win in the total. The per-step record is what distinguishes
/// them.
///
/// Deliberately free of MLX. Producing these numbers needs tensors; carrying,
/// aggregating and serialising them does not, and keeping the type here means
/// the CSV writer and every summary statistic test in microseconds on a machine
/// with no GPU and no checkpoint.
package struct StepTrace: Sendable, Codable, Equatable {

    /// CFG runs two forwards per step against different conditioning. Their
    /// residuals are not comparable and they hold separate caches, so their
    /// traces are separate too — averaging them would blend two different
    /// trajectories and hide the case where only one branch is skipping.
    package enum Branch: String, Sendable, Codable, Equatable {
        case conditional, unconditional
    }

    package let step: Int
    package let branch: Branch
    /// The video-stream sigma this step is integrating at. Recorded because
    /// every proposal to make the cache sigma-aware needs the actual schedule
    /// this checkpoint runs, not the one a paper assumed.
    package let sigma: Double

    /// Relative L1 change in block 0's residual over **every packed row**.
    /// This is what published caches for this model threshold on.
    package let wholeSequenceChange: Double
    /// The same, over the target-video rows alone.
    ///
    /// At 864x480x124 those rows are 95.1% of the sequence, so this tracks
    /// `wholeSequenceChange` closely — which is the point. It is recorded to
    /// *show* that, rather than to let a variable named `video` go on holding
    /// the whole sequence.
    package let videoChange: Double
    /// The same, over the target-audio rows alone — 2.6% of the sequence, and
    /// measured moving 32% more per step than the whole-sequence average.
    package let audioChange: Double

    package let decision: StepCachePolicy.Decision
    package let reason: StepCachePolicy.Reason
    package let consecutiveSkipsBefore: Int

    package init(step: Int, branch: Branch, sigma: Double,
                wholeSequenceChange: Double, videoChange: Double, audioChange: Double,
                decision: StepCachePolicy.Decision, reason: StepCachePolicy.Reason,
                consecutiveSkipsBefore: Int) {
        self.step = step
        self.branch = branch
        self.sigma = sigma
        self.wholeSequenceChange = wholeSequenceChange
        self.videoChange = videoChange
        self.audioChange = audioChange
        self.decision = decision
        self.reason = reason
        self.consecutiveSkipsBefore = consecutiveSkipsBefore
    }
}

extension StepCachePolicy.Decision: Codable {
    // Spelled out rather than derived so the JSON reads as "full"/"reused"
    // instead of the enum's Swift-shaped case names. A receipt is meant to be
    // legible to someone who has never opened this file.
    package init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        switch raw {
        case "full": self = .runFull
        case "reused": self = .reuse
        default:
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "unknown decision \(raw)"))
        }
    }

    package func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(self == .reuse ? "reused" : "full")
    }
}

/// Every step of one render, plus the summary statistics a sweep compares on.
package struct SamplingTrace: Sendable, Codable, Equatable {

    package var steps: [StepTrace]
    /// Wall clock for each sampler step, in seconds, indexed by step. One entry
    /// per step regardless of how many CFG branches ran inside it — the
    /// sampler's clock does not split between branches.
    package var stepSeconds: [Double]

    package init(steps: [StepTrace] = [], stepSeconds: [Double] = []) {
        self.steps = steps
        self.stepSeconds = stepSeconds
    }

    package var stepsSkipped: Int { steps.filter { $0.decision == .reuse }.count }
    package var stepsRun: Int { steps.filter { $0.decision == .runFull }.count }

    /// How many full steps each reason accounts for.
    ///
    /// The number that separates "the threshold is doing the work" from "the
    /// consecutive cap is doing the work". A configuration whose refusals are
    /// mostly `consecutiveCap` has a threshold set past the point where its own
    /// signal means anything, and the cap is the only thing holding the render
    /// together.
    package var reasonCounts: [StepCachePolicy.Reason: Int] {
        steps.reduce(into: [:]) { $0[$1.reason, default: 0] += 1 }
    }

    /// Median of the finite values. Median rather than mean because the first
    /// comparison of every render is infinite by construction and the second is
    /// usually an outlier; a mean over 20 steps is two-thirds warm-up.
    package static func median(_ xs: [Double]) -> Double {
        let f = xs.filter { $0.isFinite }.sorted()
        guard !f.isEmpty else { return .nan }
        return f.count % 2 == 1 ? f[f.count / 2]
                                : (f[f.count / 2 - 1] + f[f.count / 2]) / 2
    }

    /// Seconds per step, median — the figure to compare across configurations.
    ///
    /// **Not the mean, and not the total divided by steps.** Both are dominated
    /// by the first and last steps, which every configuration is required to
    /// run in full, so both understate the difference between two cache
    /// settings by a fixed amount that varies with step count.
    package var medianStepSeconds: Double { Self.median(stepSeconds) }

    /// The one-line form, for a terminal.
    package func summary(threshold: Double, perStreamProbe: Bool) -> String {
        let total = steps.count
        guard total > 0 else { return "step cache: unused" }
        let pct = 100.0 * Double(stepsSkipped) / Double(total)
        let refusals = reasonCounts
            .filter { $0.key != .belowThreshold && $0.value > 0 }
            .sorted { $0.value > $1.value }
            .map { "\($0.key.rawValue) \($0.value)" }
            .joined(separator: ", ")
        return String(format: "step cache [%@]: %d/%d branch-steps reused (%.0f%%), "
                      + "threshold %.3f, median change: whole %.3f video %.3f audio %.3f; "
                      + "full because — %@",
                      (perStreamProbe ? "per-stream" : "whole-sequence") as NSString,
                      stepsSkipped, total, pct, threshold,
                      Self.median(steps.map(\.wholeSequenceChange)),
                      Self.median(steps.map(\.videoChange)),
                      Self.median(steps.map(\.audioChange)),
                      (refusals.isEmpty ? "nothing" : refusals) as NSString)
    }

    /// One row per branch-step, for loading into anything that reads CSV.
    ///
    /// Non-finite values are written as empty fields rather than `inf` or `nan`,
    /// which most readers silently coerce to 0 — a first-step infinity landing
    /// in a spreadsheet as a zero would read as the quietest step of the render.
    package var csv: String {
        var out = "step,branch,sigma,whole_sequence_change,video_change,audio_change,"
                + "decision,reason,consecutive_skips_before,step_seconds\n"
        func num(_ x: Double) -> String { x.isFinite ? String(format: "%.6f", x) : "" }
        for t in steps {
            let secs = t.step < stepSeconds.count ? num(stepSeconds[t.step]) : ""
            out += "\(t.step),\(t.branch.rawValue),\(num(t.sigma)),"
                 + "\(num(t.wholeSequenceChange)),\(num(t.videoChange)),\(num(t.audioChange)),"
                 + "\(t.decision == .reuse ? "reused" : "full"),\(t.reason.rawValue),"
                 + "\(t.consecutiveSkipsBefore),\(secs)\n"
        }
        return out
    }
}
