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
    /// The headline constraint. Convenient for a one-line summary and not
    /// enough to tune on — see `constraints`.
    package let reason: StepCachePolicy.Reason
    /// Every constraint that independently forced a full step.
    ///
    /// Recorded because the headline hides the interesting case: at step 15 of
    /// the measured control both the cap and the audio probe object, the cap is
    /// reported, and relaxing the cap is exactly the change that makes the
    /// audio objection load-bearing. A sweep reading only the headline would
    /// remove the audio probe just before it starts mattering.
    package let constraints: [StepCachePolicy.Reason]
    /// What a probe that does not vote would have said. Currently video only.
    package let advisory: [StepCachePolicy.Reason]
    package let consecutiveSkipsBefore: Int

    package init(step: Int, branch: Branch, sigma: Double,
                wholeSequenceChange: Double, videoChange: Double, audioChange: Double,
                decision: StepCachePolicy.Decision, reason: StepCachePolicy.Reason,
                constraints: [StepCachePolicy.Reason] = [],
                advisory: [StepCachePolicy.Reason] = [],
                consecutiveSkipsBefore: Int) {
        self.step = step
        self.branch = branch
        self.sigma = sigma
        self.wholeSequenceChange = wholeSequenceChange
        self.videoChange = videoChange
        self.audioChange = audioChange
        self.decision = decision
        self.reason = reason
        self.constraints = constraints
        self.advisory = advisory
        self.consecutiveSkipsBefore = consecutiveSkipsBefore
    }

    // MARK: - Codable

    /// Non-finite deltas encode as JSON `null`, and decode back to infinity.
    ///
    /// **Not a style choice — the synthesised conformance could not encode this
    /// type at all.** `JSONEncoder` throws `invalidValue` on any non-finite
    /// Double, and step 0 of every render has no predecessor and therefore an
    /// infinite change by construction. So the very first step of the very
    /// first control run silently took the whole record down: the CSV wrote,
    /// the receipt recorded "benchmark record not written", and there was no
    /// `.h3-bench.json` anywhere.
    ///
    /// `null` rather than the string `"inf"`, which is the other way out: a
    /// null is what every JSON reader already understands as "no value here",
    /// and it matches what the CSV does with the same quantity. A record whose
    /// numeric column is sometimes a string is not machine-readable in any
    /// useful sense.
    private enum CodingKeys: String, CodingKey {
        case step, branch, sigma, wholeSequenceChange, videoChange, audioChange
        case decision, reason, constraints, advisory, consecutiveSkipsBefore
    }

    package init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        step = try c.decode(Int.self, forKey: .step)
        branch = try c.decode(Branch.self, forKey: .branch)
        func number(_ key: CodingKeys) throws -> Double {
            try c.decodeIfPresent(Double.self, forKey: key) ?? .infinity
        }
        sigma = try number(.sigma)
        wholeSequenceChange = try number(.wholeSequenceChange)
        videoChange = try number(.videoChange)
        audioChange = try number(.audioChange)
        decision = try c.decode(StepCachePolicy.Decision.self, forKey: .decision)
        reason = try c.decode(StepCachePolicy.Reason.self, forKey: .reason)
        // Absent in records written before the constraint list existed. Decoded
        // as the headline alone rather than as empty: those records did have a
        // binding constraint on every full step, and an empty list would read
        // as "nothing objected", which is the opposite.
        constraints = try c.decodeIfPresent([StepCachePolicy.Reason].self, forKey: .constraints)
            ?? (reason == .belowThreshold ? [] : [reason])
        advisory = try c.decodeIfPresent([StepCachePolicy.Reason].self, forKey: .advisory) ?? []
        consecutiveSkipsBefore = try c.decode(Int.self, forKey: .consecutiveSkipsBefore)
    }

    package func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(step, forKey: .step)
        try c.encode(branch, forKey: .branch)
        func put(_ v: Double, _ key: CodingKeys) throws {
            if v.isFinite { try c.encode(v, forKey: key) } else { try c.encodeNil(forKey: key) }
        }
        try put(sigma, .sigma)
        try put(wholeSequenceChange, .wholeSequenceChange)
        try put(videoChange, .videoChange)
        try put(audioChange, .audioChange)
        try c.encode(decision, forKey: .decision)
        try c.encode(reason, forKey: .reason)
        try c.encode(constraints, forKey: .constraints)
        try c.encode(advisory, forKey: .advisory)
        try c.encode(consecutiveSkipsBefore, forKey: .consecutiveSkipsBefore)
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

    /// **The figure to compare configurations on: mean seconds per step.**
    ///
    /// This was the median, and the median was badly wrong. The reasoning was
    /// that the first and last steps run in full under every configuration and
    /// dilute an average — true, and irrelevant next to what it missed. A
    /// cached render's step times are not one distribution with outliers, they
    /// are **two populations**: measured on the first control run, full steps
    /// cost 59.9 s and reused steps cost 1.26 s, with nothing in between. Twelve
    /// full and eight reused puts the median squarely inside the full-step
    /// population, so the cached arm reported 59.7 s — within noise of what
    /// dense would report — and the speed-up came out at roughly 1.00x for a
    /// cache saving nearly half the sampling time.
    ///
    /// The mean is what wall clock is made of, and the first/last-step dilution
    /// it carries is real cost that both arms pay.
    package var meanStepSeconds: Double {
        let f = stepSeconds.filter(\.isFinite)
        guard !f.isEmpty else { return .nan }
        return f.reduce(0, +) / Double(f.count)
    }

    /// Median seconds per step — a **diagnostic, not a comparison figure.**
    ///
    /// Kept because within a single population it says something the mean does
    /// not: paired with `medianFullStepSeconds` it shows whether a change made
    /// individual steps cheaper or merely skipped more of them. Those are
    /// different achievements and only the first one composes with anything
    /// else.
    package var medianStepSeconds: Double { Self.median(stepSeconds) }

    /// Median cost of the steps that actually ran the stack.
    ///
    /// The number a kernel optimisation has to move. A cache change moves the
    /// *count* of these; a fused kernel moves their *cost*. An improvement that
    /// shows up in the mean but not here came from skipping more, which is a
    /// quality trade and not a speed-up.
    package var medianFullStepSeconds: Double {
        // A cache-free render records no decisions, because there is no cache
        // to make any — but every one of its steps ran the full stack. Reading
        // the empty decision list as "no full steps" reported `nan` for the
        // dense control, which is the one arm whose full-step cost a kernel
        // change most needs to be measured against.
        guard !steps.isEmpty else { return Self.median(stepSeconds) }
        return Self.median(secondsOf(.runFull))
    }

    package var medianReusedStepSeconds: Double {
        Self.median(secondsOf(.reuse))
    }

    /// Step times for the steps with a given decision, on the conditional
    /// branch. One branch, because a step's wall clock covers both CFG forwards
    /// and attributing it to each separately would double-count it.
    private func secondsOf(_ decision: StepCachePolicy.Decision) -> [Double] {
        steps.filter { $0.branch == .conditional && $0.decision == decision }
             .compactMap { $0.step < stepSeconds.count ? stepSeconds[$0.step] : nil }
    }

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
                + "decision,reason,constraints,advisory,consecutive_skips_before,step_seconds\n"
        func num(_ x: Double) -> String { x.isFinite ? String(format: "%.6f", x) : "" }
        for t in steps {
            let secs = t.step < stepSeconds.count ? num(stepSeconds[t.step]) : ""
            // Space-separated inside one field: a reader splitting on commas
            // must not find a variable number of columns per row.
            let cons = t.constraints.map(\.rawValue).joined(separator: " ")
            let adv = t.advisory.map(\.rawValue).joined(separator: " ")
            out += "\(t.step),\(t.branch.rawValue),\(num(t.sigma)),"
                 + "\(num(t.wholeSequenceChange)),\(num(t.videoChange)),\(num(t.audioChange)),"
                 + "\(t.decision == .reuse ? "reused" : "full"),\(t.reason.rawValue),"
                 + "\(cons),\(adv),\(t.consecutiveSkipsBefore),\(secs)\n"
        }
        return out
    }
}
