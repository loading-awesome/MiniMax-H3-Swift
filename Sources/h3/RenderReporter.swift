// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MiniMaxH3

/// What a render looks like from the outside.
///
/// A render is fifteen to forty minutes of silence unless something says
/// otherwise, and "is it stuck or is it working" is the first question anybody
/// asks. So the sampling line carries a bar, the measured seconds per step, and
/// a finish time that is derived from *this* render rather than from a table.
///
/// **It behaves differently into a terminal and into a file, deliberately.** On
/// a terminal the step line rewrites itself in place, which is what you want to
/// watch. Redirected to a log it prints one line per step, because a log full of
/// carriage returns is unreadable and because a render that is killed takes the
/// last partial line with it.
struct RenderReporter {

    private let interactive = isatty(fileno(stdout)) == 1
    private let started = Date()
    private var stepTimes: [TimeInterval] = []
    private var lastStepAt: Date?
    private var phaseIndex = 0
    private var currentPhase: RenderProgress.Phase?

    /// Phases the user actually waits through, in order. `conditionEncoding`
    /// is folded into the first one: on a text-only render it is 0.0 s, and a
    /// step that always reports zero teaches people to ignore the display.
    private static let visible: [RenderProgress.Phase] = [
        .textConditioning, .sampling, .decoding, .writing,
    ]

    private static func label(_ phase: RenderProgress.Phase) -> String {
        switch phase {
        case .textConditioning:  "reading the prompt"
        case .conditionEncoding: "encoding conditioning"
        case .sampling:          "sampling"
        case .decoding:          "decoding video and audio"
        case .writing:           "writing the file"
        }
    }

    // MARK: the plan, before anything loads

    /// `fps` is passed in rather than read from `H3Video`: this type is
    /// presentation, and the one constant it needs is not worth a dependency on
    /// a package-internal module.
    static func plan(request: RenderRequest, width: Int, height: Int,
                     frameCount: Int, fps: Int, packedTokens: Int, checkpoint: String,
                     estimateSeconds: Double) {
        func row(_ k: String, _ v: String) {
            print("  \(k.padding(toLength: 10, withPad: " ", startingAt: 0))\(v)")
        }
        print("")
        row("making", request.modeDescription)
        row("size", "\(width) x \(height), \(request.seconds) s at \(fps) fps "
                  + "(\(frameCount) frames)")
        row("steps", "\(request.steps)")
        row("quality", request.qualityProfile.isApproximate
            ? "\(request.qualityProfile.rawValue) — an approximation, "
              + "cache threshold \(request.cacheThreshold)"
            : "\(request.qualityProfile.rawValue) — no approximation")
        row("model", checkpoint)
        if request.cfgScale > 1 {
            row("guidance", String(format: "%.2f (two forward passes per step)",
                                   request.cfgScale))
        }
        row("estimate", "about \(duration(estimateSeconds)) of sampling")
        print("")
    }

    // MARK: progress

    mutating func observe(_ p: RenderProgress) {
        if p.phase != currentPhase {
            finishLine()
            currentPhase = p.phase
            // A phase not on the visible list prints nothing at all. Printing it
            // under the *previous* phase's number is worse than silence: it read
            // as "[1/4]" twice in a row, which looks like the render restarted.
            if let i = Self.visible.firstIndex(of: p.phase) {
                phaseIndex = i + 1
                if p.phase != .sampling {
                    print("[\(phaseIndex)/\(Self.visible.count)] \(Self.label(p.phase))")
                }
            }
            lastStepAt = Date()
        }
        guard p.phase == .sampling, p.total > 0, p.completed > 0 else { return }

        if let last = lastStepAt { stepTimes.append(Date().timeIntervalSince(last)) }
        lastStepAt = Date()

        // **The mean over every step so far, not a recent window** — and the
        // first version of this had it exactly backwards. Under a quality
        // profile that reuses work, step cost is bimodal: a full step is ~100 s
        // and a reused one is ~2 s. A three-step window lands entirely inside
        // one mode or the other, so the countdown swung between "28m left" and
        // "29s left" on consecutive steps of the same render. The mean is both
        // steadier and the correct estimator when the two kinds interleave.
        let perStep = stepTimes.reduce(0, +) / Double(stepTimes.count)
        let remaining = Double(p.total - p.completed) * perStep

        let bar = Self.bar(p.completed, p.total, width: 22)
        let line = String(format: "[%d/%d] sampling  %@  %2d/%d   %.1f s/step   %@ left",
                          phaseIndex, Self.visible.count, bar, p.completed, p.total,
                          perStep, duration(remaining))
        if interactive {
            print("\r\(line)\u{1B}[K", terminator: "")
            fflush(stdout)
        } else {
            print(line)
        }
    }

    /// Close an in-place line before anything else prints over it.
    mutating func finishLine() {
        if interactive && currentPhase == .sampling && !stepTimes.isEmpty { print("") }
    }

    // MARK: the end

    mutating func finished(_ result: RenderResult) {
        finishLine()
        let wall = Date().timeIntervalSince(started)
        print("")
        print("done in \(duration(wall))")
        print("  video  \(result.video.path)")
        print("         \(result.width)x\(result.height), \(result.frameCount) frames, "
              + String(format: "%.2f s", result.seconds))
        if let audio = result.audio { print("  audio  \(audio.path)") }

        let t = result.timings
        print("")
        print("  where the time went")
        for (name, seconds) in [("prompt", t.textConditioning),
                                ("conditioning", t.conditionEncoding),
                                ("model load", t.modelLoad),
                                ("sampling", t.sampling),
                                ("decode", t.audioDecode + t.videoDecode),
                                ("write", t.pixelPack + t.mux)] where seconds > 0.05 {
            let share = wall > 0 ? seconds / wall * 100 : 0
            print(String(format: "    %-14@%8.1fs  %2.0f%%", name as NSString, seconds, share))
        }
        // The comparable figure. Wall clock includes a checkpoint load whose
        // cost depends on what the page cache happened to be holding, so two
        // runs of the same configuration can differ by a minute for reasons
        // that have nothing to do with either.
        let perStep = result.trace.meanStepSeconds
        if perStep.isFinite, perStep > 0 {
            print(String(format: "    %-14@%8.2fs  mean, the figure to compare on",
                         "per step" as NSString, perStep))
            // Both numbers, because they answer different questions: how long
            // the render took, and how expensive a step that actually ran the
            // stack was. A cache changes the first, a kernel changes the second.
            let full = result.trace.medianFullStepSeconds
            if full.isFinite, full > 0, result.trace.stepsSkipped > 0 {
                print(String(format: "    %-14@%8.2fs  median of the %d steps that ran the stack",
                             "full step" as NSString, full, result.trace.stepsRun))
            }
        }
        if let summary = result.cacheSummary { print("\n  \(summary)") }
    }

    // MARK: formatting

    private static func bar(_ done: Int, _ total: Int, width: Int) -> String {
        let filled = total > 0 ? Int((Double(done) / Double(total) * Double(width)).rounded()) : 0
        return String(repeating: "█", count: max(0, min(width, filled)))
             + String(repeating: "·", count: max(0, width - filled))
    }

    private func duration(_ s: TimeInterval) -> String { Self.durationText(s) }
    private static func duration(_ s: TimeInterval) -> String { durationText(s) }

    /// Minutes and seconds, because "1287 s" is a number people have to convert
    /// before it means anything.
    static func durationText(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "unknown" }
        let total = Int(seconds.rounded())
        if total < 60 { return "\(total)s" }
        if total < 3600 { return "\(total / 60)m \(String(format: "%02ds", total % 60))" }
        return "\(total / 3600)h \(String(format: "%02dm", (total % 3600) / 60))"
    }
}
