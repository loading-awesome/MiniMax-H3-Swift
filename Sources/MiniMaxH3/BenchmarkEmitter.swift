// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import H3Foundation
import H3Hardware
import H3Modules
import H3Pipeline

/// Writes the benchmark record beside the render, on every render.
///
/// **Unconditional, and that is the design.** A benchmark harness you have to
/// remember to switch on produces evidence for the runs where somebody expected
/// to need it, which is never the run that turns out to matter. The Sol-Attn
/// conclusion had to be revisited because the arms that mattered had been run
/// weeks earlier with a setting nobody recorded; there was no way to check,
/// only to re-run. Two files of a few kilobytes each, written next to a
/// hundred-megabyte mp4, buy the ability to answer that question later instead.
///
/// Nothing here can fail a render. Every path is best-effort and reports through
/// the receipt's warnings — a benchmark that could abort a fifteen-minute render
/// because a directory was read-only would be a worse instrument than none.
enum BenchmarkEmitter {

    /// The pinned mlx-swift version.
    ///
    /// A literal, because there is no runtime API that reports it — and kept
    /// honest by `BenchmarkRecordTests.pinnedMLXVersionMatchesPackageResolved`,
    /// which reads `Package.resolved` and fails if this string has drifted.
    /// Without that test this is a comment that lies the first time the
    /// dependency moves, and a record whose environment field is wrong is worse
    /// than one that has no environment field.
    static let mlxSwiftVersion = "0.31.6"

    static func record(arm: String, request: RenderRequest, result: RenderResult,
                       checkpoints: [RenderReceipt.Checkpoint],
                       machine: Machine, dimensions: (width: Int, height: Int)) -> BenchmarkRecord {
        let dit = checkpoints.first { $0.role == "dit" }
        return BenchmarkRecord(
            arm: arm,
            identity: .init(
                promptDigest: BenchmarkRecord.digest(request.prompt),
                negativePromptDigest: request.negativePrompt.map(BenchmarkRecord.digest),
                seed: request.seed,
                width: dimensions.width, height: dimensions.height,
                seconds: request.seconds, steps: request.steps,
                cfgScale: request.cfgScale,
                // Only a verified checkpoint pins identity. An unverified file
                // is recorded as nil rather than as its filename, so that two
                // records can never appear to agree on a checkpoint neither of
                // them actually hashed.
                ditDigest: dit?.verification == .verified ? dit?.sha256 : nil,
                ditSizeBytes: dit?.sizeBytes ?? 0),
            configuration: .init(
                qualityProfile: request.qualityProfile.rawValue,
                attentionBackend: result.attentionBackend,
                cacheThreshold: request.cacheThreshold,
                cacheMaxConsecutiveSkips: request.cacheMaxSkips,
                cachePerStreamProbe: !request.cacheWholeSequenceProbe,
                overrides: environmentOverrides()),
            machine: .init(
                model: machine.summary,
                cores: ProcessInfo.processInfo.processorCount,
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                libraryVersion: MiniMaxH3.version,
                mlxSwiftVersion: mlxSwiftVersion),
            phaseSeconds: result.timings.asDictionary,
            memory: .init(mlxPeakBytes: result.mlxPeakBytes,
                          mlxActiveBytesAtEnd: result.mlxActiveBytesAtEnd),
            trace: result.trace)
    }

    /// A default name for this run's arm, from what was actually on.
    ///
    /// `H3_BENCH_ARM` overrides it, which is how a sweep labels its rows. The
    /// derived form is deliberately verbose: `cached-0.100-perstream-sdpa` says
    /// what it is, where `run-3` says only when it happened.
    static func armName(request: RenderRequest, result: RenderResult) -> String {
        if let explicit = ProcessInfo.processInfo.environment["H3_BENCH_ARM"],
           !explicit.isEmpty {
            return explicit
        }
        var parts: [String] = []
        parts.append(request.cacheThreshold > 0
                     ? String(format: "cached-%.3f", request.cacheThreshold)
                     : "dense")
        if request.cacheThreshold > 0 {
            parts.append(request.cacheWholeSequenceProbe ? "wholeseq" : "perstream")
        }
        parts.append(result.attentionBackend)
        return parts.joined(separator: "-")
    }

    /// Every `H3_`-prefixed environment variable in force.
    ///
    /// The sweep knobs added during the Sol-Attn work are all environment
    /// variables, and an arm distinguished only by one of them is
    /// indistinguishable from its control in every artefact unless the variable
    /// is written down. Captured wholesale rather than by allow-list, because
    /// the variable that turns out to matter is by definition the one nobody
    /// thought to list.
    ///
    /// `H3_BENCH_ARM` is excluded — it is the label, not part of what was run,
    /// and leaving it in would make two identical configurations compare as
    /// different.
    static func environmentOverrides() -> [String: String] {
        var out = ProcessInfo.processInfo.environment
            .filter { $0.key.hasPrefix("H3_") && $0.key != "H3_BENCH_ARM" }
        // Switches that are **on by default** have to be recorded as resolved
        // values, not just when somebody overrides them. An environment scrape
        // alone records the exception and stays silent about the rule, so a run
        // with fused modulation on and a run from before it existed would both
        // show nothing here — and the fused path is a different computation, by
        // a few ulps, from the one the controls were measured on.
        out["resolved.fusedModulation"] = FusedModulation.enabled ? "on" : "off"
        return out
    }

    /// Writes `<output>.h3-bench.json` and `<output>.h3-steps.csv`.
    /// - Returns: warnings, empty when both landed.
    static func write(_ record: BenchmarkRecord, besideVideo video: URL) -> [String] {
        let base = video.deletingPathExtension()
        var warnings: [String] = []
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        do {
            try encoder.encode(record)
                .write(to: base.appendingPathExtension("h3-bench.json"), options: .atomic)
        } catch {
            warnings.append("benchmark record not written: \(error.localizedDescription)")
        }
        do {
            try Data(record.trace.csv.utf8)
                .write(to: base.appendingPathExtension("h3-steps.csv"), options: .atomic)
        } catch {
            warnings.append("step trace not written: \(error.localizedDescription)")
        }
        return warnings
    }
}
