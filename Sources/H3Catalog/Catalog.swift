import Foundation
import H3Foundation

/// Turns a configuration plus a requested mode and precision into the exact
/// files that will be opened, with every one identified from its own header.
///
/// The point of doing this as one resolution step, before anything is loaded,
/// is that **all the ways this can be wrong are cheap to detect and expensive
/// to discover later**: a missing file, an unidentifiable vendor, the wrong
/// partition for the mode, an approximation used without permission. Each of
/// those either crashes at minute twenty or, worse, renders something
/// plausible. Reading four safetensors headers takes milliseconds.
package struct Catalog: Sendable {

    package struct Resolution: Sendable {
        package let mode: RenderMode
        package let precision: String
        package let dit: CheckpointIdentity
        package let textEncoder: CheckpointIdentity
        package let videoVAE: CheckpointIdentity
        package let audioVAE: CheckpointIdentity
        package let tokenizerDirectory: URL?

        package var description: String {
            var out = ["  mode           \(mode.rawValue)  (needs the "
                       + "\(mode.requiredPartition.rawValue) partition)",
                       "  precision      \(precision)",
                       "  dit            \(dit.summary)",
                       "  text encoder   \(textEncoder.summary)",
                       "  video vae      \(videoVAE.summary)",
                       "  audio vae      \(audioVAE.summary)"]
            out.append("  tokenizer      " + (tokenizerDirectory?.path ?? "NOT CONFIGURED"))
            return out.joined(separator: "\n")
        }
    }

    let config: H3Configuration
    package init(config: H3Configuration) { self.config = config }

    /// Every configured checkpoint, identified, whether or not it is usable.
    ///
    /// This is what `h3 doctor` prints. It reports rather than throws, because
    /// the value of a diagnostic is seeing *all* the problems at once — a
    /// doctor that stops at the first missing file makes you run it four times.
    package func inventory() -> [(role: String, path: String, result: Result<CheckpointIdentity, Error>)] {
        var rows: [(String, String, Result<CheckpointIdentity, Error>)] = []
        /// `expect` cross-checks the slot the file is filed under against the
        /// partition the file says it is.
        ///
        /// **This is the check that would have caught the mistake that started
        /// all of this.** A run on 2026-08-05 rendered every reference case
        /// against the FL2VA checkpoint, because one path was hard-coded and
        /// nothing compared it to the mode. The file loads, the architecture
        /// matches, the render succeeds, and the weights were never trained to
        /// consume reference blocks. A config that names an FL2VA file in the
        /// `ref2va` slot is now a reported problem, not an "ok".
        func add(_ role: String, _ raw: String?,
                 expect: CheckpointIdentity.Partition? = nil) {
            guard let raw, !raw.isEmpty else { return }
            guard let url = config.resolve(raw) else { return }
            guard FileManager.default.fileExists(atPath: url.path) else {
                rows.append((role, url.path,
                             .failure(H3Error.checkpointMissing(role: role, path: url.path))))
                return
            }
            let result = Result { try CheckpointIdentity.identify(url: url) }
            if case .success(let id) = result, let expect,
               let actual = id.partition, actual != expect {
                rows.append((role, url.path, .failure(H3Error.wrongPartition(
                    needed: expect.rawValue, loaded: actual.rawValue,
                    mode: "the \(expect.rawValue) slot in this configuration"))))
                return
            }
            rows.append((role, url.path, result))
        }
        for (k, v) in config.checkpoints.fl2va.sorted(by: { $0.key < $1.key }) {
            add("dit fl2va/\(k)", v, expect: .fl2va)
        }
        for (k, v) in config.checkpoints.ref2va.sorted(by: { $0.key < $1.key }) {
            add("dit ref2va/\(k)", v, expect: .ref2va)
        }
        for (k, v) in config.checkpoints.textEncoder.sorted(by: { $0.key < $1.key }) {
            add("text-encoder/\(k)", v)
        }
        add("video vae", config.checkpoints.videoVAE)
        add("audio vae", config.checkpoints.audioVAE)
        return rows
    }

    /// The files a render will actually open, validated for the mode.
    ///
    /// `precision` names a key in the config's per-partition table, not a
    /// dtype: the caller has already been told by the memory planner which key
    /// fits, and the catalog's job is to find it and prove it is what it claims.
    package func resolve(mode: RenderMode, precision: String) throws -> Resolution {
        let ditTable = mode.requiredPartition == .ref2va
            ? config.checkpoints.ref2va : config.checkpoints.fl2va
        let role = "dit \(mode.requiredPartition.rawValue)/\(precision)"

        guard let ditPath = ditTable[precision], let ditURL = config.resolve(ditPath) else {
            let have = ditTable.keys.sorted().joined(separator: ", ")
            throw H3Error.checkpointMissing(
                role: role,
                path: "no '\(precision)' entry for \(mode.requiredPartition.rawValue) "
                    + "in the configuration (have: \(have.isEmpty ? "nothing" : have))")
        }
        let dit = try identify(role, ditURL)
        try dit.validate(forMode: mode,
                         allowApproximate: config.policy.allowApproximateWeights)

        // The text encoder has no partition, so its only per-mode requirement
        // is that it exists at the requested precision — falling back a tier
        // silently would change the conditioning without changing the render's
        // description of itself.
        let tePrecision = precision.hasPrefix("pruned") ? String(precision.dropFirst(7)) : precision
        guard let tePath = config.checkpoints.textEncoder[tePrecision],
              let teURL = config.resolve(tePath) else {
            throw H3Error.checkpointMissing(
                role: "text encoder/\(tePrecision)",
                path: "no '\(tePrecision)' text-encoder entry in the configuration")
        }
        guard let vvPath = config.checkpoints.videoVAE, let vvURL = config.resolve(vvPath) else {
            throw H3Error.checkpointMissing(role: "video vae", path: "not configured")
        }
        guard let avPath = config.checkpoints.audioVAE, let avURL = config.resolve(avPath) else {
            throw H3Error.checkpointMissing(role: "audio vae", path: "not configured")
        }

        return Resolution(mode: mode, precision: precision, dit: dit,
                          textEncoder: try identify("text encoder", teURL),
                          videoVAE: try identify("video vae", vvURL),
                          audioVAE: try identify("audio vae", avURL),
                          tokenizerDirectory: config.resolve(config.checkpoints.tokenizer))
    }

    private func identify(_ role: String, _ url: URL) throws -> CheckpointIdentity {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw H3Error.checkpointMissing(role: role, path: url.path)
        }
        return try CheckpointIdentity.identify(url: url)
    }
}
