// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX

/// Dumps the sampler's per-step inputs and outputs so they can be diffed
/// against the reference pipeline's, tensor by tensor.
///
/// Four readings of the DMD audio step all produced renders that looked right
/// and sounded wrong, and each was argued for from the reference source rather
/// than measured against it. Reading source is how those four were chosen; it
/// is not how the right one gets identified. This writes the same tensors
/// FastVideo's patched denoising stage writes, so `Tools/FastH3/compare_trajectory.py`
/// can name the first step that diverges and, from the velocity column, say
/// whether the fault is the update rule or the forward pass.
///
/// Off unless `H3_CAPTURE_DIR` is set: it forces an eval and a host copy of
/// every latent at every step, which is not something to pay for in a render.
struct TrajectoryCapture {

    let root: URL

    /// Nil unless `H3_CAPTURE_DIR` names a directory we can create.
    init?(log: (String) -> Void) {
        guard let dir = ProcessInfo.processInfo.environment["H3_CAPTURE_DIR"],
              !dir.isEmpty else { return nil }
        let url = URL(fileURLWithPath: dir, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: url,
                                                    withIntermediateDirectories: true)
        } catch {
            log("  capture: cannot use \(dir): \(error.localizedDescription)")
            return nil
        }
        self.root = url
        log("  capture: writing trajectory tensors to \(dir)")
    }

    /// numpy cannot read a bfloat16 `.npy`, and the comparison is a cosine —
    /// float32 costs nothing it can measure and everything reads it.
    func write(_ array: MLXArray, _ name: String) {
        let url = root.appendingPathComponent("\(name).npy")
        do {
            try save(array: array.asType(.float32), url: url)
        } catch {
            FileHandle.standardError.write(
                Data("capture: \(name) failed: \(error)\n".utf8))
        }
    }

    func write(text: String, _ name: String) {
        try? text.write(to: root.appendingPathComponent(name),
                        atomically: true, encoding: .utf8)
    }

    /// Shapes go in the manifest because the comparison flattens: a layout
    /// permutation and a wrong update rule both read as a low cosine, and only
    /// the shapes tell them apart.
    func note(_ line: String) {
        let url = root.appendingPathComponent("meta.txt")
        let entry = line + "\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(Data(entry.utf8))
            try? handle.close()
        } else {
            try? entry.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
