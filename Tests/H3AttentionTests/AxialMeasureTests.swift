// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import H3Foundation
@testable import H3Attention

/// **The stop condition for Phase 6E**, measured on q/k/v captured from a real
/// render rather than on Gaussian noise.
///
/// Synthetic input cannot answer this. Attention over random vectors has no
/// spatial or temporal structure, so a topology built entirely out of spatial
/// and temporal structure has nothing to exploit and would look far worse than
/// it is. Structured synthetic input has the opposite problem — it flatters the
/// method by exactly the structure the generator was given. Only H3's own
/// tensors settle it.
///
/// The error is reported **per span, never averaged**. §8 of `SOL_ATTN.md`
/// found the conditioning rows degrade differently from the video tail, and a
/// whole-tensor figure hides that: the prefix is 4.7% of the sequence at
/// production shape, so a catastrophic prefix error moves a global average by
/// almost nothing. Those rows carry the prompt, the references and the target
/// audio — the ones a viewer notices first when they go wrong.
///
///     H3_QKV=/Volumes/big_daddy/scratch_disk/H3_Swift/qkv \
///     swift test -c release --filter axialErrorOnRealAttention
///
/// **What would stop 6E:** prefix or audio error outside the band dense
/// backends are held to, or error that swings between consecutive captured
/// steps — an operator whose accuracy is unstable along the trajectory is the
/// mechanism behind the pulsing that Sol-Attn produced and no tensor metric
/// caught.
@Suite("axial measured", .serialized)
struct AxialMeasureTests {

    /// Production geometry: 864×480×124 gives latentT 37 on a 27×15 patchified
    /// frame, so 14,985 video tokens and everything before them is prefix.
    private func topology(sequenceLength s: Int, landmarks: Int) -> AxialTopology? {
        let g = LatentGeometry(width: 864, height: 480, length: 124)
        return AxialTopology(sequenceLength: s,
                             videoSpan: (s - g.videoTokens) ..< s,
                             geometry: g, landmarks: landmarks)
    }

    private func relRMS(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = a.asType(.float32) - b.asType(.float32)
        return MLX.sqrt(MLX.mean(d * d)).item(Float.self)
             / MLX.sqrt(MLX.mean(b.asType(.float32) * b.asType(.float32))).item(Float.self)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_QKV"] != nil))
    func axialErrorOnRealAttention() throws {
        let dir = ProcessInfo.processInfo.environment["H3_QKV"]!
        let files = try FileManager.default.contentsOfDirectory(atPath: dir)
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        try #require(!files.isEmpty, "no captures in \(dir)")

        for file in files {
            let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/\(file)"))
            guard let q0 = loaded["q"], let k0 = loaded["k"], let v0 = loaded["v"] else { continue }
            // Captured as [S, heads, d]; the topology takes [heads, S, d].
            let q = q0.transposed(1, 0, 2), k = k0.transposed(1, 0, 2), v = v0.transposed(1, 0, 2)
            let s = q.dim(1), d = q.dim(2)
            let scale = 1.0 / Float(d).squareRoot()
            MLX.eval(q, k, v)

            let want = MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0).asType(.float32),
                keys: k.expandedDimensions(axis: 0).asType(.float32),
                values: v.expandedDimensions(axis: 0).asType(.float32),
                scale: scale, mask: nil).squeezed(axis: 0)
            MLX.eval(want)

            guard let probe = topology(sequenceLength: s, landmarks: 3) else {
                print("\n\(file): S=\(s) does not match the production lattice — skipped")
                continue
            }
            let prefix = probe.prefix

            print("\n\(file)  S=\(s) D=\(d)  prefix rows 0..<\(prefix.count) "
                  + "(\(String(format: "%.1f", 100 * Double(prefix.count) / Double(s)))%)")
            print("  landmarks  density   total    prefix    video")
            for landmarks in [2, 3, 5, 9] {
                guard let t = topology(sequenceLength: s, landmarks: landmarks) else { continue }
                let got = AxialReference.attend(queries: q, keys: k, values: v,
                                                scale: scale, topology: t)
                MLX.eval(got)
                let total = relRMS(got, want)
                let pfx = relRMS(got[0..., prefix, 0...], want[0..., prefix, 0...])
                let vid = relRMS(got[0..., t.videoSpan, 0...], want[0..., t.videoSpan, 0...])
                print(String(format: "  %9d   %.3f   %.4f   %.4f   %.4f",
                             landmarks, t.density, total, pfx, vid))
            }
        }
    }

    /// Does the error hold still between consecutive sampling steps?
    ///
    /// **A separate question from how large it is, and the one that killed
    /// Sol-Attn.** Its per-call error was 0.29 — a number that looked
    /// survivable — while the output pulsed, because the operator changed from
    /// step to step. A deterministic topology should not have that failure by
    /// construction: the same query attends the same keys at every step. This
    /// measures whether that holds in the numbers rather than assuming it from
    /// the design.
    ///
    ///     H3_CHURN=/Volumes/big_daddy/scratch_disk/H3_Swift/qkv_churn \
    ///     swift test -c release --filter axialErrorStability
    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_CHURN"] != nil))
    func axialErrorStability() throws {
        let dir = ProcessInfo.processInfo.environment["H3_CHURN"]!
        let files = try FileManager.default.contentsOfDirectory(atPath: dir)
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        try #require(files.count >= 2, "need consecutive steps in \(dir)")

        print("\naxial error across consecutive steps — a swing here is the "
              + "mechanism behind pulsing")
        print("  step                          total    prefix     video")
        for file in files {
            let a = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/\(file)"))
            let q = a["q"]!.transposed(1, 0, 2)
            let k = a["k"]!.transposed(1, 0, 2)
            let v = a["v"]!.transposed(1, 0, 2)
            let s = q.dim(1)
            let scale = 1.0 / Float(q.dim(2)).squareRoot()
            guard let t = topology(sequenceLength: s, landmarks: 3) else { continue }
            let want = MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0).asType(.float32),
                keys: k.expandedDimensions(axis: 0).asType(.float32),
                values: v.expandedDimensions(axis: 0).asType(.float32),
                scale: scale, mask: nil).squeezed(axis: 0)
            let got = AxialReference.attend(queries: q, keys: k, values: v,
                                            scale: scale, topology: t)
            MLX.eval(got, want)
            let name = file.replacingOccurrences(of: ".safetensors", with: "")
            print(String(format: "  %-28@ %.4f   %.4f   %.4f", name as NSString,
                         relRMS(got, want),
                         relRMS(got[0..., t.prefix, 0...], want[0..., t.prefix, 0...]),
                         relRMS(got[0..., t.videoSpan, 0...], want[0..., t.videoSpan, 0...])))
        }
        print("")
    }
}
