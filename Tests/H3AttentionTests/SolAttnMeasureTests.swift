// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import MLXFast
import H3Foundation
@testable import H3Attention

/// The equivalence class, measured against q/k/v captured from a real render.
///
/// **This is the experiment that decides whether the backend can ship.** The
/// protocol requires a backend to state the relative-RMS band inside which its
/// output counts as the same answer as dense, and conformance gates per backend
/// against that number. Everything else in this test target measures whether
/// the implementation computes Sol-Attn correctly; this measures how far
/// Sol-Attn itself sits from dense on H3's own attention, which is the number
/// the contract is about.
///
/// Synthetic input cannot answer it. Gaussian q/k/v put the method at rel_rms
/// 0.17 at beta = 0 — attention with no dominant blocks has nothing for a
/// sparse method to keep — and structured input flatters it by however much
/// structure the generator was given. Only real tensors settle it.
///
/// Capture them with:
///
///     H3_CAPTURE_QKV=/Volumes/big_daddy/scratch_disk/H3_Swift/qkv \
///     H3_CAPTURE_BLOCKS=0,24,49 H3_CAPTURE_AFTER=0.25 \
///     h3 render --prompt "..." --out /tmp/x.mp4 \
///       --width 864 --height 480 --seconds 5 --steps 20
///
/// then run:
///
///     H3_QKV=/Volumes/big_daddy/scratch_disk/H3_Swift/qkv \
///     swift test -c release --filter equivalenceClassOnRealAttention
@Suite("Sol-Attn measured", .serialized)
struct SolAttnMeasureTests {

    /// The packed layout is `[text][cond][ref][audio][video]`, so conditioning
    /// is everything before the video tail. Derived from the geometry rather
    /// than guessed: at 864x480x124, `latentT` 37 and a 27 x 15 patchified
    /// frame give 14,985 video tokens.
    private func videoSpan(sequenceLength s: Int) -> Range<Int> {
        let g = LatentGeometry(width: 864, height: 480, length: 124)
        return (s - g.videoTokens) ..< s
    }

    private func relRMS(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = a.asType(.float32) - b.asType(.float32)
        return MLX.sqrt(MLX.mean(d * d)).item(Float.self)
             / MLX.sqrt(MLX.mean(b.asType(.float32) * b.asType(.float32))).item(Float.self)
    }

    private func time(_ n: Int = 3, _ body: () -> MLXArray) -> Double {
        MLX.eval(body())
        let t0 = Date()
        for _ in 0 ..< n { MLX.eval(body()) }
        return Date().timeIntervalSince(t0) * 1000 / Double(n)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_QKV"] != nil))
    func equivalenceClassOnRealAttention() throws {
        let dir = ProcessInfo.processInfo.environment["H3_QKV"]!
        let files = try FileManager.default.contentsOfDirectory(atPath: dir)
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        try #require(!files.isEmpty, "no captures in \(dir)")

        for file in files {
            let loaded = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/\(file)"))
            // Captured as [S, heads, d]; the backend takes [heads, S, d].
            guard let q0 = loaded["q"], let k0 = loaded["k"], let v0 = loaded["v"] else { continue }
            let q = q0.transposed(1, 0, 2)
            let k = k0.transposed(1, 0, 2)
            let v = v0.transposed(1, 0, 2)
            let heads = q.dim(0), s = q.dim(1), d = q.dim(2)
            let scale = 1.0 / Float(d).squareRoot()
            let span = videoSpan(sequenceLength: s)
            MLX.eval(q, k, v)

            let want = MLXFast.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
                values: v.expandedDimensions(axis: 0), scale: scale, mask: nil).squeezed(axis: 0)
            MLX.eval(want)
            let denseMs = time {
                MLXFast.scaledDotProductAttention(
                    queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0),
                    values: v.expandedDimensions(axis: 0), scale: scale, mask: nil)
            }

            print("\n\(file)  H=\(heads) S=\(s) D=\(d)  conditioning rows 0..<\(span.lowerBound)")
            print(String(format: "  dense %.1f ms", denseMs))

            print("  beta   density  relRMS    cond-rows  speedup")
            for beta in [Float(0.8), 1.0, 1.2, 1.5, 2.0] {
                var c = SolAttnConfig(); c.beta = beta
                guard let got = SolAttnMetalKernel.attend(queries: q, keys: k, values: v,
                                                          scale: scale, config: c,
                                                          videoSpan: span) else { continue }
                MLX.eval(got)
                let density = SolAttnReference.density(queries: q, keys: k, values: v,
                                                       config: c, videoSpan: span)
                // The conditioning rows carry text, audio and lip-sync; §8 found
                // they degrade differently from the video tail, so they are
                // reported separately rather than averaged away.
                let cond = relRMS(got[0..., 0 ..< span.lowerBound, 0...],
                                  want[0..., 0 ..< span.lowerBound, 0...])
                let ms = time { SolAttnMetalKernel.attend(queries: q, keys: k, values: v,
                                                          scale: scale, config: c,
                                                          videoSpan: span)! }
                print(String(format: "  %.1f    %.3f    %.4f    %.4f     %.2fx",
                             beta, density, relRMS(got, want), cond, denseMs / ms))
            }

            print("  block  density  relRMS    speedup     (one latent frame = 405 tokens)")
            for bs in [64, 128, 256, 384, 448, 512, 1_024, 2_048] {
                var c = SolAttnConfig(); c.blockSize = bs
                guard let got = SolAttnMetalKernel.attend(queries: q, keys: k, values: v,
                                                          scale: scale, config: c,
                                                          videoSpan: span) else {
                    print("  \(bs)  declined"); continue
                }
                MLX.eval(got)
                let density = SolAttnReference.density(queries: q, keys: k, values: v,
                                                       config: c, videoSpan: span)
                let ms = time { SolAttnMetalKernel.attend(queries: q, keys: k, values: v,
                                                          scale: scale, config: c,
                                                          videoSpan: span)! }
                print(String(format: "  %5d  %.3f    %.4f    %.2fx", bs, density,
                             relRMS(got, want), denseMs / ms))
            }

            // The sink costs ~17% (§8 measured that on CUDA) and buys accuracy on
            // exactly the rows that carry dialogue. Worth its own line.
            var noSink = SolAttnConfig(); noSink.exactConditioningKV = false
            if let got = SolAttnMetalKernel.attend(queries: q, keys: k, values: v,
                                                   scale: scale, config: noSink,
                                                   videoSpan: span) {
                MLX.eval(got)
                let cond = relRMS(got[0..., 0 ..< span.lowerBound, 0...],
                                  want[0..., 0 ..< span.lowerBound, 0...])
                let ms = time { SolAttnMetalKernel.attend(queries: q, keys: k, values: v,
                                                          scale: scale, config: noSink,
                                                          videoSpan: span)! }
                print(String(format: "  no sink: relRMS %.4f  cond-rows %.4f  %.2fx",
                             relRMS(got, want), cond, denseMs / ms))
            }
        }
    }
}
