// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import Foundation
import MLX
import H3Foundation
@testable import H3Attention

/// Does the router pick the same blocks at step k and step k+1?
///
/// **This is the question that decides which fix the temporal artifact needs.**
/// A viewer watching a sparse render reported the subject "pulsing and warping",
/// and a localised temporal measurement confirmed it — roughly twice dense's
/// frame-to-frame acceleration in the quiet shot. Two mechanisms could produce
/// that, and they call for completely different work:
///
///  * **Routing churn.** The selection is recomputed independently every step
///    with nothing tying it to the previous one. Any block whose proxy score
///    sits near `tau` flips in and out, so the operator changes discontinuously
///    mid-trajectory and a region near the threshold oscillates. The fix is
///    hysteresis — keep a block selected unless its score falls clearly below
///    the threshold — which is pure Swift, needs no kernel change, and **keeps
///    the sparsity**.
///  * **Block geometry.** At blockSize 64 a routing block is 2.4 rows of one
///    latent frame, a horizontal sliver, so adjacent slivers of one object can
///    disagree. The fix is Morton ordering, which is a declared seam with no
///    implementation behind it and a much larger job.
///
/// The distinction matters because the artifact was only removed in the sweep by
/// *giving up sparsity* — forcing more blocks and more steps dense, which cut
/// sparse coverage from 77% of block-steps to 52% and with it almost all of the
/// speedup. Hysteresis would remove it without paying that.
///
/// High overlap between adjacent steps exonerates churn and points at geometry.
/// Low overlap is the answer.
///
///     H3_CHURN=/Volumes/big_daddy/scratch_disk/H3_Swift/qkv_churn \
///     swift test -c release --filter routingStabilityAcrossSteps
@Suite("Sol-Attn churn")
struct SolAttnChurnTests {

    @Test(.enabled(if: ProcessInfo.processInfo.environment["H3_CHURN"] != nil))
    func routingStabilityAcrossSteps() throws {
        let dir = ProcessInfo.processInfo.environment["H3_CHURN"]!
        let files = try FileManager.default.contentsOfDirectory(atPath: dir)
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        try #require(files.count >= 2, "need at least two consecutive steps in \(dir)")

        func selection(_ file: String, beta: Float, blockSize: Int) throws -> MLXArray {
            let a = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/\(file)"))
            let q = a["q"]!.transposed(1, 0, 2)
            let k = a["k"]!.transposed(1, 0, 2)
            let v = a["v"]!.transposed(1, 0, 2)
            let s = q.dim(1)
            let g = LatentGeometry(width: 864, height: 480, length: 124)
            let pooling = SolAttnRouting.pool(keys: k, values: v, queryCount: s,
                                              blockSize: blockSize)
            let pq = SolAttnRouting.poolQueries(q, blockSize: blockSize)
            let sink = SolAttnRouting.sinkKeyBlocks(videoSpan: (s - g.videoTokens) ..< s,
                                                    blockSize: blockSize, enabled: true)
            return SolAttnRouting.select(pooledQueries: pq, pooling: pooling,
                                         beta: beta, sinkKeyBlocks: sink)
        }

        print("\nrouting stability between consecutive steps — block 24")
        for beta in [Float(0.8), 1.2] {
            print("  beta \(beta)")
            for i in 0 ..< (files.count - 1) {
                let a = try selection(files[i], beta: beta, blockSize: 64)
                let b = try selection(files[i + 1], beta: beta, blockSize: 64)
                let af = a.asType(.float32), bf = b.asType(.float32)
                let inter = MLX.sum(af * bf).item(Float.self)
                let union = MLX.sum(MLX.maximum(af, bf)).item(Float.self)
                let flips = MLX.mean(MLX.abs(af - bf)).item(Float.self)
                let da = MLX.mean(af).item(Float.self), db = MLX.mean(bf).item(Float.self)
                print(String(format:
                    "    %@ -> %@   jaccard %.3f   flipped %.1f%% of all blocks   density %.3f -> %.3f",
                    files[i].replacingOccurrences(of: ".safetensors", with: ""),
                    files[i + 1].replacingOccurrences(of: ".safetensors", with: ""),
                    inter / max(union, 1), flips * 100, da, db))
            }
        }
        print("")
    }
}
