// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXFast
import H3Foundation

/// The block-sparse attention kernel, in Metal.
///
/// **This is a reimplementation, not a binding.** The published Sol-Attn kernels
/// are Triton (`kijai/ComfyUI-SolAttn_triton`) and CUDA (NVIDIA's `sol-engine`),
/// and Triton does not target Metal. What transfers is the method; none of the
/// code does. `SolAttnReference` is the definition this must match.
///
/// ## Shape of the thing
///
/// One threadgroup per (head, query block). 256 threads cover 64 query rows,
/// four lanes to a row, each lane owning 32 of the 128 head dimensions. Scores
/// need the whole 128-wide dot product, so the four lanes reduce across
/// themselves with `quad_shuffle_xor` — the lanes serving one row are exactly a
/// quad, which is why the row-major thread mapping is not arbitrary.
///
/// **The selection is read as flags rather than as a compacted index list.**
/// Compacting a ragged per-(head, query block) selection is real work on the
/// MLX side, and it buys nothing here: a threadgroup scans 247 bytes, and every
/// thread in it shares one `(head, query block)` so the branch is uniform. That
/// uniformity is also what makes the `threadgroup_barrier` calls inside the
/// selected branch safe — a barrier reached by only part of a threadgroup is
/// undefined behaviour, and a compacted list would not have changed that, but a
/// per-*thread* selection would have made it impossible.
///
/// ## Numerics
///
/// One running `(m, l, acc)` per row, updated by both the exact tiles and the
/// rank-1 corrections, in that block order. Rescaling once per tile rather than
/// once per key is the usual flash-attention arrangement and is why the scores
/// for a whole tile are computed before any of them are exponentiated.
///
/// `m` starts at `-3e38` rather than `-INFINITY` on purpose: the first rescale
/// computes `exp(m - m_new)`, and with both terms infinite that is `exp(NaN)`.
/// A large finite sentinel underflows to zero instead, which is the answer that
/// was wanted.
package enum SolAttnMetalKernel {

    /// Query rows per threadgroup, and threads per threadgroup.
    ///
    /// 64 rows x 4 lanes = 256 threads. Registers per thread are
    /// `32 q + 32 acc + 32 scores` floats; at 256 threads that is 96 KB of
    /// register file per threadgroup, which leaves several resident per core on
    /// an M-series GPU.
    package static let queryRowsPerGroup = 64
    package static let lanesPerRow = 4
    package static let threadsPerGroup = 256
    /// Key rows staged in threadgroup memory at a time.
    ///
    /// 32 rows x 128 dims x 2 bytes, for K and V, is 16 KB — half the 32 KB
    /// threadgroup allowance, so occupancy is not gated by it. Storing the tiles
    /// as `half` rather than `float` is what buys that: `half` carries ten
    /// mantissa bits against bf16's seven, so nothing is lost relative to the
    /// input it came from.
    package static let keyTileRows = 32

    private static let source = """
        const uint qb   = threadgroup_position_in_grid.x;
        const uint h    = threadgroup_position_in_grid.y;
        const uint tid  = thread_index_in_threadgroup;
        const uint row  = tid / LANES;
        const uint lane = tid % LANES;

        constexpr uint DPT = D / LANES;
        constexpr uint DV  = DPT / 4;          // the lane's slice, in float4s
        const uint gRow = qb * BQ + row;

        threadgroup half ktile[TILE * D];
        threadgroup half vtile[TILE * D];

        // Vectorised four wide throughout. The scalar version of these loops
        // measured 1.80x against dense; the loads, not the arithmetic, were the
        // cost. `lane * DPT` is a multiple of 32, so every `half4` and `float4`
        // access below is aligned by construction.
        float4 qf[DV];
        float4 acc[DV];
        float sc[TILE];

        // Out-of-range rows still run: they must reach every barrier the rest of
        // the threadgroup reaches. They read zeros and their result is dropped.
        const uint qBase = (h * (uint)S + min(gRow, (uint)S - 1)) * (uint)D + lane * DPT;
        for (uint i = 0; i < DV; ++i) {
            const uint b = qBase + i * 4;
            qf[i] = (gRow < (uint)S)
                  ? float4((float)q[b], (float)q[b+1], (float)q[b+2], (float)q[b+3])
                  : float4(0.0f);
            acc[i] = float4(0.0f);
        }

        float m = -3.0e38f;
        float l = 0.0f;

        // The kernel's query blocking (BQ) and the router's (BS) are separate
        // sizes: BQ is chosen for the register file, BS for how precisely the
        // selection should follow the attention structure. This maps one to the
        // other. It is exact only because the caller guarantees BS is a whole
        // multiple of BQ, which is what stops a threadgroup's rows from
        // straddling two routing blocks with different selections.
        const uint routingQB = (qb * (uint)BQ) / (uint)BS;
        const uint selBase = (h * (uint)NQ + routingQB) * (uint)NK;

        for (uint j = 0; j < (uint)NK; ++j) {
            const uint bStart = j * (uint)BS;
            if (bStart >= (uint)S) break;

            if (sel[selBase + j] != 0) {
                const uint bEnd = min(bStart + (uint)BS, (uint)S);
                for (uint ts = bStart; ts < bEnd; ts += (uint)TILE) {
                    const uint rows = min((uint)TILE, bEnd - ts);

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint i = tid; i < rows * (uint)D; i += (uint)THREADS) {
                        const uint src = (h * (uint)S + ts) * (uint)D + i;
                        ktile[i] = (half)k[src];
                        vtile[i] = (half)v[src];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    float tmax = -3.0e38f;
                    for (uint t = 0; t < rows; ++t) {
                        const threadgroup half4* kt =
                            (const threadgroup half4*)(ktile + t * (uint)D + lane * DPT);
                        float part = 0.0f;
                        for (uint i = 0; i < DV; ++i) part += dot(qf[i], float4(kt[i]));
                        part += quad_shuffle_xor(part, 1u);
                        part += quad_shuffle_xor(part, 2u);
                        sc[t] = part * scale;
                        tmax = max(tmax, sc[t]);
                    }

                    const float mnew = max(m, tmax);
                    const float corr = exp(m - mnew);
                    for (uint i = 0; i < DV; ++i) acc[i] *= corr;

                    float lsum = 0.0f;
                    for (uint t = 0; t < rows; ++t) {
                        const float p = exp(sc[t] - mnew);
                        lsum += p;
                        const threadgroup half4* vt =
                            (const threadgroup half4*)(vtile + t * (uint)D + lane * DPT);
                        for (uint i = 0; i < DV; ++i) acc[i] += p * float4(vt[i]);
                    }
                    l = l * corr + lsum;
                    m = mnew;
                }
            } else {
                // Rank-1 stand-in: one score against the block's pooled key,
                // carrying the block's summed values and its row count.
                const uint pb = (h * (uint)NK + j) * (uint)D + lane * DPT;
                const device float4* pk = (const device float4*)(pooledK + pb);
                const device float4* sv = (const device float4*)(summedV + pb);
                float part = 0.0f;
                for (uint i = 0; i < DV; ++i) part += dot(qf[i], pk[i]);
                part += quad_shuffle_xor(part, 1u);
                part += quad_shuffle_xor(part, 2u);
                const float pj = part * scale;

                const float mnew = max(m, pj);
                const float corr = exp(m - mnew);
                const float pe   = exp(pj - mnew);
                for (uint i = 0; i < DV; ++i) acc[i] = acc[i] * corr + pe * sv[i];
                l = l * corr + pe * counts[j];
                m = mnew;
            }
        }

        if (gRow < (uint)S) {
            const uint oBase = (h * (uint)S + gRow) * (uint)D + lane * DPT;
            const float inv = 1.0f / l;
            for (uint i = 0; i < DV; ++i) {
                out[oBase + i*4 + 0] = acc[i].x * inv;
                out[oBase + i*4 + 1] = acc[i].y * inv;
                out[oBase + i*4 + 2] = acc[i].z * inv;
                out[oBase + i*4 + 3] = acc[i].w * inv;
            }
        }
        """

    /// Built once. MLX keys its compile cache on the kernel name plus the
    /// template arguments — **and not on the input dtypes**, which is why `T`
    /// is a template argument here even though the body never names it. Without
    /// it a float32 call and a bf16 call hash to the same kernel and the second
    /// one silently runs the first one's compiled code against the wrong
    /// element type.
    private static let kernel = MLXFast.metalKernel(
        name: "solattn",
        inputNames: ["q", "k", "v", "pooledK", "summedV", "counts", "sel", "scale"],
        outputNames: ["out"],
        source: source)

    /// Runs the sparse attention. `[H, S, D]` in, `[H, S, D]` float32 out.
    ///
    /// Returns `nil` for any shape this kernel was not written for, so the
    /// caller falls back to dense rather than this silently doing something
    /// approximate at a shape nobody measured.
    package static func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                               scale: Float, config: SolAttnConfig,
                               videoSpan: Range<Int>?) -> MLXArray? {
        let heads = queries.dim(0)
        let s = queries.dim(1)
        let d = queries.dim(2)
        guard d % (lanesPerRow * 4) == 0 else { return nil }
        guard config.blockSize % keyTileRows == 0 else { return nil }
        // A threadgroup covers `queryRowsPerGroup` consecutive rows and reads
        // one routing selection for all of them, so a routing block must be a
        // whole number of threadgroups. Equality is not required and requiring
        // it is what silently sent every blockSize but 64 to the dense path.
        guard config.blockSize % queryRowsPerGroup == 0 else { return nil }
        guard keys.dim(1) == s, values.dim(1) == s else { return nil }

        let pooling = SolAttnRouting.pool(keys: keys, values: values,
                                          queryCount: s, blockSize: config.blockSize)
        let pooledQ = SolAttnRouting.poolQueries(queries, blockSize: config.blockSize)
        let sink = SolAttnRouting.sinkKeyBlocks(videoSpan: videoSpan,
                                                blockSize: config.blockSize,
                                                enabled: config.exactConditioningKV)
        let selected = SolAttnRouting.select(pooledQueries: pooledQ, pooling: pooling,
                                             beta: config.beta, sinkKeyBlocks: sink)

        let groups = (s + queryRowsPerGroup - 1) / queryRowsPerGroup

        let out = kernel(
            [queries, keys, values,
             pooling.pooledKeys.asType(.float32), pooling.summedValues.asType(.float32),
             pooling.counts.asType(.float32), selected.asType(.uint8), MLXArray(scale)],
            template: [("T", queries.dtype), ("D", d), ("S", s),
                       ("NK", pooling.keyBlocks), ("NQ", pooling.queryBlocks),
                       ("BS", config.blockSize), ("BQ", queryRowsPerGroup),
                       ("TILE", keyTileRows), ("LANES", lanesPerRow),
                       ("THREADS", threadsPerGroup)],
            grid: (threadsPerGroup * groups, heads, 1),
            threadGroup: (threadsPerGroup, 1, 1),
            outputShapes: [[heads, s, d]],
            outputDTypes: [.float32])
        return out[0]
    }
}
