// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import MLX
import MLXFast

/// Test-only implementation used to measure the fused-kernel ceiling. It is
/// deliberately absent from H3Modules: the measured render gain failed the 5%
/// gate, so production must not acquire a dormant alternative math path.
enum FusedQKRoPE {
    private static let threads = 128

    private static let source = """
        const uint row = threadgroup_position_in_grid.x;
        const uint tid = thread_index_in_threadgroup;
        const uint token = row / (uint)HEADS;
        const uint head = row - token * (uint)HEADS;
        const uint qBase = token * (uint)(3 * INNER) + head * (uint)D;
        const uint kBase = qBase + (uint)INNER;
        const uint outBase = row * (uint)D;

        threadgroup float qPartial[THREADS];
        threadgroup float kPartial[THREADS];
        threadgroup T qNormalized[D];
        threadgroup T kNormalized[D];

        const float qv = (float)qkv[qBase + tid];
        const float kv = (float)qkv[kBase + tid];
        qPartial[tid] = qv * qv;
        kPartial[tid] = kv * kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = (uint)THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                qPartial[tid] += qPartial[tid + stride];
                kPartial[tid] += kPartial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float qInv = metal::rsqrt(qPartial[0] / (float)D + eps[0]);
        const float kInv = metal::rsqrt(kPartial[0] / (float)D + eps[0]);
        qNormalized[tid] = (T)(qv * qInv * (float)qWeight[tid]);
        kNormalized[tid] = (T)(kv * kInv * (float)kWeight[tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < (uint)HALF) {
            const uint r = (token * (uint)HALF + tid) * 4;
            const T c = rope[r];
            const T negS = rope[r + 1];
            const T s = rope[r + 2];
            const T c2 = rope[r + 3];
            const T qa = qNormalized[tid];
            const T qb = qNormalized[tid + HALF];
            const T ka = kNormalized[tid];
            const T kb = kNormalized[tid + HALF];

            const T qac = (T)((float)c * (float)qa);
            const T qbs = (T)((float)negS * (float)qb);
            const T qas = (T)((float)s * (float)qa);
            const T qbc = (T)((float)c2 * (float)qb);
            qOut[outBase + tid] = (T)((float)qac + (float)qbs);
            qOut[outBase + tid + HALF] = (T)((float)qas + (float)qbc);

            const T kac = (T)((float)c * (float)ka);
            const T kbs = (T)((float)negS * (float)kb);
            const T kas = (T)((float)s * (float)ka);
            const T kbc = (T)((float)c2 * (float)kb);
            kOut[outBase + tid] = (T)((float)kac + (float)kbs);
            kOut[outBase + tid + HALF] = (T)((float)kas + (float)kbc);
        } else if (tid >= (uint)ROT) {
            qOut[outBase + tid] = qNormalized[tid];
            kOut[outBase + tid] = kNormalized[tid];
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "h3_fused_qk_rmsnorm_rope_fixture",
        inputNames: ["qkv", "qWeight", "kWeight", "rope", "eps"],
        outputNames: ["qOut", "kOut"],
        source: source)

    static func apply(qkv: MLXArray, qWeight: MLXArray, kWeight: MLXArray,
                      rope: MLXArray, heads: Int, headDim: Int,
                      eps: Float) -> (q: MLXArray, k: MLXArray)? {
        guard qkv.ndim == 2, qWeight.ndim == 1, kWeight.ndim == 1,
              headDim == threads, qWeight.dim(0) == headDim,
              kWeight.dim(0) == headDim, qkv.dim(1) == 3 * heads * headDim,
              rope.ndim == 6, rope.dim(0) == 1, rope.dim(1) == qkv.dim(0),
              rope.dim(2) == 1, rope.dim(-2) == 2, rope.dim(-1) == 2,
              qkv.dtype == qWeight.dtype, qkv.dtype == kWeight.dtype,
              qkv.dtype == rope.dtype else { return nil }
        let half = rope.dim(-3)
        let rot = half * 2
        guard rot <= headDim else { return nil }

        let s = qkv.dim(0)
        let inner = heads * headDim
        let out = kernel(
            [qkv, qWeight, kWeight, rope, MLXArray([eps])],
            template: [("T", qkv.dtype), ("D", headDim), ("HEADS", heads),
                       ("INNER", inner), ("HALF", half), ("ROT", rot),
                       ("THREADS", threads)],
            grid: (threads * s * heads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[s, heads, headDim], [s, heads, headDim]],
            outputDTypes: [qkv.dtype, qkv.dtype])
        return (out[0], out[1])
    }
}
