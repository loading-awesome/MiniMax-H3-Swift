// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

#include <metal_stdlib>
using namespace metal;

/// MLX's float32 -> bfloat16 conversion rounds to nearest, ties to even.
/// Dropping the low 16 bits changes the routed arithmetic at every value whose
/// discarded half is above the midpoint, so the native merge must make the
/// rounding explicit rather than relying on an integer shift.
inline ushort h3_bf16_rne(float value) {
    uint bits = as_type<uint>(value);
    uint bias = 0x7fffu + ((bits >> 16) & 1u);
    return ushort((bits + bias) >> 16);
}

inline float h3_bf16_value(float value) {
    return as_type<float>(uint(h3_bf16_rne(value)) << 16);
}

inline float h3_read_shard(device const ushort* gpu,
                           device const half* ane0,
                           device const half* ane1,
                           uint row, uint col,
                           uint n_gpu, uint n_ane0, uint n_ane1) {
    if (col < n_gpu) {
        return as_type<float>(uint(gpu[row * n_gpu + col]) << 16);
    }
    if (col < n_gpu + n_ane0) {
        return h3_bf16_value(float(ane0[row * n_ane0 + col - n_gpu]) * 16.0f);
    }
    return h3_bf16_value(float(ane1[row * n_ane1 + col - n_gpu - n_ane0]) * 16.0f);
}

// -----------------------------------------------------------------------------
// Phase 3: Native GPU-to-ANE Pack Kernel
// Converts bf16 [S, K] input -> multiplies by 1/16 -> converts fp16 -> transposes into [K, paddedS]
// -----------------------------------------------------------------------------
kernel void h3_pack_bf16_to_fp16_transpose(
    device const ushort* src_bf16 [[buffer(0)]],
    device half* dst_fp16         [[buffer(1)]],
    constant uint& s              [[buffer(2)]],
    constant uint& k              [[buffer(3)]],
    constant uint& padded_s       [[buffer(4)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint row = gid.y; // 0 .. S-1
    uint col = gid.x; // 0 .. K-1

    if (row >= s || col >= k) return;

    // Convert uint16_t (bf16 bit representation) to float
    ushort u = src_bf16[row * k + col];
    uint u32 = ((uint)u) << 16;
    float val = as_type<float>(u32);

    // Apply 1/16 scaling and convert to half (fp16)
    half scaled = (half)(val * 0.0625f);

    // Transpose destination layout: [K, paddedS]
    dst_fp16[col * padded_s + row] = scaled;
}

// -----------------------------------------------------------------------------
// Phase 4: Native ANE Output Merge (Attention Output)
// Merges GPU output suffix with ANE die 0 & die 1 IOSurfaces, unscaling 16x into destination bf16 buffer
// -----------------------------------------------------------------------------
kernel void h3_merge_attn_out(
    device const ushort* gpu_suffix_bf16 [[buffer(0)]],
    device const half* ane0_fp16         [[buffer(1)]],
    device const half* ane1_fp16         [[buffer(2)]],
    device ushort* dst_bf16              [[buffer(3)]],
    constant uint& s                     [[buffer(4)]],
    constant uint& n_gpu                 [[buffer(5)]],
    constant uint& n_ane0                [[buffer(6)]],
    constant uint& n_ane1                [[buffer(7)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint row = gid.y; // 0 .. S-1
    uint col = gid.x; // 0 .. N_total-1

    uint n_total = n_gpu + n_ane0 + n_ane1;
    if (row >= s || col >= n_total) return;

    float val = 0.0f;

    if (col < n_gpu) {
        // Read GPU suffix (bf16)
        ushort u = gpu_suffix_bf16[row * n_gpu + col];
        val = as_type<float>(((uint)u) << 16);
    } else if (col < n_gpu + n_ane0) {
        // Read ANE 0 (fp16) and unscale 16x
        uint c = col - n_gpu;
        val = ((float)ane0_fp16[row * n_ane0 + c]) * 16.0f;
    } else {
        // Read ANE 1 (fp16) and unscale 16x
        uint c = col - (n_gpu + n_ane0);
        val = ((float)ane1_fp16[row * n_ane1 + c]) * 16.0f;
    }

    dst_bf16[row * n_total + col] = h3_bf16_rne(val);
}

// -----------------------------------------------------------------------------
// Phase 4: Native FC1 Fused SwiGLU Merge
// Merges GPU and ANE shards, applies SiLU gating (gate * SiLU(up)), emitting [S, FFN] directly
// -----------------------------------------------------------------------------
kernel void h3_merge_fc1_swiglu(
    device const ushort* gpu_suffix_bf16 [[buffer(0)]],
    device const half* ane0_fp16         [[buffer(1)]],
    device const half* ane1_fp16         [[buffer(2)]],
    device ushort* dst_bf16              [[buffer(3)]],
    constant uint& s                     [[buffer(4)]],
    constant uint& n_gpu                 [[buffer(5)]],
    constant uint& n_ane0                [[buffer(6)]],
    constant uint& n_ane1                [[buffer(7)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint row = gid.y; // 0 .. S-1
    uint col = gid.x; // 0 .. FFN_dim-1 (where FFN_dim = N_total / 2)

    uint ffn_dim = (n_gpu + n_ane0 + n_ane1) / 2;
    if (row >= s || col >= ffn_dim) return;

    // Gate column index = col, Up column index = col + ffn_dim
    uint gate_col = col;
    uint up_col = col + ffn_dim;

    float gate_val = h3_read_shard(gpu_suffix_bf16, ane0_fp16, ane1_fp16,
                                   row, gate_col, n_gpu, n_ane0, n_ane1);
    float up_val = h3_read_shard(gpu_suffix_bf16, ane0_fp16, ane1_fp16,
                                 row, up_col, n_gpu, n_ane0, n_ane1);

    // H3 uses SiLU(gate) * up. Reversing the halves preserves shapes and often
    // looks plausible, which is why the model contract pins the order.
    float sigmoid_tail = 1.0f / (1.0f + exp(abs(gate_val)));
    float sigmoid_gate = h3_bf16_value(
        gate_val < 0.0f ? sigmoid_tail : 1.0f - sigmoid_tail);
    float silu_gate = h3_bf16_value(gate_val * sigmoid_gate);
    float out_val = silu_gate * up_val;

    dst_bf16[row * ffn_dim + col] = h3_bf16_rne(out_val);
}
