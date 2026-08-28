// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

#ifndef H3ANEBRIDGE_H
#define H3ANEBRIDGE_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#import <stdbool.h>
#import <stddef.h>
#import <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A minimal seam onto the Neural Engine, split into three pieces that have
/// genuinely different lifetimes:
///  * `H3ANEProgram` — a compiled MIL matmul, fixed at one `(s, k, n)`.
///  * `H3ANETensor`  — an IOSurface with a 64-byte-aligned row stride.
///  * the run / job API calls — bind tensors and evaluate.

bool h3_ane_is_available(void);

#pragma mark - Tensors

typedef struct H3ANETensor H3ANETensor;

H3ANETensor* _Nullable h3_ane_tensor_create(int rows, int width);
void h3_ane_tensor_free(H3ANETensor* _Nullable t);

void* _Nullable h3_ane_tensor_ptr(H3ANETensor* _Nonnull t);
size_t h3_ane_tensor_row_bytes(H3ANETensor* _Nonnull t);
bool h3_ane_tensor_is_dense(H3ANETensor* _Nonnull t);

bool h3_ane_tensor_write(H3ANETensor* _Nonnull t, const void* _Nonnull src,
                         int rows, int width);
bool h3_ane_tensor_write_prefix(H3ANETensor* _Nonnull t, const void* _Nonnull src,
                                int rows, int width);

#pragma mark - Programs

typedef struct H3ANEProgram H3ANEProgram;

H3ANEProgram* _Nullable h3_ane_program_create(int s, int k, int n);

typedef enum {
    H3ANEFormMatmul = 0,
    H3ANEFormConv   = 1,
} H3ANEForm;

H3ANEProgram* _Nullable h3_ane_program_create_form(int s, int k, int n, H3ANEForm form);
H3ANEForm h3_ane_program_form(H3ANEProgram* _Nonnull p);
void h3_ane_program_free(H3ANEProgram* _Nullable p);

int h3_ane_program_s(H3ANEProgram* _Nonnull p);
int h3_ane_program_k(H3ANEProgram* _Nonnull p);
int h3_ane_program_n(H3ANEProgram* _Nonnull p);

bool h3_ane_run(H3ANEProgram* _Nonnull p,
                H3ANETensor* _Nonnull x, H3ANETensor* _Nonnull w,
                H3ANETensor* _Nonnull y, int instance_hint);

bool h3_ane_run_pair(H3ANEProgram* _Nonnull p0, H3ANETensor* _Nonnull x0,
                     H3ANETensor* _Nonnull w0, H3ANETensor* _Nonnull y0,
                     H3ANEProgram* _Nonnull p1, H3ANETensor* _Nonnull x1,
                     H3ANETensor* _Nonnull w1, H3ANETensor* _Nonnull y1);

#pragma mark - Fused Attention Programs

/// A fixed-shape, explicit-stable-softmax attention program. Q is expected to
/// be pre-scaled by `1/sqrt(head_dim)`; Q, K, V and Y are packed fp16
/// `[heads, sequence, head_dim]` tensors.
typedef struct H3ANEAttentionProgram H3ANEAttentionProgram;

H3ANEAttentionProgram* _Nullable h3_ane_attention_program_create(
    int heads, int sequence, int head_dim);
void h3_ane_attention_program_free(H3ANEAttentionProgram* _Nullable p);

bool h3_ane_attention_run(
    H3ANEAttentionProgram* _Nonnull p,
    H3ANETensor* _Nonnull q, H3ANETensor* _Nonnull k,
    H3ANETensor* _Nonnull v, H3ANETensor* _Nonnull y,
    int instance_hint);

/// Runs one head group on each physical ANE concurrently. The runtime assigns
/// the two simultaneous evaluations to distinct dies; the instance hints alone
/// do not select a die.
bool h3_ane_attention_run_pair(
    H3ANEAttentionProgram* _Nonnull p0,
    H3ANETensor* _Nonnull q0, H3ANETensor* _Nonnull k0,
    H3ANETensor* _Nonnull v0, H3ANETensor* _Nonnull y0,
    H3ANEAttentionProgram* _Nonnull p1,
    H3ANETensor* _Nonnull q1, H3ANETensor* _Nonnull k1,
    H3ANETensor* _Nonnull v1, H3ANETensor* _Nonnull y1);

#pragma mark - Asynchronous Bounded Job API (Phase 2)

typedef struct H3ANEJob H3ANEJob;

typedef enum {
    H3ANEJobStateFree = 0,
    H3ANEJobStatePacking,
    H3ANEJobStateReady,
    H3ANEJobStateRunning,
    H3ANEJobStateComplete,
    H3ANEJobStateError
} H3ANEJobState;

/// Submits an asynchronous dual-die ANE job pair. Returns non-NULL job handle.
H3ANEJob* _Nullable h3_ane_job_submit_pair(H3ANEProgram* _Nonnull p0, H3ANETensor* _Nonnull x0,
                                           H3ANETensor* _Nonnull w0, H3ANETensor* _Nonnull y0,
                                           H3ANEProgram* _Nonnull p1, H3ANETensor* _Nonnull x1,
                                           H3ANETensor* _Nonnull w1, H3ANETensor* _Nonnull y1);

/// Waits for job completion with a explicit nanosecond timeout.
bool h3_ane_job_wait(H3ANEJob* _Nonnull job, uint64_t timeout_ns);

/// Returns the current lifecycle state of a job.
H3ANEJobState h3_ane_job_status(H3ANEJob* _Nonnull job);

/// Retires a completed job handle and frees its state.
void h3_ane_job_retire(H3ANEJob* _Nullable job);

#pragma mark - Native GPU Pack & Fused Merge Kernels (Phases 3 & 4)

/// Native GPU-to-ANE Pack Kernel:
/// Converts bf16 [S, K] input -> scales 1/16 -> converts fp16 -> transposes into fp16 IOSurface [K, paddedS].
bool h3_ane_pack_bf16_to_fp16_transpose(const void* _Nonnull srcBF16,
                                         H3ANETensor* _Nonnull dstTensor,
                                         int s, int k,
                                         void* _Nullable commandQueue);

/// Native ANE Output Merge:
/// Merges GPU output suffix with ANE die 0 & die 1 IOSurfaces, unscaling 16x into destination bf16 buffer.
bool h3_ane_merge_attn_out(const void* _Nonnull gpuSuffixBF16,
                           H3ANETensor* _Nonnull ane0Tensor,
                           H3ANETensor* _Nonnull ane1Tensor,
                           void* _Nonnull dstBF16,
                           int s, int nGpu, int nAne0, int nAne1,
                           void* _Nullable commandQueue);

/// Native FC1 Fused SwiGLU Merge:
/// Merges GPU and ANE shards, applies SiLU gating (gate * SiLU(up)), emitting [S, FFN] directly.
bool h3_ane_merge_fc1_swiglu(const void* _Nonnull gpuSuffixBF16,
                             H3ANETensor* _Nonnull ane0Tensor,
                             H3ANETensor* _Nonnull ane1Tensor,
                             void* _Nonnull dstBF16,
                             int s, int nGpu, int nAne0, int nAne1,
                             void* _Nullable commandQueue);

/// Persistent-MLP seam: reads separate fp16 `[S,F]` gate/up ANE outputs,
/// applies `SiLU(gate) * up`, and writes fp16 `[F,S]` directly into the input
/// surface for a row-sharded fc2 program. `inputUnscale` restores any scaling
/// applied before fc1; `outputScale` prepares the activation for fc2.
bool h3_ane_swiglu_transpose_fp16(H3ANETensor* _Nonnull gateTensor,
                                  H3ANETensor* _Nonnull upTensor,
                                  H3ANETensor* _Nonnull dstTensor,
                                  int s, int ffn,
                                  float inputUnscale, float outputScale,
                                  void* _Nullable commandQueue);

/// Encodes both ANE islands' SwiGLU seams into one Metal command buffer. The
/// two ranges may use different calibrated fc2 scales, but have the same shape.
bool h3_ane_swiglu_transpose_pair_fp16(
    H3ANETensor* _Nonnull gate0, H3ANETensor* _Nonnull up0,
    H3ANETensor* _Nonnull dst0, float outputScale0,
    H3ANETensor* _Nonnull gate1, H3ANETensor* _Nonnull up1,
    H3ANETensor* _Nonnull dst1, float outputScale1,
    int s, int ffn, float inputUnscale,
    void* _Nullable commandQueue);

/// Split-contraction form of the persistent seam. Each gate/up array contains
/// four fp16 partials over disjoint contraction ranges. They are accumulated
/// in fp32 before SwiGLU, improving both ANE throughput and long-dot accuracy.
bool h3_ane_swiglu_transpose_pair_split4_fp16(
    H3ANETensor* _Nonnull const* _Nonnull gate0,
    H3ANETensor* _Nonnull const* _Nonnull up0,
    H3ANETensor* _Nonnull dst0, float outputScale0,
    H3ANETensor* _Nonnull const* _Nonnull gate1,
    H3ANETensor* _Nonnull const* _Nonnull up1,
    H3ANETensor* _Nonnull dst1, float outputScale1,
    int s, int ffn, float inputUnscale,
    void* _Nullable commandQueue);

/// Final persistent-island join. Adds the GPU's bf16 partial and four fp16
/// fc2 piece outputs from each die, unscaling each die only once, into bf16.
bool h3_ane_merge_mlp_island_partials(
    const void* _Nonnull gpuPartialBF16,
    H3ANETensor* _Nonnull const* _Nonnull ane0Partials, float ane0Unscale,
    H3ANETensor* _Nonnull const* _Nonnull ane1Partials, float ane1Unscale,
    void* _Nonnull dstBF16, int s, int hidden,
    void* _Nullable commandQueue);

#ifdef __cplusplus
}
#endif

#endif /* H3ANEBRIDGE_H */
