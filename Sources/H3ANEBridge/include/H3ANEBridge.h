// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

#ifndef H3ANEBRIDGE_H
#define H3ANEBRIDGE_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#import <stdbool.h>
#import <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A minimal seam onto the Neural Engine, split into three pieces that have
/// genuinely different lifetimes. Conflating them is what made the first
/// version copy 66 GB of unchanging weights per render.
///
///  * `H3ANEProgram` — a compiled MIL matmul, fixed at one `(s, k, n)`.
///    Compilation is expensive and the shape is constant for a whole render,
///    so one program serves every step.
///  * `H3ANETensor`  — an IOSurface with a 64-byte-aligned row stride. A
///    weight tensor is written once per block and read a thousand times; an
///    activation tensor is rewritten every call. Separating them lets the
///    caller keep the first and reuse the second.
///  * the run calls    — bind three tensors to a program and evaluate.
///
/// Everything is fp16 and row-major. `h3_ane_tensor_write` is the only way in,
/// and it refuses a shape that does not match what the tensor was allocated
/// for, because the alternative — the previous version's unchecked `memcpy` of
/// a caller-sized buffer into a fixed surface — overran by 147 MB at
/// production sequence length.

/// True only when the host matches the validated machine/OS ABI and the
/// AppleNeuralEngine framework exposes every private selector this bridge
/// drives. Researchers may set `H3_ANE_ALLOW_UNVALIDATED=1` to probe another
/// host deliberately; production callers should treat false as normal fallback.
bool h3_ane_is_available(void);

#pragma mark - Tensors

/// An fp16 IOSurface of `rows` x `width`, rows padded to a 64-byte stride.
typedef struct H3ANETensor H3ANETensor;

H3ANETensor* _Nullable h3_ane_tensor_create(int rows, int width);
void h3_ane_tensor_free(H3ANETensor* _Nullable t);

/// Base address of the surface. Rows are `h3_ane_tensor_row_bytes` apart,
/// which is not `width * 2` unless `h3_ane_tensor_is_dense` is true.
void* _Nullable h3_ane_tensor_ptr(H3ANETensor* _Nonnull t);
size_t h3_ane_tensor_row_bytes(H3ANETensor* _Nonnull t);

/// True when the row stride equals `width * sizeof(fp16)`, so the surface can
/// be read as a flat `[rows, width]` buffer with no gaps. Holds for every
/// production shape here (K=5376 and N=3072 are both 64-byte multiples in
/// fp16), but not in general.
bool h3_ane_tensor_is_dense(H3ANETensor* _Nonnull t);

/// Copies a row-major `[rows, width]` fp16 buffer in, honouring the stride.
/// Returns false — without writing anything — if the shape does not match
/// what this tensor was allocated for.
bool h3_ane_tensor_write(H3ANETensor* _Nonnull t, const void* _Nonnull src,
                         int rows, int width);

/// Copies a row-major `[rows, width]` fp16 buffer into the leading `width`
/// columns of each row, leaving the remainder as allocated (zero).
///
/// This exists for sequence padding. The engine's throughput depends on the
/// activation's minor extent: at the production shard it sustains 3.87 TFLOP/s
/// a die at s=14336 and s=16384 but only 2.45 at s=15731, which is prime and
/// pads badly. Compiling for the next multiple of 64 and writing the real
/// sequence into the front recovers the full rate; the surplus columns compute
/// garbage from zeros and are sliced off the output.
///
/// Returns false without writing if `rows` differs or `width` exceeds the
/// allocation.
bool h3_ane_tensor_write_prefix(H3ANETensor* _Nonnull t, const void* _Nonnull src,
                                int rows, int width);

#pragma mark - Programs

/// A compiled `y[s,n] = x[k,s]^T @ w[k,n]` on the engine.
///
/// The weight orientation is the engine's, not the checkpoint's. Three layouts
/// were measured at the production shard: contracting over the last axis of
/// both operands (`transpose_y`) costs 2.42 TFLOP/s a die, while contracting
/// the activation's last axis against the weight's first runs at 3.79 — the
/// same rate as the fully-transposed form verified in
/// `docs/ANE_REVERSE_ENGINEERING.md`, but without obliging the caller to
/// transpose a 169 MB activation every call. Only the weight is transposed,
/// once, when it is uploaded.
typedef struct H3ANEProgram H3ANEProgram;

/// Compiles and loads. Returns NULL if the engine refuses the shape, which is
/// a normal outcome to fall back from rather than an error to report.
H3ANEProgram* _Nullable h3_ane_program_create(int s, int k, int n);

/// How the linear is expressed to the engine's compiler.
///
/// The two forms compute the same thing and differ only in what the ANE
/// compiler is handed, which is worth 40% of the rate. `H3ANEFormMatmul`
/// declares `a` as `[1,k,1,s]` and transposes it to `[1,1,s,k]` **inside the
/// graph**, so the engine moves a 169 MB activation before it multiplies
/// anything. `H3ANEFormConv` is a 1x1 convolution over `[1,k,1,s]` — channels
/// in, channels out, sequence as the spatial axis — which is the engine's
/// native shape and needs no transpose.
///
/// The two forms want the weight in different orientations, and the conv form
/// wants the one the checkpoint already holds:
///
/// | form | x | w | y |
/// |---|---|---|---|
/// | matmul | `[k,s]` | `[k,n]` | `[s,n]` |
/// | conv   | `[k,s]` | `[n,k]` | `[n,s]` |
///
/// The conv form's output arrives channel-major, so the caller transposes it
/// on the GPU — which is where transposes are cheap — instead of asking the
/// engine to do it.
typedef enum {
    H3ANEFormMatmul = 0,
    H3ANEFormConv   = 1,
} H3ANEForm;

H3ANEProgram* _Nullable h3_ane_program_create_form(int s, int k, int n, H3ANEForm form);

/// Which form this program was compiled in; it decides the tensor shapes the
/// run calls demand.
H3ANEForm h3_ane_program_form(H3ANEProgram* _Nonnull p);

void h3_ane_program_free(H3ANEProgram* _Nullable p);

/// Tensor shapes this program requires: x is `[k, s]`, w is `[k, n]`,
/// y is `[s, n]`. The first two carry `k` as the leading axis because that is
/// what the engine runs fastest on; only `y` comes back in the caller's own
/// orientation, which is what lets it be aliased rather than copied.
int h3_ane_program_s(H3ANEProgram* _Nonnull p);
int h3_ane_program_k(H3ANEProgram* _Nonnull p);
int h3_ane_program_n(H3ANEProgram* _Nonnull p);

/// Evaluates synchronously on the calling thread.
///
/// `instance_hint` is passed as `kANEFAneInstanceHint` and is advisory at
/// best: `docs/ANE_REVERSE_ENGINEERING.md` measured a job hinted to instance 2
/// burning energy on die 0. It does not select a die. Use `h3_ane_run_pair`
/// to get two dies busy.
bool h3_ane_run(H3ANEProgram* _Nonnull p,
                H3ANETensor* _Nonnull x, H3ANETensor* _Nonnull w,
                H3ANETensor* _Nonnull y, int instance_hint);

/// Evaluates two independent shards on two threads and waits for both.
///
/// This is the only thing that engages the second die. `h3_ane_run` blocks in
/// `evaluateWithQoS:`, so calling it twice in a row costs exactly twice as
/// much — measured 17.8 ms once against 36.0 ms back-to-back. Concurrency is
/// what the kernel's load balancer reacts to, not the instance hint.
bool h3_ane_run_pair(H3ANEProgram* _Nonnull p0, H3ANETensor* _Nonnull x0,
                     H3ANETensor* _Nonnull w0, H3ANETensor* _Nonnull y0,
                     H3ANEProgram* _Nonnull p1, H3ANETensor* _Nonnull x1,
                     H3ANETensor* _Nonnull w1, H3ANETensor* _Nonnull y1);

#ifdef __cplusplus
}
#endif

#endif /* H3ANEBRIDGE_H */
