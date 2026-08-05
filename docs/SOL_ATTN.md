# Sol-Attn on Metal — what it is, and what porting it costs

Read from `kijai/ComfyUI-SolAttn_triton` and arXiv 2607.24027 (Ye, Han et al.,
*"Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention
Sparsification"*), 2026-08-05.

The reference implementation is **already MiniMax-H3-shaped**. It is the only
one kijai tested against, and several of its features exist specifically because
of this model's packed sequence. That makes it unusually good news and one
piece of bad news.

---

## What it does

Block-sparse self-attention, **training-free and approximate**. Attention is
computed in key-value blocks; during the online softmax, each block's proxy
score is compared against a threshold, and only blocks above it are computed
exactly. The rest are approximated by reusing their proxy scores directly.

The full proxy score map is never materialised — that is the efficiency claim
over top-k methods, which have to build the whole distribution before they can
rank it.

`tau` is the threshold, and it is a *density dial* rather than a budget:

| tau | blocks kept exact |
|---|---|
| 1.0 | ~16% |
| 1.5 | ~7% |
| 2.0 | ~2.7% |

Reported: **2.1× end-to-end on video generation**, 2.3× on editing, quality
preserved. The user's framing — variable-rate against constant-rate encoding —
is exactly right: dense attention spends the same compute on every pair, and
this spends it where the scores say it matters.

## Four things it needs that a naive `attend(q,k,v)` cannot express

Every one of these is a *policy about where the render is*, not about the
tensors, and each is why the first version of this repo's attention seam was the
wrong shape.

**1. Position in the sigma schedule.** Dense before `start_percent` (0.2 in the
paper) and after `end_percent` (0.9). Early steps set global structure and late
steps set fine detail; approximating either shows.

**2. Which transformer block.** `dense_blocks` exists because "the first and
last blocks are the most approximation-sensitive: their error reaches the output
with no later block to absorb it."

**3. Where the conditioning rows are.** H3 packs
`[text][cond][ref_img][ref_audio][audio][video]` into one self-attention
sequence, and sparsifying the conditioning costs sync and prompt adherence. Two
levels, both H3-specific:

| mode | what stays exact | cost |
|---|---|---|
| `exact_kv` | every query sees the conditioning rows exactly | ~3% |
| `exact_kv_and_rows` | those rows also run as dense *queries*, making the generated audio stream exact | ~17-20% |

This is a direct consequence of the layout documented in
`FRAGILE_CONTRACTS.md` #10, and it is the reason `AttentionContext` carries the
video span rather than just a length.

**4. Sequence length.** Below `min_tokens` (4096) the routing overhead exceeds
the saving.

## Morton reordering is a pipeline transform, not an attention one

Video tokens are permuted into Z-order so that a 64-token block is a compact 3D
neighbourhood instead of a two-row strip of a single frame. That makes a
block-level routing decision a decision about a *region* rather than a
horizontal sliver.

It permutes hidden states and position ids around the whole block stack and
restores them before the final layer, so it is **exactly neutral for dense
attention** — tokens and their positions move together. That is why this repo
models it as `TokenOrdering` at the pipeline level rather than inside the
backend, which only declares whether it wants it.

**H3's temporal axis is not uniformly spaced.** `videoTGrid` places latent
frames on a stretched grid, so interleaving t/h/w equally can be worse than
Z-ordering within each frame and leaving frame order alone. kijai's node
defaults to `2d_frame` for exactly this reason and warns that `3d` "may degrade
at some frame counts". The per-frame curve is the safer default here too.

The H3 span registration also has to survive a detail worth noting: the video
segment must be **block-aligned**, because the kernels block from absolute
position 0. A span that starts off a 64-boundary splits every Z-order cell
across two blocks, joining opposite ends of the volume. The reference rotates
the permutation by the misalignment to fix it.

## The bad news: Triton does not target Metal

There is no binding to write. `_tri_fwd.py` and `_int8_fwd.py` are Triton
kernels compiled for CUDA, tested on 4090 and 5090 with SM90+ paths for TMA
descriptors. A Metal Sol-Attn is a **reimplementation**, against
`MLXFast.metalKernel`, which mlx-swift does expose — JIT-compiled Metal from a
source string, structurally the same move Triton makes.

What ports for free is the design: the thresholding criterion, the sink
structure, the Morton curve, the dense-block policy, the schedule window. What
does not is the kernel, and the kernel is where the performance lives.

## Why it is worth the work anyway

The roofline says there is nowhere else to go. Measured on this port:

* the model already runs at **~90% of achievable matmul throughput**, and bf16,
  fp16 and fp32 GEMM land within 6% of each other on pre-M5 silicon, so there is
  no precision trick left;
* all memory traffic accounts for ~7 seconds of a 1222-second forward;
* attention is quadratic in a 15,000-20,000-token sequence.

Dense attention is the cost, and sparsifying it is the one available lever that
is not "use a smaller model". A 2× on a 23-minute render is 11 minutes.

## Open questions before implementation

1. **Does the equivalence class survive H3's audio stream?** The `exact_kv_and_rows`
   mode exists because the generated audio degrades otherwise. Our oracles can
   measure that directly — `speech_check.py` gives WER on the rendered
   waveform — so this is testable rather than a matter of opinion.
2. **What is the class, numerically?** The backend protocol requires one. It has
   to be measured against dense output at production shape, per tau, and it will
   not be a single number: sparsity varies with tau, and the sensible product
   surface is probably two or three named presets with a measured class each,
   rather than a raw float.
3. **int8 QK.** The reference reports it free in quality, a help at tau ≤ 1.5 and
   a net loss at tau ≥ 2.0 where the quantise pass outweighs the shrinking exact
   branch. Worth having, worth deferring until the bf16 path is correct.
