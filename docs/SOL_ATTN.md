# Sol-Attn on Metal — what it is, and what it is worth

Sources, in order of authority:

* **The paper.** arXiv:2607.24027v1, 27 Jul 2026, NVIDIA. Li, Li, Chen, Ye, Liu,
  Yu, Wang, Zhang, Xie, Xie, Han. *"Sol-Attn: Accelerating Video Generation
  Inference via On-the-Fly Attention Sparsification."* 16 pages.
* **NVIDIA's official H3 integration.** `NVlabs/Sana`, branch `sol-engine`,
  `models/minimax_h3/optimized/sol_attn_h3.py`, plus the write-up at
  <https://nvlabs.github.io/Sana/Sol-Engine/H3/>. **This is post-paper: H3 is not
  evaluated in the paper at all.**
* **kijai's ComfyUI port**, `kijai/ComfyUI-SolAttn_triton`, which adds two things
  the paper and the official backend do not have.

A first pass at this document attributed several implementer additions to the
paper. Those are corrected below and marked, because the distinction decides
what a Metal port has to reproduce and what it merely may.

---

## 1. The criterion

**Naming, first, because it is a trap.** In the paper **τ is the threshold** and
**β is the knob**. kijai's node exposes a parameter *called* `tau` whose tooltip
says "Threshold beta" — it is the paper's **β**. This document uses β.

Per query block *i*, over the *N* key blocks:

```
ŝ_ij = Q̄_i K̄_jᵀ                    proxy score, pooled query vs pooled key
τ_i  = μ_i + β σ_i                   μ, σ are the mean and sd of the row ŝ_i
S_i  = { j : ŝ_ij > τ_i }            the exact set
```

μ and σ are statistics **across key blocks for one query block** — not a running
maximum, not the online-softmax normaliser. τ is computed *before* the kernel
runs, which is what makes routing nearly free.

The paper's justification (§3.1): aggregated over steps, layers, heads and query
blocks, standardized proxy logits are "consistently near-Gaussian within each
model", so β maps to a density **ρ(β) = 1 − Φ(β)**. That is exactly the observed
behaviour of kijai's node:

| β | tooltip density | 1 − Φ(β) |
|---|---|---|
| 1.0 | ~16% | 15.87% |
| 1.5 | ~7% | 6.68% |
| 2.0 | ~2.7% | 2.28% |

**τ without the N×N map** (§3.1 Eq. 5, Appendix B): from first and second moments
of the pooled keys, `O(Ld + Nd²)` time and `O(d²)` storage. The released kernel
uses the cheaper **diagonal estimator** (Eq. 15), `σ²_diag = (Q̄ ⊙ Q̄) v_Kᵀ`, at
`O(d)` per query block. This is why routing costs **0.33 ms against top-k's
3.80 ms and top-p's 10.8 ms** (Fig. 5b) — 11.5× and 32.7×.

kijai's implementation adds an optional Cornish-Fisher correction using third and
fourth moments, clamped to ±1σ, for non-Gaussianity. Not in the paper.

## 2. The approximation is a correction term, not a mask

This is the part that decides quality, and it is the part a naive "block-sparse
attention" implementation would omit.

Taylor expansion about the pooled key, keeping the zeroth-order term (§3.2
Eq. 8-10). With `V̂_j` the *sum* of `V_j` along tokens, `U_i` the rejected set:

```
D_i = Σ_{j∈U_i} B·exp(Q_i K̄_jᵀ)        + Σ_{j∈S_i} RowSum(exp(Q_i K_jᵀ))
N_i = Σ_{j∈U_i} exp(Q_i K̄_jᵀ) V̂_j      + Σ_{j∈S_i} exp(Q_i K_jᵀ) V_j
        └──── approximate ────┘              └──── exact ────┘
```

A rejected block contributes a **rank-1 term** — a per-row scalar weight times
the summed value vector — to **both numerator and denominator**, through the same
online-softmax state as the exact blocks.

Two details an implementer will get wrong once:

* The approximate weight uses the **full query tokens** `Q_i`, not the pooled
  query `Q̄_i`. Routing recovers `ŝ_i` as the *column mean* of that same tile
  (Eq. 11), so **one B×C GEMM serves both routing and correction**. That identity
  is the whole trick.
* Selected columns must be masked out of the approximate branch or their
  contribution is counted twice.

Error bound (Appendix B Eq. 16): denominator error is second order in the
centred score spread; numerator error also depends on alignment with the values.
Ablation (Fig. 9) shows the correction reduces relative ℓ2 error and raises
cosine similarity across 70–90% sparsity, **with a widening advantage as sparsity
rises**.

**Honest kernel accounting** (Fig. 10): Sol-Attn's kernel is *slower* than plain
block-sparse attention at identical block indices — by 9.4% / 3.9% / 1.6% at
90/85/80% sparsity — because it does the correction. It wins end to end only
because BSA needs a separate routing stage. So a Metal port that skips the
correction would be faster per kernel and worse per pixel.

## 3. Blocks and the loop

64×64 physical blocks, symmetric. A second, independent chunk size *C* over the
pooled-key sequence "affects efficiency but not the mathematical output".

Nested: the outer loop scans pooled keys chunk by chunk (dense), the inner loop
visits selected blocks within the chunk (sparse). One online-softmax state in
registers, updated by **both** branches. The running max plays no part in
selection.

## 4. What is measured, and what is not

Reported end-to-end, attention swap only, at ~85% sparsity for video:

| model | sparsity | speedup | vs dense |
|---|---|---|---|
| Wan2.1-14B 720p 81f | 85.1% | **2.02×** | VBench 76.13 vs 75.90 |
| HunyuanVideo-13B 720p 129f | 85.8% | **2.12×** | 76.81 vs 77.06 |
| LTX 2.3-22B 1080p | 89.8% | 1.9× | 74.69 vs 74.59 |
| Bernini-14B editing | 85.0% | 2.34× | ≥ dense on 3 of 4 axes |
| Ideogram 4 2K T2I | 90.2% | 1.56× | **55.31 vs 59.24** |

Kernel-level against FA3 on H100: **5.41× at 128K tokens and 90% sparsity**.
Memory: 1.45 GB against SVG2's 11.52 GB.

**Not measured anywhere, and these matter to us:**

* **Quality as a function of β.** Every comparison is at *matched sparsity*.
  There is no β-to-quality table. Ours would have to be measured.
* **Sparsity by timestep, layer or head.** Fig. 3 explicitly aggregates those
  axes away.
* **Any audio-video model.** No audio metric appears in the paper.

## 5. Corrections to a first reading

**There is no late-schedule dense phase in the paper.** It specifies a dense
*warm-up* only — first 20% of steps and the first layer, attributed to "prior
work" and never ablated. kijai's `end_percent 0.9` is **his addition**. Any claim
that "the paper runs dense after 90%" is wrong; ours said so and has been fixed.

**H3 does not need Morton reordering.** NVIDIA's official H3 integration states
the packed video tail "is already a contiguous grid-ordered block, and the
routing works on it directly." Morton is absent from the paper (zero occurrences
of Morton, Z-order or Hilbert) and absent from the official backend's 917-file
tree. It is kijai's addition, and his own H3 note explains why the 3D curve is
wrong here: `FRAME_PER_TOKEN` is `(1, 4, 4, 4, 4)`, so index-adjacent frames are
1 or 4 real frames apart and a 3D curve groups temporally distant tokens.
**Conclusion: `TokenOrdering` stays in our design as a seam, but is not required
for H3 and should default to `.none`.**

**int8 QK is not in the paper either.** Sage-style per-token absmax scales with
mean-smoothed K, exact branch only — the pooled tensors are O(N·d) so quantising
them saves nothing. Implementer's addition; quantisation lives in Sol-Engine as
a separate composable technique upstream.

## 6. The audio finding, which is the most important thing here

NVIDIA's H3 integration defaults `sink_mode` to **`prefix`**, not `text`, and the
reason is recorded in the release notes: the audio rows "carry the soundtrack and
are themselves *generated* (the model returns an audio velocity for them)." With
a text-only sink they hit a case where **"the picture scored best of its set
while its dialogue fell apart."**

That is the exact failure this port's oracles were built to catch, found
independently by the people who wrote the method. Two consequences:

* The conditioning sink is **not optional** for H3. It is the difference between
  a good-looking render and one whose speech is gone.
* **The cost is disputed and worth measuring.** NVIDIA quotes roughly **1% extra
  density and 1% extra attention** for the prefix sink. kijai's node quotes
  **~3%** for `exact_kv` and **~17–20%** for `exact_kv_and_rows`, and defaults to
  the expensive one. Those are different policies, not just different numbers,
  and `speech_check.py` can settle which is needed.

Also from the release contract: *"The sink does not change query routing: an
MMDiT integration should still compute valid text query rows with dense attention
and use Sol-Attn for image/video query rows."*

## 7. Where this sits in the priority order

NVIDIA's own H3 page breaks the 3.95× on 8×GB200 (1344×768, 124 frames) down:

| technique | contribution |
|---|---|
| Cross-step cache (First Block Cache) | **1.534× → 3.95×, the largest single gain** |
| Kernel fusion, chiefly fused RMS-AdaLN | "the primary win" for H3's 13B AdaLN parameters |
| **Sol-Attn** | **1.25× additional** |

An earlier draft of this document called Sol-Attn "the one available lever that
is not a smaller model." That was wrong. On H3 specifically it is the *smallest*
of the three and the only one that needs a Metal kernel.

Amdahl agrees. At our verified shape attention is 37% of DiT FLOPs (S = 15,750),
rising to 54% at S = 31,500 — so the ceiling from sparsification alone is ~1.5×
at 864×480×124 and ~1.9× at twice the length. **Sol-Attn is worth most for the
long, high-resolution renders that are currently unaffordable**, which is a real
argument for it, but a different one from "2× everything".

Recommended order: **cross-step cache first** (pure Swift, no kernel, biggest
win), **fused AdaLN second** (bounded kernel work, and this port already has the
AdaLN residency machinery), **Sol-Attn third**.

## 8. Measured on real H3 attention, 2026-08-05

Run on an RTX PRO 6000 Blackwell (sm_120) against q/k/v captured from an actual
render at 864x480x124, step 5, blocks 0 and 24. `S = 15731, H = 56, D = 128`.

**Random tensors were tried first and are worthless here.** Gaussian q/k/v give
rel_rms 0.17 even at beta = 0, because unstructured attention has no dominant
blocks to keep. Any class measured that way describes the worst possible input.
It is a useful negative control and nothing else: whatever quality Sol-Attn
delivers comes from real structure in real attention.

### The equivalence class, at kijai's beta = 1.2 with `exact_kv`

| | rel_rms | cos | kernel speedup |
|---|---|---|---|
| block 0 | **0.132** | 0.9914 | 2.39x |
| block 24 | **0.245** | 0.9710 | 2.72x |

**This port gates DiT blocks against CUDA at rel_rms 8.5e-03 to 2.3e-02.**
Sol-Attn is 10-30x outside that. It cannot be gated against the existing
goldens, it must declare its own class, and conformance has to run per backend.
That is what `equivalenceClass` on the protocol is for, and this is the number.

### The sink is worth more, and costs more, than either source says

At beta = 1.2, block 0, measuring the conditioning rows separately:

| | conditioning-row rel_rms | time |
|---|---|---|
| no sink | 0.210 | 7.27 ms |
| `exact_kv` | **0.153** (-27%) | 8.52 ms (**+17%**) |

NVIDIA quote ~1% cost for the prefix sink; kijai's node quotes ~3% for
`exact_kv`. **Measured here it is 17%** at this shape. The benefit is real and
larger than advertised — 27% less error on exactly the rows carrying text,
audio and lip-sync — but it is not close to free.

### The middle of the stack is hardest to approximate

All three blocks, beta = 1.2 with `exact_kv`:

| block | rel_rms | conditioning rows | speedup |
|---|---|---|---|
| 0 | 0.132 | 0.153 | 2.37x |
| **24** | **0.245** | 0.337 | 2.69x |
| 49 | 0.142 | 0.330 | 2.91x |

Sensitivity peaks in the middle: block 24 is ~1.8x harder to approximate than
either end.

**This does not straightforwardly contradict the `dense_blocks` convention, and
it is worth being precise about why.** What is measured here is how well
Sol-Attn approximates *that block's own attention output*. The convention —
the paper's first-layer warm-up, kijai's "first and last" default — is about
*propagation*: error in the last block reaches the output undamped, while error
in block 0 has 49 blocks of processing after it. Those are different
quantities, and both claims can hold at once.

The right `dense_blocks` choice is approximation-difficulty weighted by
propagation, and only the first factor is measured here. What the numbers do
say is that excluding only the ends leaves the hardest block sparsified.

Also worth noting: the **conditioning rows degrade much more deeper in the
stack** — 0.153 at block 0 against 0.330-0.337 at blocks 24 and 49. The sink
does most of its work early.

### Beta means on H3 what it means elsewhere

Measured over 1000 attention calls of a real render:

| | measured | Gaussian |
|---|---|---|
| skewness | +0.231 | 0 |
| excess kurtosis | **+2.012** | 0 |
| density at beta=1 | 14.67% | 15.87% |
| density at beta=2 | 2.69% | 2.28% |

The distribution is **not** Gaussian — clearly heavy-tailed — but `1 - Phi(beta)`
still predicts density within 8% and 18%. So published tau settings transfer to
this model. The Cornish-Fisher correction, which uses exactly the third and
fourth moments above, is earning its place here.

### What it is worth end to end

2.4x on the attention kernel, and attention is 37% of DiT FLOPs at this shape,
so **~1.25x end to end** — independently reproducing NVIDIA's stated 1.25x for
H3 from a different direction.

Against the cross-step cache already shipped at **1.93x with no kernel at all**,
that keeps Sol-Attn third on the list.

### Reproducing this

Triton **3.3.0 cannot compile the kernel on sm_120** — `PassManager::run failed`
in `make_ttgir`. 3.7.1 works. The kernel modules also run standalone without
ComfyUI: `__init__.py` imports comfy, but `_tri_fwd`, `_preprocess` and
`_int8_fwd` only import each other, so a package directory with an empty
`__init__.py` is enough.

Routing goldens — per-(query block, head) thresholds plus pooled `kc`/`vc` at
beta 1.2 — were at `/Volumes/scratch_disk/H3_renders/qkv/routing_*.pt`, 7.4 MB
each. A Metal kernel that selects the same blocks implements the same
approximation; comparing routing is a stronger test than comparing outputs.

> **Those files no longer exist.** `H3_renders` is gone from every mounted
> volume, and with it both the routing goldens and the captured q/k/v the whole
> of §8 was measured against. Every number in §8 stands as recorded and none of
> it can currently be re-derived. Re-capturing is not hard — `DiTBlock.qkvCapture`
> still exists and writes the same layout — but it needs a render.

## 9. Open questions before a Metal port

**Question 1 is answered and the port exists.** See §10. Questions 2 and 3 are
open and are what stop the backend being selected by `auto`.

1. ~~**Is `MLXFast.metalKernel` expressive enough for the inner loop?**~~ Yes.
   The API JIT-compiles an arbitrary Metal function body with template
   parameters and threadgroup control, so the data-dependent branch is ordinary
   Metal control flow. It turned out not to be the difficulty at all — the
   selection is uniform across a threadgroup, because every thread in one shares
   a `(head, query block)`, so there is no divergence and the branch is free.
   What actually costs is arithmetic throughput; see §10.
2. **What is the equivalence class, per β?** The protocol requires one and the
   paper does not provide it. It has to be measured against dense output at
   production shape, and it will not be a single number — the sensible product
   surface is two or three named presets, each with a measured class.
3. **Which sink policy does H3 actually need?** 1% or 17%, decided by WER on the
   rendered waveform rather than by opinion.

## 10. The Metal port, measured 2026-08-05

`SolAttnRouting` + `SolAttnMetalKernel` + `SolAttnBackend`, against
`SolAttnReference` as the numerical definition. One threadgroup per (head, query
block); 256 threads over 64 query rows, four lanes per row, each lane owning 32
of the 128 head dimensions and the four reducing through `quad_shuffle_xor`.

Measured on the M3 Ultra at the verified shape — `H = 56, S = 15731, D = 128`,
bf16, beta 1.2 with the exact-KV sink:

| | time | vs dense |
|---|---|---|
| dense `sdpa` (MLX fused) | 428.8 ms | 1.00x |
| routing only (pool + proxy + tau) | 3.5 ms | 0.8% of dense |
| sol-attn, routing + kernel, blockSize 64 | 232.6 ms | **1.84x** |
| sol-attn, blockSize 128 | 209.4 ms | **2.05x** |
| sol-attn, blockSize 256 | 200.1 ms | **2.14x** |

Selected density is 0.173 at every block size, so a kernel that cost nothing
would give 5.5x. At 2.14x the kernel is running at **39% of that ceiling** —
about 20% of the GPU's fp32 peak against the ~61% MLX's fused SDPA achieves.
The gap is entirely arithmetic throughput: this kernel accumulates in scalar
registers with `float4` loads, and MLX's uses the matrix units. `simdgroup_matrix`
is the obvious next move and is worth roughly the remaining 2.5x.

**Routing cost is not the problem, and that was worth measuring before optimising
anything.** The paper's O(d) variance estimator (Eq. 15) exists to avoid
materialising the N x N proxy map; at this sequence length that map is 247^2 per
head and pooling plus proxy plus threshold costs 0.8% of the dense call. There is
no reason to implement the cheaper estimator until the sequence is far longer.

### Block size: there is no trade-off, there is a cliff

Swept on structured input at production shape, at two different input locality
scales, measuring density, speed and rel_rms together:

| blockSize | locality 256 | | | locality 1024 | | |
|---|---|---|---|---|---|---|
| | density | speedup | relRMS | density | speedup | relRMS |
| 64 | 0.143 | 2.20x | 0.0026 | 0.149 | 2.13x | 0.0027 |
| 128 | 0.143 | 2.45x | 0.0026 | 0.149 | 2.35x | 0.0027 |
| 256 | 0.143 | **2.62x** | 0.0026 | 0.149 | 2.43x | 0.0027 |
| 512 | 0.221 | 1.78x | 0.0057 | 0.149 | 2.62x | 0.0027 |
| 1024 | 0.359 | 1.09x | 0.2080 | 0.148 | **2.68x** | 0.0027 |
| 2048 | 0.391 | 1.02x | 0.9661 | 0.281 | 1.40x | 0.0059 |
| 4096 | 0.375 | 1.03x | 1.2909 | 0.375 | 1.03x | 0.5218 |

**The knee sits exactly at the input's locality scale, and moves when that
moves.** That is the control worth running: a knee that always landed at 1024
would be a fact about the generator. Below the knee, speed rises monotonically
and rel_rms does not move at all; above it, density jumps, and speed and quality
collapse *together*. Both collapses have one cause — a block wider than the
content's coherence spans unrelated material, its pooled key stops representing
anything, and more blocks clear the threshold.

So block size is not a speed-versus-quality dial. Below the locality scale
larger blocks are free; above it they cost on both axes at once. The only
question is where H3's own knee is.

### H3's locality scale is derivable, not unknown

At 864x480, `spatialDownsample` 16 gives a 54 x 30 latent, and `patchSize`
`[1, 2, 2]` patchifies that to 27 x 15 = **405 tokens per latent frame**, with
**27 tokens to a latent row**. `latentT` is 37 at 124 frames, so 14,985 video
tokens plus 414 audio — which is the 15,731 everything here is measured at.

That reframes `blockSize` 64: it is 2.4 rows of one frame — the "two-row strip"
the `TokenOrdering` documentation describes, chosen because it is the published
default and not because anything about H3 suggests it. One latent frame is 405
tokens, so the first block size that is a whole spatial neighbourhood rather than
a sliver is 384 or 448, and temporal correlation between adjacent latent frames
may well put the real knee above that.

**The default stays 64 for now regardless**, because the sweep above is
synthetic: it establishes the *shape* of the curve and that the knee tracks
locality, not where H3's knee is. Settling that needs captured q/k/v — the same
capture the equivalence class needs — and it is the same experiment, so it costs
nothing extra to answer both at once. If the knee lands where the frame geometry
suggests, the default should be 384 or 448 and that is worth roughly another
1.2x on top of what is measured here.

> **It did not, and §11 measures the opposite.** On real H3 attention there is
> no flat region at all: quality degrades monotonically from blockSize 64
> upward. The frame-geometry argument above is left standing because it is the
> reasoning that motivated the experiment, and because the experiment refuted
> it — 384 measures three times worse than 64 at block 49. Synthetic input with
> clean block-contiguous clusters has a free region; H3's attention does not.

### What is established, and what is not

Established, by test:

* the kernel reproduces `SolAttnReference` to rel_rms < 2e-3 at head dim 128 —
  including the short tail block, the conditioning sink, bf16 input, and routing
  blocks of 128 and 256 spanning several threadgroups;
* the reference reproduces dense attention **exactly** (< 1e-6) in both cases
  where the method degenerates to it: every block selected, and every block
  constant so the rank-1 correction is algebraically exact;
* the correction beats masking — same selection, rejected blocks dropped — which
  is the only reason to carry it.

Not established: **the equivalence class of this implementation**. Everything
above says the code computes Sol-Attn correctly. None of it says how far
Sol-Attn sits from dense on real H3 attention, which needs q/k/v captured from a
render on this machine. Until that number exists, `SolAttnBackend` is listed
after `SDPABackend` in the registry, so `auto` resolves to dense and reaching it
requires `--attention sol`.

## 11. Measured on this machine, against real captured attention

q/k/v captured from a live render at 864x480x124 — `H = 56, S = 15993, D = 128`,
progress 0.25, blocks 0, 24 and 49 — via `DiTBlock.qkvCapture`, and stored at
`/Volumes/big_daddy/scratch_disk/H3_Swift/qkv`. Dense reference is MLX's fused
SDPA on the same tensors, 451 ms.

`S` is 15,993 rather than the 15,731 quoted elsewhere because the capture prompt
is longer; the video tail is 14,985 tokens either way, so the conditioning prefix
is 1,008 rows here.

### The equivalence class, at beta 1.2 with the exact-KV sink

| block | density | rel_rms | conditioning rows | speedup |
|---|---|---|---|---|
| 0 | 0.137 | 0.193 | 0.183 | 2.23x |
| 24 | 0.164 | **0.287** | 0.369 | 1.91x |
| 49 | 0.152 | 0.222 | 0.481 | 2.01x |

**`SolAttnBackend.equivalenceClass` is 0.29**, the worst of these rounded up.

**§8's central qualitative finding reproduces from an independent
implementation**: sensitivity peaks in the middle of the stack. Block 24 is the
worst block in both, on different hardware, in different languages, against
different captured tensors.

The absolute figures sit consistently above the Triton kernel's — 0.193/0.287/0.222
here against 0.132/0.245/0.142 there. The most likely cause is the **Cornish-Fisher
correction**, which kijai's node applies from the third and fourth moments of the
proxy distribution and this implementation does not. §8 measured H3 at skewness
+0.231 and excess kurtosis +2.012, so a Gaussian threshold is systematically
mis-placed on this model and that correction is earning its place. It is the
obvious accuracy work, as `simdgroup_matrix` is the obvious speed work.

### Beta

| beta | density | rel_rms (block 24) | speedup |
|---|---|---|---|
| 0.8 | 0.245 | 0.208 | 1.43x |
| 1.0 | 0.200 | 0.246 | 1.69x |
| 1.2 | 0.164 | 0.287 | 1.91x |
| 1.5 | 0.123 | 0.358 | 2.29x |
| 2.0 | 0.085 | 0.492 | 3.03x |

Measured density tracks `1 - Phi(beta)` loosely and runs high — 0.085 at beta 2.0
against a Gaussian 0.023, before the sink's ~6.3% is subtracted — which is the
heavy tail again.

### Block size: no free region on real attention

At beta 1.2, rel_rms against dense:

| blockSize | block 0 | block 24 | block 49 | speedup (block 0) |
|---|---|---|---|---|
| 64 | **0.193** | **0.287** | **0.222** | 2.23x |
| 128 | 0.219 | 0.323 | 0.371 | 2.64x |
| 256 | 0.262 | 0.385 | 0.511 | 2.99x |
| 384 | 0.259 | 0.400 | 0.671 | 2.91x |
| 512 | 0.269 | 0.431 | 0.651 | 3.18x |
| 1024 | 0.303 | 0.490 | 0.685 | 3.13x |

**This refutes §10's synthetic result and the frame-geometry argument built on
it.** There is no flat region: error rises from 64 upward at every block, and
worst at block 49, where 384 is three times worse than 64. One latent frame being
405 tokens turns out not to make 384 a natural block size — real attention does
not respect the frame boundary the way synthetic clusters respected their segment
boundary.

Speed does improve with block size, so this is a genuine quality-for-speed dial
rather than a free lunch. **The default stays 64**, which is now a measurement
rather than an inherited convention.

### What the sink costs and buys

At block 0, where it does most of its work:

| | rel_rms | conditioning rows | speedup |
|---|---|---|---|
| `exact_kv` sink | 0.193 | **0.183** | 2.23x |
| no sink | 0.250 | 0.564 | 3.49x |

The sink cuts conditioning-row error by **68%** and costs **36% of the speed** —
against NVIDIA's quoted ~1%, kijai's ~3%, and the 17% §8 measured on CUDA. It is
much more expensive here and worth every bit of it: those rows carry text, audio
and lip-sync. Deeper in the stack the benefit shrinks (block 49: 0.481 with,
0.502 without), confirming §8's "the sink does most of its work early".

### What this does not settle

Whether 0.29 is acceptable. It is more than ten times outside the 8.5e-03 to
2.3e-02 band this port gates DiT blocks against, and no tensor comparison can
decide the question — it has to be answered on rendered output: Whisper WER on
the generated waveform and face-landmark detection per frame, against a dense
render at the same seed. That A/B is the next experiment and it is what stands
between this backend and `auto`.

## 12. The overnight sweep, 2026-08-06 — and where this actually lands

31 renders, all with the cross-step cache on (`--quality balanced`), because
that is what anyone renders with; tuning against `faithful` optimises a
configuration nobody uses. Full data in
`/Volumes/big_daddy/scratch_disk/H3_Swift/sweep/results.csv`.

### The artifact none of §8 or §11 could see

A viewer watching the first sparse render reported the subject "pulsing and
warping". **Every measurement in this document up to that point was blind to
it**: rel_rms on a single attention call says nothing about coherence across
frames. Localised frame-to-frame acceleration in the quietest shot reproduces it
— dense 0.84, sol 1.74 — and on a static single-shot talking head, where there
is nothing else moving, the default config measures **2.1x dense**.

That is the honest lesson of the exercise. A per-call tensor metric was the
wrong instrument, and only a human watching the output found the failure.

### Rankings (race prompt, two seeds, cache on)

| config | temporal vs dense | detail vs dense | speed vs dense |
|---|---|---|---|
| `blockSize 256` | 4.66x | 0.67 | 1.11x |
| `blockSize 128` | 2.59x | 0.68 | 1.10x |
| `beta 1.5` | 2.38x | 0.73 | 1.12x |
| **default** (b1.2) | 1.40x / 1.10x | 0.86 / 0.57 | **1.09x** |
| `beta 0.8` | 1.39x | **0.95** | 1.04x |
| `edges 5` | 1.06x / 0.66x | 0.84 / 0.61 | 1.09x |
| **`b0.8 + edges5 + warm.35`** | **0.89x / 0.80x** | 0.85 / 0.72 | **1.02x** |

Two seeds shown where measured. **blockSize is settled: 64 is right**, and
larger is not a dial — it is worse on every axis at once, confirming §11 from a
different direction.

**The sweep's own noise floor is ~0.4 on this metric.** beta is monotonic by
construction, yet 0.8/1.0/1.2 measured 1.883/2.273/1.894 — a spread of 0.39 on a
knob whose true ordering is known. Every render is a different sample. Single-run
rankings inside that band mean nothing, which is why everything above was
replicated.

**A blur trap, which nearly inverted the ranking.** Configurations scoring
*better than dense* temporally were partly just softer — `edges5` on seed 11 is
0.66x dense on acceleration and 0.61x on detail. Low acceleration is only good
news if the detail survived. Against the one human-calibrated point available
(the pair a viewer called "both look good", at a 0.90 detail ratio), several
"winners" sit below the bar.

### The mechanism is error magnitude, not routing instability

Three candidates, and the measurement discriminates:

* **Routing churn** — selection recomputed per step with nothing tying steps
  together. **Ruled out.** Block 24 at consecutive steps: Jaccard **0.92**, with
  1.3-2.0% of blocks flipping and density identical. Hysteresis would change
  almost nothing.
* **Block geometry** — a 64-token block is 2.4 rows of one latent frame. Still
  open, and the `TokenOrdering` seam exists for it, but it is not implemented.
* **Approximation magnitude** — the likeliest, by elimination and by the
  evidence. At rel_rms 0.29 per call the error is large and its *direction*
  differs every step because the latents do, so it injects roughly uncorrelated
  noise into each step of the trajectory. That is temporal incoherence directly.
  It also explains why the only things that helped were the ones that inject
  less error — lower beta, more dense blocks, more dense steps — rather than
  anything that stabilised the routing.

**So the productive work is accuracy at fixed sparsity, not scheduling.** The
Cornish-Fisher correction is the concrete item: §8 measured H3 at excess kurtosis
+2.012, this port omits the correction, and §11 measured this implementation
0.04-0.08 worse than the Triton kernel at matched beta.

### Whether it is worth having, at this shape: no

| | 5 s | 10 s |
|---|---|---|
| default settings | 1.09x | **1.17x** |
| tuned to near-dense quality | 1.02x | 1.06x |

**Tuning away the artifact consumes the speedup.** `edges5 + warm.35` cuts sparse
coverage from 77% of block-steps to 52%, and 1.09x becomes 1.02x. The gain that
survives at quality is 2-6%.

Three structural reasons, none of them a defect in the port:

1. **Amdahl.** Attention is 37% of DiT FLOPs at 864x480x124, so even a free
   attention kernel caps the whole render at 1.59x.
2. **The cache gets there first.** On a reuse step blocks 1-49 are skipped
   wholesale and Sol-Attn never runs. Cache alone is 1.79x measured here
   (1324 s -> 739 s); Sol-Attn adds 9% on top of that, not another 93%.
3. **The kernel is at a third of its own ceiling** — ~20% of GPU peak against
   MLX's ~61%.

**But the length trend is real and in the predicted direction**: 1.09x at 5 s
becomes 1.17x at 10 s as attention's FLOP share rises toward 54%. Sol-Attn earns
its place on long, large renders — exactly what §7 said — and not here.

### Conclusion

`SolAttnBackend` stays behind `SDPABackend` in the registry. It is correct, it
preserves lip-sync (124/124 landmarks on a talking head, dense and sparse
alike), and at this shape it is not worth its quality cost. The two pieces of
work that would change that are `simdgroup_matrix` (3x kernel headroom) and the
Cornish-Fisher correction (accuracy at fixed sparsity), in that order.
