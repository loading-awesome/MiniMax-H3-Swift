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
beta 1.2 — are at `/Volumes/scratch_disk/H3_renders/qkv/routing_*.pt`, 7.4 MB
each. A Metal kernel that selects the same blocks implements the same
approximation; comparing routing is a stronger test than comparing outputs.

## 9. Open questions before a Metal port

1. **Is `MLXFast.metalKernel` expressive enough for the inner loop?** The exact
   branch is data-dependent control flow over a selection buffer — the one place
   Metal's occupancy model differs most from CUDA's. The criterion itself is
   cheap arithmetic on block centroids and needs no exotic hardware.
2. **What is the equivalence class, per β?** The protocol requires one and the
   paper does not provide it. It has to be measured against dense output at
   production shape, and it will not be a single number — the sensible product
   surface is two or three named presets, each with a measured class.
3. **Which sink policy does H3 actually need?** 1% or 17%, decided by WER on the
   rendered waveform rather than by opinion.
