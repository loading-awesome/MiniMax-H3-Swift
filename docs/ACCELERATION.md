# Acceleration — what was measured, and what it costs

Updated 2026-08-05. Every number here was measured on an M3 Ultra at
864x480x124, 20 steps, seed 11, against an uncached control of the same prompt
and seed.

The order of these techniques is not intuition. NVIDIA's own H3 breakdown of
their 3.95x on 8xGB200 puts **cross-step caching first** (1.534x -> 3.95x of the
total), **fused RMS-AdaLN second** — "the primary win" for H3's 13B AdaLN
parameters — and **sparse attention third**, at 1.25x additional. Only the third
needs a Metal kernel, which is why it is last here too.

---

## 1. Cross-step cache

A diffusion step produces a *change* to the latent, and on a back-loaded
schedule adjacent steps often produce nearly the same change. Block 0 is a cheap
probe: run it, compare its residual against last step's, and if it has barely
moved, reuse the previous **total residual** and skip the other 49 blocks.

Caching the total residual rather than the output matters — caching the output
would pin the render to a stale latent, where a reused *delta* applies to
wherever the trajectory has actually reached.

### The sweep

| threshold | steps skipped | wall clock | speedup | **detail (lap var)** | vs control | audio envelope | audio spectral |
|---|---|---|---|---|---|---|---|
| control | 0/20 | 1381 s | — | **0.00349** | — | 1.000 | 1.000 |
| 0.05 | 0/20 | 1324 s | none | 0.00349 | 0% | **1.000** | **1.000** |
| **0.10** | 10/20 | 716 s | **1.93x** | 0.00293 | **-16%** | 0.995 | 0.971 |
| 0.15 | 13/20 | 531 s | 2.60x | 0.00250 | -28% | 0.994 | 0.955 |
| 0.25 | 14/20 | 471 s | 2.93x | 0.00197 | **-44%** | 0.990 | 0.934 |

**0.10 is the default**, because the trade turns bad immediately after: 0.15 to
0.25 buys 13% more speed and costs another 16% of high-frequency detail.

The 0.05 row is worth keeping even though it skipped nothing. The observed
change never fell below the threshold, so the cache was present and never fired
— and the output came back at envelope **1.000**, spectral **1.000**: bit
identical to the uncached control. That is the proof that the block-0 split
perturbs nothing on its own.

### The measurement that inverts the naive reading

**Caching does not make the render cleaner. It makes it softer, and the two
look identical in every metric except detail.**

Under caching, frame-to-frame luminance variance *falls* (sd 0.0021 -> 0.0007),
motion falls 25%, and lip-sync correlation *rises* (r 0.762 -> 0.823). Read
alone, every one of those looks like an improvement, and the first draft of
this analysis reported them as one — twice, citing a community report that
caching "made output cleaner."

It is not cleaner. Reusing a residual smooths the trajectory, and smoothing
shows up as lower variance everywhere at once. Only Laplacian variance
distinguishes "stable" from "blurred", and it says the 0.25 render has lost
**44% of its high-frequency content**.

The same trap catches lip-sync: **correlation measures alignment, not richness**.
Blander mouth motion correlates *better* with a smooth audio envelope while
being worse animation. No single oracle catches this. Running all of them does.

## 2. The probe is measured per stream, and the A/B says why

At 864x480x124 the packed sequence is **95.1% video rows and 2.6% audio rows**.
Every published cache for this model averages its probe over the whole sequence,
so the decision is made almost entirely by the video tokens.

Measured, the two streams do not move together:

```
median change 0.079 = max(whole-sequence 0.060, audio-only 0.079)
```

The audio residual moves **32% more per step** than the whole-sequence average.
A whole-sequence probe systematically under-reports it and skips steps the audio
needed.

`H3StepCache` therefore measures both and gates on the worse. Whether that
matters was settled by an A/B at threshold 0.10 changing nothing but the probe:

| | per-stream | whole-sequence |
|---|---|---|
| steps skipped | 10/20 | 12/20 |
| audio envelope vs control | +0.995 | +0.996 |
| audio spectral vs control | +0.971 | +0.970 |
| WER | 0.20 | 0.20 |
| **lip-sync r** | **+0.783** | **+0.664** |
| **lip-sync margin over control** | **+0.718** | **+0.453** |

**Half of the original hypothesis was wrong.** The audio *waveform* is
unaffected by the probe choice — envelope and spectral correlation match to
three decimals, and so do the transcripts. Caching does not damage audio
quality, and the reported community failure ("all cache methods currently warp
audio") is not what this measures.

**What the probe protects is synchronisation.** The lip-sync margin drops 37%
with a whole-sequence probe. That is mechanically sensible: audio and video are
generated in one forward pass, so skipping a step the audio needed leaves the
two trajectories out of step. The waveform still sounds correct in isolation; it
no longer matches the mouth. **That is what lip drift is**, and it is invisible
to every oracle except `lipsync-check`.

**The confound, and why it probably does not explain the result:** the
whole-sequence arm skipped two more steps, so more approximation could account
for worse sync on its own. But per-stream at *13* skips (threshold 0.15) scored
margin **+0.776** — better than per-stream at 10 skips, and far better than
whole-sequence at 12. Probe choice, not skip count. Still one render per arm.

## 3. Three ways the cache can be got wrong

**One cache per conditioning stream.** CFG runs two forwards per step with
different conditioning; their residuals are not comparable. A shared cache
compares a conditional residual against an unconditional one and reuses across
the gap, with every number staying finite.

**Bound consecutive reuse.** A skipped step does not update the probe, so the
same comparison recurs and the sampler can coast to the end on one stale delta.

**Never skip the first or last step.** The first has no history; the last lands
directly in the decoded pixels.

## 4. Not started, and why

**Fused RMS-AdaLN** is second on NVIDIA's list and this port already has the
AdaLN residency machinery. Bounded kernel work against `MLXFast.metalKernel`.

**Sol-Attn** is third at 1.25x on H3 and needs a Metal kernel written from
scratch — Triton does not target Metal. Measured on this machine: MLX's masked
attention path never skips a K block, so handing it a 90% sparse mask is
**1.02x slower** than dense. But genuinely shortened K/V delivers **8.45x at 10%
density**, so the hardware pays out at Sol-Attn's target sparsity — 82-100% of
the proportional speedup. See `SOL_ATTN.md`.

Amdahl bounds it: attention is 37% of DiT FLOPs at S=15,750, rising to 54% at
S=31,500. Sparsification alone caps near 1.5x at the verified shape and ~1.9x at
twice the length, so it earns its place on long renders rather than short ones.
