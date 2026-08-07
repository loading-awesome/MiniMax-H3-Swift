# Phase 6 — measured distribution and performance

Where the remaining speed is, what may be spent to get it, and what has to be
true before any of it ships.

The order is not negotiable and the reason is Sol-Attn. That work produced a
correct Metal kernel, a real speed number and a conclusion, and the conclusion
had to be revisited because the arms that mattered were run weeks apart with a
setting nobody wrote down. Nothing it produced could have revealed that. So the
evidence system is built first, and every phase after it is gated on numbers the
system can reproduce.

---

## 6A — Benchmark contract *(done)*

Every render writes two files beside its mp4, unconditionally:

| file | contents |
|---|---|
| `<output>.h3-bench.json` | identity, configuration, machine, phase timings, memory peak, the full step trace |
| `<output>.h3-steps.csv` | one row per branch-step: sigma, three deltas, decision, reason, wall clock |

`h3 bench --dir <path>` reads them and prints the table.

**Unconditional is the design.** A harness you have to remember to switch on
produces evidence for the runs somebody expected to need it, which is never the
run that turns out to matter. Two files of a few kilobytes, beside a
hundred-megabyte mp4.

### What the contract enforces

`BenchmarkRecord.speedup(over:)` **throws** rather than returning a number when
the two runs disagree on prompt, seed, dimensions, steps, cfg, checkpoint,
machine or MLX revision. A caveat in a log line is how the cache-disabled
Sol-Attn figures travelled; an error means the number does not exist. An
unverified checkpoint is recorded as `nil` and reported as an absence, never as
a mismatch — two runs that both skipped verification are not known to differ.

Configuration is exempt, because configuration is what an experiment varies. It
is not silent, though: `h3 bench` prints every knob that differs from the
control beside the speed-up, including defaults resolved at run time such as
`resolved.fusedModulation`. A number next to an unmentioned difference is a
result attributed to the wrong cause.

Arms that cannot be lined up still appear in the table, with the reason. A
comparison tool that drops what it cannot use produces a clean table with an
invisible hole in it.

### The reported figure is median seconds per step

Not wall clock, which includes a checkpoint load whose cost depends on what the
page cache happened to be holding. Not `sampling / steps`, which every
configuration dilutes by the first and last steps it is required to run in full,
by an amount that varies with step count.

Timing boundaries sit **after an explicit `eval`**. MLX is lazy: a clock stopped
before one measures graph construction and hands the arithmetic to whoever
forces it. Under a cache that is not a small error — a skipped step builds
almost no graph, so the arm doing less work would appear to have moved its cost
rather than saved it.

### Three deltas, two votes

The cache measures block 0's residual change over the whole packed sequence, the
audio rows and the video rows. It decides on `max(whole, audio)`, exactly as
before. **The video figure is recorded and deliberately given no vote** — an
instrument that changes what it measures is worthless as a control, so the
baseline is established first and 6C decides afterwards.

Before this, the variable holding the whole-sequence change was named
`observedVideoChange`.

### Refusals are attributed

"14 of 20 steps skipped" cannot distinguish a threshold doing the work from a
consecutive-cap absorbing the consequences of one set too loose. `reasonCounts`
can: `belowThreshold`, `videoAboveThreshold`, `audioAboveThreshold`,
`consecutiveCap`, `warmup`, `cooldown`, `nonFinite`, `noHistory`. An arm whose
refusals are mostly `consecutiveCap` has a threshold past the point where its
own signal means anything, and the cap is the only thing holding the render
together.

### Also fixed here

`model_load` is a phase. It was in none, and `total` was the sum of the phases,
so a minute or more of every render belonged to nobody. Harmless until the
numbers mattered.

### The controls *(measured 2026-08-06)*

Eight renders, 864×480×124, 20 steps, seed 7, one prompt, one machine, one
binary.

> **Every number in this section is conditioned on one easy prompt** — "a red
> kite over a beach at sunset". Smooth sky, sand and water, almost no
> high-frequency content. It was chosen for being the README's example, which
> made it a convenient control and a poor probe.
>
> The worry was that this conditions the speed figures too — reuse count
> depends on how far the residual moves per step, so a calm scene should
> qualify more steps than a busy one, making 45% and 1.79× easy-content
> artefacts.
>
> **Measured, and it does not.** A busy market street with a speaking subject
> reused the same 9 steps at cap 3, with a median whole-sequence delta of 0.080
> against the beach's 0.082 — step for step, the same U-shaped curve. The delta
> trajectory is driven by the flow schedule, not by scene content, so the speed
> figures here generalise.
>
> The **quality** figures in this section do not, and one of them was wrong for
> an unrelated reason. See the retraction under 6C.

```
arm                             runs    s/step  full step   spread  speedup  reused
control-dense                      2     59.39      59.21     1.4%    1.00x      0%
control-cached                     3     33.23      58.84     1.1%    1.79x     45%
fused-cached                       3     32.64      58.29     0.9%    1.82x     45%
```

**The cross-step cache is worth 1.79×** and that is the number every later
proposal has to add to, not replace.

**The instrument checks out.** A full step under the cache costs 58.84 s and a
step under dense costs 59.21 s — the same work, agreeing to 0.6%, inside the
1.4% control spread. The cache does not make steps cheaper; it makes fewer of
them. If those two figures ever diverge, something is measuring the wrong thing.

Repeat spread is 0.9–1.4%. **Any claimed gain below about 1.5% is not a gain.**

### Renders are not bit-reproducible, and that bounds every quality claim

Three `control-cached` runs at the same seed and configuration produced three
different mp4s. Decoding them separates what varies from what does not:

| stream | result |
|---|---|
| audio | **bit-identical across all three** |
| video | runs 2 and 3 identical to each other; run 1 differs |

So the video path carries some run-to-run nondeterminism — plausibly MLX kernel
or accumulation-order selection responding to machine state, since the seed,
the configuration and the audio are all fixed. The practical consequence is a
noise floor on every visual metric:

```
                        accel   detail   motion    dssim
control-cached-2/3      0.979    0.991    0.973   0.0003   <- same config, noise
fused-cached-1          1.000    1.024    0.998   0.0017
control-dense-1         1.039    1.013    1.069   0.0138   <- different scene
```

**A quality difference below roughly 2% on `accel` or 1% on `detail` cannot be
attributed to a configuration change.**

**And note what is missing from that table.** It once carried a third
calibration — "dense carries 8.4% more high-frequency detail than cached, which
is the cache's actual quality cost". That number was 1.084 measured on
downsampled frames; natively it is 1.013, and dense sits at dssim 0.0138, which
is a different scene. It was a filter artefact attached to an invalid
comparison, and it was quoted three times before it was checked.

**That does not mean the cache's cost is unmeasured — it means this session
measured it badly.** `docs/ACCELERATION.md` has a threshold sweep from
2026-08-05 that is better controlled than anything above:

| threshold | skipped | speedup | detail (lap var) | vs control |
|---|---|---|---|---|
| control | 0/20 | — | 0.00349 | — |
| 0.05 | 0/20 | none | 0.00349 | **0%** |
| 0.10 | 10/20 | 1.93× | 0.00293 | −16% |
| 0.15 | 13/20 | 2.60× | 0.00250 | −28% |
| 0.25 | 14/20 | 2.93× | 0.00197 | −44% |

Three things make that stronger than the measurements in this document. It has
a **zero-skip negative control** — at threshold 0.05 the cache was live and
never fired, and the output came back bit-identical to the uncached run, which
proves the block-0 split perturbs nothing by itself and that divergence comes
only from actual skipping. It shows a **monotonic dose-response** across four
points, which noise and recomposition do not produce. And its absolute values
(0.0035, 0.0029) sit in the same range as the *native* measurements here
(0.0021), not the downsampled ones (0.0068), so it was very likely measured at
full resolution.

**So the shipped claim of 16% stands, and it is the threshold axis.** The cap
axis is a different question, and the threshold table gives it a useful prior:
if detail cost tracks the number of skipped steps, then going from 9 skips to
10 — which is all cap 5 does — should cost roughly a tenth of 16%, call it 1.6%.
That is consistent with cap 5 clearing a viewing when the metrics had wrongly
condemned it.

**Do not run anything else on the GPU during a control sweep.** Test runs taken
beside the first attempt moved its step time from 25 s to 29 s, larger than most
of the gains this roadmap is chasing, and that sweep was discarded.

---

## 6B — Exact fusion *(measured; failed its gate; off by default)*

**Verdict first: 1.81% on wall clock, 0.94% on the steps the kernel touches,
against a 5% gate.** The gain is real — three fused runs versus three control
runs with no overlap between the sets, and the full-step figure repeating to
0.1% — it is simply small. `H3_FUSED_MODULATION=1` turns it on; the default is
the readable MLX path.

The arithmetic predicted this and was done afterwards, which is the lesson worth
keeping. Roughly fourteen `[S, hidden]` tensor passes saved per block at 169 MB
each, over fifty blocks, is about 118 GB — a few hundred milliseconds against a
58.8-second step. **A step at this shape is bound by attention and the large
GEMMs, not by modulation traffic.** MLX also fuses elementwise chains of its
own, so the path being replaced was never as naive as the source reads.

It is also not a free swap. The norm's reduction reassociates, so a few ulps at
block 0 propagate through fifty blocks and twenty steps into a different render
— same quality, different pixels. Enabling it would need its own quality pass,
which for 1% is not a trade worth making.

Kept rather than deleted because the measurement is machine-specific: the ratio
of memory bandwidth to compute is what makes this small here, and that ratio is
not the same on every Apple part. Anyone who turns it on should re-run the
controls first.

---

### What was built

`FusedModulation` folds RMSNorm, the per-token AdaLN gather and the modulation
into one pass, and does the same for the gated residual.

**The projection stays on MLX's `matmul`.** The 13 B AdaLN parameters are 26 of
the checkpoint's 66 GB and have to be read whatever happens; fusion cannot avoid
that and nothing written here would beat a tuned GEMM. What it removes is the
traffic downstream — most of all the two gathers, whose only job is to broadcast
about nine distinct table rows into a tensor the size of the hidden state so an
elementwise kernel can read them back.

At S = 15,731 and hidden = 5,376 a `[S, hidden]` bf16 tensor is 169 MB. Four
fused sites per block, fifty blocks, two CFG branches.

### Numerics

The gated residual is **bit-identical** in both dtypes — no reduction, every op
elementwise, both paths rounding at the same points.

The norm cannot be and should not be asked to be: it sums 5,376 squares and
MLX's reduction tree is not this kernel's. Measured at production width:

| dtype | relative RMS | bit-identical elements |
|---|---|---|
| fp32 | 4.3e-08 | 90.4% |
| bf16 | 2.6e-05 | 99.9994% |

A tolerance cannot tell *the sum was reassociated* from *the kernel is slightly
wrong*, so the norm is checked against a double-precision CPU oracle and the
assertion is that **fusion moves the error rather than adding to it**: 5.9e-08
fused against 5.7e-08 unfused, both at the fp32 noise floor.

Per-element deviation is bounded at eight ulp against a denominator carrying the
tensor's own scale. `h * (1 + s) + sh` cancels, so a near-zero result came from
operands of order one and its error is an ulp of *those*; dividing by the result
would report thousands of ulps and mean nothing. Worst observed: 3.0 ulp fp32,
0.6 ulp bf16.

### Gates

- [x] No increase in peak memory — 87.2 GB on every arm, identical.
- [x] Full suite green.
- [ ] **Median production-step improvement ≥ 5% — FAILED at 0.94%.**

The remaining gates (CUDA tap conformance) were not run: a kernel that is off by
default and fails its speed gate does not need its conformance re-established.
They become live again only if someone turns it on.

Separately evaluable and not yet attempted: a fused Q/K RMSNorm plus split-half
RoPE. Kept separate because the oracle already measured the unfused attention
path as bit-identical to the reference's fused kernel, so this one is a pure
traffic argument with no correctness upside.

`H3_FUSED_MODULATION=0` forces the readable path at run time, for bisecting a
suspected difference on a real render without a rebuild.

---

## Priority after the controls

The measured trace reorders what was planned. Sigma-curve work is **dropped**
unless a later trace shows a threshold-controlled region worth fitting — the
cache already identifies the useful middle window without one.

```
1. Consecutive-cap sweep: 4 -> 5 -> 6            speed done; quality UNRESOLVED
2. A quality method that survives recomposition  <- blocking 1
3. AdaLN schedule batching
4. MLX dense-block compilation
5. Canonical weight layout and GEMM profiling
6. Selective resident quantized matmul research
```

Items 3–6 are untried and, unlike 6B, are not bounded above by activation
traffic — 6B measured 1% because a step at this shape is bound by attention and
the large GEMMs, which is where 5 and 6 aim.

**Cap 5 is a live 6.5–10% that cannot currently be accepted or rejected**,
because no measure in this tree can compare two renders that depict different
scenes. That is now the blocking problem for the cache, and it blocks any
future cache work too — every change to this policy moves the trajectory.

Items 3–6 are not blocked by it. A kernel that computes the same arithmetic
faster leaves the trajectory alone, so it can be validated by the equivalence
classes and taps this tree already has. 6B was settled in an afternoon for
exactly that reason.

---

## Where the time actually goes — do this arithmetic first

**6B was built before anyone worked out what it could possibly be worth.** It
could not have exceeded about 1%, and the sum takes ten minutes. So here is
that sum for everything left, from the measured control: 660.5 s of sampling
over 20 steps with 9 reused, giving **60.0 s per full forward**, at cfg 1.0 so
one forward per step.

At S = 15,731, hidden 5,376, 50 blocks, inner 7,168, ffn 14,336:

| per block | TFLOP | share of a forward |
|---|---:|---:|
| attention (QK^T and A·V) | 7.095 | **36.9%** |
| mlp fc1 | 4.850 | 25.2% |
| qkv projection | 3.637 | 18.9% |
| mlp fc2 | 2.425 | 12.6% |
| attention out projection | 1.212 | 6.3% |
| AdaLN projection | 0.0016 | **0.008%** |

**961 TFLOP per forward, achieved in 60.0 s — 16.0 TFLOP/s.**

**The model is compute-bound, not bandwidth-bound.** Reading all 66.3 GB of
weights once per forward is 83 ms at 800 GB/s: **0.1% of a step.** Every
argument in this tree that reasons from memory traffic — including the one that
motivated 6B — is reasoning about a rounding error.

### What that does to the remaining items

| item | ceiling | verdict |
|---|---:|---|
| 3. AdaLN schedule batching | ~0.05% | **dead.** The projection is 0.008% of FLOPs and its weight reads are 26 GB — 0.05% of a step. Precomputing all 20 steps saves a rounding error, and `AdalnProj` already measured its fp32 upcast at 0.2%. |
| 4. MLX dense-block compilation | ~0.05% | **dead.** Roughly a thousand kernel launches per forward at ~30 µs is 30 ms against 60 s. |
| 5. Canonical weight layout, GEMM profiling | up to 63% of FLOPs | **live.** This is the target. |
| 6. Selective int8 matmul | same 63%, ~2× arithmetic rate | **live**, at a quality cost that needs the same measurement problem solved. |
| (Sol-Attn, axial) | 36.9% of FLOPs | already measured, already rejected on quality |

So the priority list collapses. Items 3 and 4 are struck; the only untried
work with a multiple in it is making the dense GEMMs faster, or doing them in
fewer bits.

### Measured: the model is at MLX's ceiling, and the roadmap closes

`GEMMCeilingTests` (`H3_BIG=1 swift test --filter gemmCeiling`), isolated
kernels at production shapes.

> **These rows use a contiguous `[K,N]` right operand, which is not the layout
> production runs.** Kept because they are the correct reference for MLX's
> attainable rate, and because the conclusion drawn from them originally was
> wrong in a way worth leaving visible. The layout the model actually uses is
> in the table below them.

| kernel | ms | TFLOP/s |
|---|---:|---:|
| qkv `[S,H]×[H,3I]` | 209.2 | 17.4 |
| attention out `[S,I]×[I,H]` | 70.9 | 17.1 |
| mlp fc1 `[S,H]×[H,2F]` | 279.9 | 17.3 |
| mlp fc2 `[S,F]×[F,H]` | 155.5 | 15.6 |
| attention `S=15731, 56×128` | 439.0 | 16.2 |
| **square 8192³, clean best case** | 62.5 | **17.6** |

**The model achieves 16.0 TFLOP/s.** The square GEMM's 17.6 is *not* the
reference it should be compared against, and an earlier version of this section
claimed it was — reporting the model as running at "91% of the fastest rate this
library reaches". That comparison was invalid: every row in the table above uses
a contiguous `[K,N]` right operand, while the checkpoint stores `[N,K]` and
every production projection calls `matmul(x, weight.T)`. The model was being
measured against a layout it does not run.

At the layout it does run (`weightLayout`, same interleaved-median protocol):

| kernel | TFLOP | `[N,K].T`, as production runs it | `[K,N]` |
|---|---:|---:|---:|
| qkv | 3.637 | **16.4** TFLOP/s | 17.1 |
| attention out | 1.212 | **16.2** | 16.8 |
| mlp fc1 | 4.850 | **16.4** | 17.0 |
| mlp fc2 | 2.425 | **14.3** | 15.7 |
| attention | 7.095 | **16.2** | n/a — not a weight matmul |

**Aggregate isolated rate in the model's own layout: 16.0 TFLOP/s. The model
achieves 16.0 TFLOP/s.** It is not at 91% of what MLX can do; it is at what MLX
can do, and the 9% gap was the layout penalty misattributed to overhead.

The kernel accounting moves the same way:

```
GEMM per block, model's layout   761.2 ms      (materialised [K,N]: 723.8 ms)
+ attention                      439.0 ms
                               = 1200.2 ms  x 50 blocks = 60.0 s
real forward, from the controls              = 60.0 s
```

The earlier figure — 1154.5 ms per block, 96.2% — used the `[K,N]` timings, so
**1.87 s of its 2.3 s "non-kernel envelope" was the weight-layout penalty**, not
overhead.

**Do not read the 100.0% as exact.** Non-kernel work in the block is real and
has been measured directly: fused modulation 0.94%, the AdaLN projection 0.37%,
block compilation 0.54%. Those cannot all fit inside a residual of zero, so the
honest statement is that this accounting is good to **±2%**, and within that
band there is no overhead left to reclaim. Two independent routes — summing
isolated kernels, and measuring each non-kernel candidate on its own — agree
that essentially all of a forward is the five kernels.

So items 3, 4 and 5 are all **struck**:

| item | why it is dead |
|---|---|
| 3. AdaLN schedule batching | **measured 0.37%** of a full step, not estimated |
| 4. Dense-block compilation | **0.54%** for one block and **1.15%** for two, interleaved medians |
| 5. Weight layout, GEMM profiling | exact and real, but **3.0–3.1% of a full step**, below the 5% gate — and it is the whole of the gap once thought to be overhead |

### Items 3 and 4 were struck on an estimate and are now struck on a measurement

The first pass costed item 4 as kernel-launch overhead alone — about a thousand
dispatches at ~30 µs against a 60 s forward, so 0.05% — and that was the wrong
instrument. Compilation also removes intermediate materialisation and gives the
optimiser visibility across the attention and MLP boundaries, and a launch count
sees neither. Measured at the unit that matters, full block latency at
production width (`CompiledBlockTests`, `H3_BIG=1`):

| configuration | per block | vs plain | of a full step |
|---|---:|---:|---:|
| plain, one block | 1258.8 ms | — | — |
| **compiled, runtime `tEmb`, one block** | **1252.3 ms** | **1.005×** | **0.54%** |
| plain, two blocks | 1278.2 ms/block | — | — |
| **compiled, runtime `tEmb`, two blocks** | **1264.4 ms/block** | **1.011×** | **1.15%** |

Both compiled arms pass `tEmb` as a real input shared by every block, exactly as
the production stack does. Measurements alternate ABBA/BAAB and report the
median of ten samples per arm; this replaced the plain-then-compiled timer after
that timer produced a 6.98% ordering outlier. Widening from one to two blocks
adds 0.61 percentage points, inside the 1.4% control noise floor. There is no
measured cross-boundary opportunity worth compiling a larger stack for.

Compilation is also **not bit-identical** — relative RMS is 2.35e-4 for one
block and 3.28e-4 for two, compounding over fifty blocks and twenty steps into a
different render. At 0.54–1.15% that is not a trade worth making.

Two useful things fell out of it. The synthetic block at production width
measures 1258.8 ms, and 50 × that is 62.9 s against the 60.0 s forward measured
in the controls — **a third independent route to the kernel accounting**, from
a real block rather than from summing isolated kernels. It lands 4.8% high,
which is the honest width of this whole accounting: three routes agree that
essentially all of a forward is the five kernels, and none of them resolves the
remainder to better than a couple of percent. And the 1.8% gap between the two compiled
rows prompted a direct measurement of the AdaLN projection, which is **4.4 ms
per block with the fp32 upcast on, 0.37% of a step** — so item 3 is worth about
a third of one percent, not the 0.008% its FLOP count suggested and not the 1.8%
the gap suggested. FLOPs were the wrong unit; so was the gap.

### Canonical weight layout — real, exact, below the gate

The first GEMM ceiling used an already contiguous `[K,N]` right operand, while
the checkpoint and every production DiT projection store `[N,K]` and call
`matmul(x, weight.T)`. `weightLayout` measures those two physical layouts on
identical values, with the same interleaved-median protocol:

| projection | checkpoint `[N,K].T` | materialised `[K,N]` | speedup |
|---|---:|---:|---:|
| QKV | 221.3 ms | 212.5 ms | 1.042× |
| attention out | 74.9 ms | 72.3 ms | 1.036× |
| MLP fc1 | 295.5 ms | 284.8 ms | 1.038× |
| MLP fc2 | 169.5 ms | 154.2 ms | 1.099× |

Every output is bit-identical (`rel_rms = 0`). Two complete runs put the saving
at 36.1–37.4 ms per block, **4.9–5.0% of GEMM time**. Attention is unchanged,
so that becomes 1.81–1.87 s across fifty blocks, **3.0–3.1% of a 60.0 s full
step** and about 20 seconds over the eleven full steps in a cached twenty-step
render. It is a real optimization, but it fails the roadmap's 5%
production-step gate. The
loader and memory-lifecycle complexity of replacing 66 GB of resident weights
is not justified for a sub-gate result, so production weights stay
checkpoint-native.

**Fewer bits does not work here either — measured.** MLX's quantised matmul at
production shapes:

| shape | bf16 | int8 | int4 |
|---|---:|---:|---:|
| qkv `[S,H]×[H,3I]` | 223.2 ms | 227.4 ms (0.98×) | 227.3 ms (0.98×) |
| mlp fc1 `[S,H]×[H,2F]` | 294.9 ms | 299.8 ms (0.98×) | 289.9 ms (1.02×) |
| mlp fc2 `[S,F]×[F,H]` | 162.7 ms | 147.9 ms (1.10×) | 151.9 ms (1.07×) |

About 1.02× across the GEMMs, which are 63% of a forward: **1.2% overall,
below the 1.4% noise floor**, and two of three shapes are slower. int4 is no
better, so there is no going further down.

**And this was predictable from a figure already in this document.**
Quantisation pays when the work is bandwidth-bound on weights — batch-1 decode,
where every weight is read once per token. At S = 15,731 each weight read is
amortised over 15,731 rows, and reading all 66.3 GB once per forward is 83 ms,
0.1% of a step. Halving the bytes saves 0.05%. The arithmetic would have to get
faster instead, and MLX's path evidently dequantises to compute.

Worth separating from the checkpoints: the int8 files this tree already ships
are a **disk format, not a compute format** — `h3 doctor` reports them as
"dequantised at load — saves disk, not memory". They run bf16 matmuls and are
unaffected by any of the above.

**So the only remaining lever is fewer FLOPs**: sparse attention (36.9% of the
forward — Sol-Attn measured and rejected on quality, axial measured and rejected
in 6D) or fewer steps (the cache, shipped at 1.79× plus cap 5's 8–24%).

### The roadmap is closed

Every lever has now been measured rather than estimated:

| lever | result |
|---|---|
| cross-step cache | **shipped**, 1.79× |
| consecutive cap 3 → 5 | **shipped**, 8.4–9.5% at 20 steps, 24.5% at 40 |
| fused modulation | 0.94% — off by default |
| AdaLN schedule batching | 0.37% — struck |
| dense-block compilation | 0.54–1.15% — struck |
| weight layout / GEMM tuning | exact 3.0–3.1% full-step gain, below gate — struck |
| GEMM efficiency in the model's own layout | model is *at* MLX's isolated rate, not 91% of it — nothing to reclaim |
| int8 / int4 matmul | 1.2% — struck |
| Sol-Attn sparse attention | rejected on quality |
| axial-union topology | rejected on accuracy in 6D, no kernel written |

What is left is not a tuning knob. It would be a hand-written Metal GEMM that
beats MLX's, with no evidence yet that the hardware has headroom MLX is leaving;
or a different cache design that refreshes the residual partially rather than
all-or-nothing. Both are projects, not experiments, and neither should start
without first establishing what the hardware can actually do.

A third possibility exists and should be named rather than assumed away: a
hand-written Metal GEMM that beats MLX's. That is a much larger undertaking
than "canonical weight layout", with no evidence yet that the hardware has
headroom MLX is leaving on the table, and it should not be started without
first establishing what the hardware can actually do.

---

## 6C — Consecutive-cap sweep *(ACCEPTED — cap 5 is the default from 2026-08-06)*

The cache has three levers: threshold, per-stream probe, consecutive cap. The
6A trace showed the threshold does not control the middle of the schedule — from
step 4 to 15 the deltas peak at 0.096 against a threshold of 0.10, so every
refusal there is `consecutiveCap`. That made the cap the untested lever and this
phase is about it.

Residual semantics are unchanged throughout. There is one cached residual for
the whole packed stack, so **audio and video cannot skip independently**; what
can be independent is the admission test, and already is — reuse requires both
`whole < threshold` and `audio < threshold`.

### Status

| question | answer |
|---|---|
| Does cap 5 go faster? | **Yes — 8.4–9.5% at 20 steps, 24.5% at 40**, on every probe and seed |
| Does cap 5 cost quality? | **Not visibly**, across fast motion, close-up dialogue, 40 steps and a second seed |
| Do the automated metrics settle it? | **No.** The reason is structural, and it is not going to be fixed — see below |
| Shipped? | **Yes.** `cacheMaxSkips` default 3 → 5 |

**Accepted on viewing.** That is the honest description and not a lapse: the
measurement section below establishes that no automated comparison available
here can resolve an effect of this size, and the one metric that wrongly
condemned cap 5 was an artefact. The clips were watched at every stress axis
and cap 5 was clear at all of them.

### The retraction

**This section previously rejected every cap arm, reporting that they lost more
detail than they gained in speed. That was wrong twice over.**

*First*, `Tools/coherence_check.py` downsampled every frame to 216×120 and
computed "detail" on that. An area-average is a low-pass filter, so the figure
read mid-frequency structure with the fine detail already removed. Natively the
numbers reverse:

| arm | detail, downsampled | detail, native |
|---|---|---|
| cap-4 | 0.978 | **1.084** |
| cap-5 | 0.884 | **1.037** |
| cap-6 | 0.846 | **1.057** |

"Cap 5 costs 11.6% of detail" is +3.7% with the filter off.

*Second*, and not fixed by resolution: **every cap setting changes the
trajectory, so every arm renders a different scene.** A detail ratio between two
clips only means something while their compositions match, and past dssim ≈ 0.01
they do not. The proof this is real rather than pedantic — on the speaker probe
the **dense** arm, no cache and no approximation, scored *lower* detail than a
cached one at dssim 0.108. The metric was reading which market stall was in shot.

`coherence_check.py` now measures detail natively, keeps the downsampling for
the temporal measures that need it, and prints dssim first with every ratio
marked void above 0.01. A number with a footnote gets quoted without the
footnote.

### What is measured and holds

```
arm      reuses   end-to-end vs cap 3      on the speaker probe
cap-3         9   1.000x                   1.000x
cap-4         9   1.016x  (inside noise)   —
cap-5        10   1.098x                   1.065x
cap-6        10   1.100x                   —
```

**Audio is unaffected** across every cap: spectral correlation 0.993–0.997,
envelope above 0.99. That measure needs no matched scene, so it is not subject
to the confound above.

**The delta curve is schedule-driven, not content-driven.** The speaker probe
reused 9 steps at cap 3 with a median whole-sequence delta of 0.080, against the
beach's 0.082, on completely different content. This is why the speed results
generalise even though the quality ones do not — and it retires an earlier worry
that 45% reuse and 1.79× were easy-content artefacts.

### The replay model was exact

Every rendered arm reproduced its projection, step for step:

| cap | projected reuses | actual | projected refreshes | actual |
|---|---|---|---|---|
| 3 | 9 | 9 | 7, 11, 15 | 7, 11, 15 |
| 4 | 9 | 9 | 8, 13 | 8, 13 |
| 5 | 10 | 10 | 9, 15 | 9, 15 |
| 6 | 10 | 10 | 10 | 10 |

So `StepCachePolicyReplayTests` can *screen* future policy proposals — worth
about an hour of GPU per rejected idea. It says nothing about quality.

Two things it established that a naive reading got wrong. **Raising the cap does
not remove refusals, it moves them**: the counter resets at every refusal, so
caps 3 and 4 do identical work with refreshes merely relocated, and the first
estimate of "three capped steps removed, ~26%" was an upper bound that does not
exist. And **unbounded is not unbounded** — with no cap the cache reuses steps
4–14 and is then stopped by the audio probe at step 15. Once the cap is relaxed,
the per-stream audio probe is the only thing between the cache and the late
high-change region, which is invisible at cap 3 because the cap fires first.

### Cap 4 remains the interesting arm

It does the **same amount of work** as cap 3 — nine reuses, refreshes relocated
from 7/11/15 to 8/13 — so any difference between them is residual *age* alone,
with step count held constant. Its dssim of 0.0096 is the only one in the set
under the validity line, and natively it shows no detail loss.

### A prior from the threshold axis

`docs/ACCELERATION.md` has a threshold sweep with a zero-skip negative control
(the cache live but never firing returned bit-identical output) and a monotonic
dose-response: 0%, −16%, −28%, −44% of detail as skipping rises 0/10/13/14 of 20.
That is better controlled than anything measured here, and its shipped claim of
**16% at the default threshold stands.**

It also gives the cap axis a prior. If detail cost tracks the number of skipped
steps, going from 9 skips to 10 — all cap 5 does — should cost about a tenth of
16%, so roughly 1.6%. Consistent with cap 5 clearing a viewing after the metrics
had wrongly condemned it.

### What would actually settle it

Not another single-clip comparison. Every arm renders a different scene, so:

1. **Distribution over seeds.** `Tools/arm_compare.py` measures each clip on its
   own and asks whether the arms differ by more than the seeds within an arm do.
   It never takes a ratio between two clips. With a handful of seeds it prints
   the within-arm spread as the noise floor and labels a smaller gap "not a
   finding" rather than leaving that to the reader.
2. **Reference-free absolute measures.** Speech WER against the known prompt
   line, lip-sync margin, face and landmark stability. None need a matched
   scene. Requires `openai-whisper`, not installed.
3. **Blinded human A/B.** The only measure that has caught anything here first:
   the Sol-Attn pulsing artefact was found by a viewer after every tensor metric
   passed it, and cap 5 was cleared by eye after the metrics wrongly condemned it.

### Stress set

Cap 5 passed a first viewing. Before acceptance, four axes chosen for where a
*residual* cache should break — which is about how fast the trajectory moves,
not how detailed the frame is:

| arm | what it stresses |
|---|---|
| `moto` | fast motion, tracking camera, background whipping past. Every pixel changes every frame, so a reused residual has the least chance of still being right |
| `talk` | close-up dialogue. The documented failure for this model — every published H3 cache degrades audio — and the speaker probe only had a voice at market distance |
| `moto40` | 40 steps instead of 20 |
| `seed 42` | same configuration, different draw: separates a cap effect from a lucky seed |

**The 40-step axis is subtler than it looks**, and "more steps lets the cache
coast further" is only half right. A finer schedule moves the latent less per
step, so *n* steps of residual age covers *less* trajectory — a fixed cap is
**more** conservative at 40 steps. But smaller deltas also fall under the
threshold more often, so the cap binds more of the time and the skipped
*fraction* rises. Which dominates is a measurement; the replay model answers it
from the 40-step deltas once that arm lands.

### Promotion gate

**At least 5% end-to-end over the cached control** — not over dense. Beating
dense is not an achievement here; the cache already does it. Cap 5 clears this.

Nothing is promoted from one prompt and one seed. A qualifying arm needs multiple
seeds, the stress axes above, and: localized temporal acceleration, Laplacian
detail measured natively, speech WER, audio spectral and envelope correlation,
lip-sync margin, face and landmark stability, and blinded playback. Faithful
mode stays completely cache-free.

Reject immediately on visible pulsing, geometry drift or warping — the failures
that every tensor metric passed during the Sol-Attn work.

### Not run

Unbounded reuse. It remains a cliff-finding experiment, and the audio probe
rather than the cap is what would stop it.

### If the cap moves, the sigma work does not follow

The original plan for this phase was a sigma-aware threshold curve. **Drop it
unless a later trace shows a threshold-controlled region worth fitting.** The
measured deltas are U-shaped in step index, not monotone in sigma, so a sigma
curve would be fitted backwards; and the cache already finds the useful middle
window without one. Splitting whole-sequence from video is likewise not worth
promoting: video runs ~3% above whole-sequence and disagrees about a decision
exactly once in twenty steps, at step 4, in the direction of doing *more* work.

---

## 6D — Axial-union oracle *(measured; STOP — do not build 6E)*

**The stop condition fires. The deterministic topology is less accurate than
the content-adaptive routing it was meant to replace, at comparable density.**

Measured on q/k/v captured from a real render at blocks 0, 24 and 49, 25%
through the schedule, S = 15,993:

```
             density   video rel-RMS by block
landmarks              block 0   block 24   block 49
   2          0.194     0.5574     0.4803     0.2865
   3          0.217     0.5023     0.4502     0.2739
   5          0.263     0.4328     0.3934     0.2329
   9          0.355     0.3088     0.3033     0.1902

prefix rel-RMS: 0.0000 at every density and every block
```

**Sol-Attn measured 0.29 and was rejected on quality** — it produced visible
pulsing that no tensor metric caught. Axial at a useful density (3 landmarks,
21.7%) sits at 0.45–0.50 on the video rows: roughly 1.6× worse than something
already known to be unshippable. It only reaches Sol-Attn's 0.29 at 9
landmarks and 35.5% density, where there is little sparsity left to spend.

In hindsight the result is not surprising. Content-adaptive routing picks the
blocks that carry the mass; a fixed topology picks blocks by position and
misses whatever the content put somewhere else. Determinism was bought, and
this is the price.

### Two things the design got right, and they are worth keeping

**Prefix error is exactly zero**, at every density and every block. Prefix
queries attend everything and every query keeps the whole prefix, so the text,
the references and the target audio are bit-exact rather than approximately
right. That is the failure every published cache and sparse method for this
model shares, and this topology does not have it. Any future sparse work should
inherit that structure whatever else it does.

**The error does not move between steps.** Consecutive captures at schedule
progress 0.25 and 0.30 give 0.4438 and 0.4469 — a 0.7% swing. The design's
central claim holds: the same query attends the same keys at every step, so
the operator cannot change discontinuously mid-trajectory. That is precisely
the mechanism behind Sol-Attn's pulsing, and it is absent here. The accuracy is
stably bad rather than unstably mediocre, which is the better failure to have —
it means the topology is a fair test of the *idea* and not of an artefact.

### The speed case, for completeness

Attention is 36.9% of a forward. At 21.7% density the ideal is 1.4× on the
whole forward; the Sol-Attn kernel achieved roughly half its ideal, which would
put this near 1.2×. That is the upside, against a video error 1.6× worse than
one already rejected.

### What was built and what it is worth keeping for

`AxialTopology` (integer arithmetic, no MLX, tested without a GPU) and
`AxialReference` (two implementations, both one softmax over the union). They
stay as the oracle any future topology proposal is measured against, and the
tests encode the two traps:

* **Sliced SDPA cannot be combined.** Separate attentions over the frame, the
  column and the prefix each carry their own denominator, and summing them is
  not attention over the union. `fullSelectionIsDense` catches it — such an
  implementation cannot reproduce dense even when the union is everything.
* **The online-softmax recurrence is silent when wrong.**
  `attendStreamingKeys` is the arithmetic a kernel would implement, validated
  against the masked definition across key tiles that do not divide the
  sequence, since the ragged final tile is where it breaks.

Also here: the density figure is counted by frame rather than estimated by
inclusion-exclusion. The first version added the three axes and subtracted an
estimate of their overlap, and came out 4.2% wrong against a counted mask.

---

## 6E — Correctness-first Metal topology backend *(NOT STARTED — gated on 6D, which said stop)*

One backend behind the existing `H3AttentionBackend` seam. Exact prefix
handling, one softmax normalisation, dense fallback for unsupported geometry.
No `[S,S]` score matrix and no giant additive mask. Fixed named topology
profiles, no arbitrary public knobs. Opt-in only, never selected by `auto`.
Approximation and resolved topology recorded in events and receipts.

Correctness first. SIMD tuning begins only after a full production render passes
quality review.

---

## 6F — Production decision gate

Evaluate at minimum: text-to-video with dialogue; talking-head/lip-sync;
reference-conditioned video; multiple seeds; five- and ten-second durations;
against both 6A controls.

Ships only if it produces a material incremental wall-clock gain **with the
cache enabled**, stays inside control-adjusted audio, detail, temporal, face and
lip-sync thresholds, survives blinded human A/B review, does not expand peak
memory beyond the declared machine profile, and has a stable equivalence class.

**If near-dense quality leaves less than roughly 10% incremental end-to-end
improvement, stop.** The Sol-Attn evidence already shows 2–6% does not justify
the artifact and maintenance risk.

---

## Quality tools

`Tools/FaceCheck.swift`, `Tools/LipSyncCheck.swift`, `Tools/speech_check.py`,
`Tools/audio_match.py`, plus temporal-acceleration and Laplacian-detail
measurements.

**PSNR may be recorded and is not a gate.** Generated media shifts spatially
between configurations by amounts a viewer would not notice and PSNR punishes
severely; ranking on it selects for blur. That is not hypothetical here — the
overnight sweep would have crowned its blurriest clip on temporal smoothness
alone, at 0.66× dense motion and 0.61× dense detail, until a sharpness guard was
added.

---

## Sol-Attn disposition

Frozen, not deleted. It stays as an opt-in negative control, as a Metal-kernel
and routing testbed, and as the evidence for why per-call relative RMS is
insufficient — it sat at 0.29 against dense while producing a temporal pulsing
artifact that no tensor metric caught and a viewer found in seconds.

Behind dense in the registry. No productization work. If it starts affecting the
production graph, it moves to a research target.
