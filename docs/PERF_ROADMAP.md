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

Separately evaluated below: fused Q/K RMSNorm plus split-half RoPE. The oracle
already measured the unfused attention path as bit-identical to the reference's
fused kernel, so this is a pure traffic argument with no correctness upside.

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
| 3. AdaLN schedule batching | initially estimated ~0.05% | **measure the complete six-table schedule; FLOPs alone miss the small-M projection cost.** |
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
has been measured directly: fused modulation 0.94%, AdaLN schedule batching
0.8–1.3% at render level, block compilation 0.54%. Those cannot all fit inside
a residual of zero, so the honest statement is that this accounting is good to
**±2%**, and within that band there is no overhead left to reclaim. Two
independent routes — summing isolated kernels, and measuring each non-kernel
candidate on its own — agree that essentially all of a forward is the five
kernels.

So items 3, 4 and 5 are all **struck**:

| item | why it is dead |
|---|---|
| 3. AdaLN schedule batching | **0.8–1.3% render-level saving**, 360–729 MiB, mode-dependent drift |
| 4. Dense-block compilation | **0.54%** for one block and **1.15%** for two, interleaved medians |
| 5. Weight layout, GEMM profiling | exact and real, but **3.0–3.1% of a full step**, below the 5% gate — and it is the whole of the gap once thought to be overhead |

### AdaLN schedule batching — fast locally, irrelevant to the render

The single-output projection probe was the wrong final unit: a block consumes
all six modulation tables, and the schedule and cache decide how often the
projection actually runs. `adalnScheduleBatching` therefore compares both
supported schedule shapes with all six outputs evaluated: text-to-video/audio
has one deduplicated row at the first step and two thereafter; paired visual and
audio references have three, then four.

| measurement | t2va, across four runs | paired ref2va |
|---|---:|---:|
| twenty separate projections, per block | 267.9–300.9 ms | 338.1 ms |
| one schedule projection, per block | 22.5–36.7 ms | 30.1 ms |
| local projection speedup | 7.7–12.5× | 11.25× |
| dense twenty-step saving | 11.65–13.21 s (**1.0–1.1%**) | 15.40 s (**~1.3%**) |
| cap-5 saving | 5.09–6.04 s (**0.8–0.9%**) | 7.12 s (**~1.1%**) |
| persistent modulation tables | 360 MiB | 729 MiB |
| numerical result | rel_rms 6.58e-6, not identical | bit-identical in measured arm |

The cap-aware estimate includes block 0 on all twenty steps and blocks 1–49 on
the ten full steps. Precomputing cannot know those cache decisions, so it must
materialise every block's schedule, including the ten steps where those 49
blocks will be skipped. GEMM accumulation is mode/shape dependent: t2va moved
by 6.58e-6 while the larger reference arm happened to remain bit-identical.
The optimization therefore cannot claim a universal exact equivalence class.
A 0.8–1.3% gain does not justify 360–729 MiB of new lifecycle state or a render
quality gate. Item 3 is closed without a production implementation.

### Fused Q/K RMSNorm plus RoPE — 4.2× locally, 1.2% on the render

The first ceiling fixture allocated independent contiguous Q and K tensors and
reported 132.4 ms per block, apparently enough to justify a kernel. That was the
wrong boundary. Production receives one contiguous QKV projection and Q/K are
split/reshape views into it; MLX optimises that graph very differently. The
correct fixture begins at the real QKV output and measures the exact path the
block runs.

A correctness-first Metal fixture consumes QKV directly, performs both fp32
per-head reductions, rounds the normalized values to bf16 at the readable
path's boundary, then applies the 96-channel split-half rotation:

| production shape `S=15731, H=56, D=128` | result across four runs |
|---|---:|
| Q/K RMSNorm only | 13.4–14.0 ms |
| Q/K RoPE only | 7.2–7.4 ms |
| complete unfused chain | 19.5–20.1 ms |
| fused Metal fixture | 4.6–4.8 ms (**4.19–4.22×**) |
| measured full-step saving | 0.74–0.77 s (**~1.2–1.3%**) |
| measured cap-5 render saving | 7.57–7.83 s of 660.5 s (**~1.2%**) |
| numerical result | rel_rms 9.62e-6; worst 1.08 bf16 ulp; not bit-identical |

Even an impossible free kernel is bounded at 1.62–1.68% of a full step. The
real kernel is fast and its error is confined to reduction-order noise, but it
still changes the trajectory and fails the 5% gate by fourfold. The Metal code
therefore remains a test-only fixture: no production branch, environment flag,
receipt field or dormant alternative math path was added.

### Dense-block compilation — measured and struck

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

One useful thing fell out of it. The synthetic block at production width
measures 1258.8 ms, and 50 × that is 62.9 s against the 60.0 s forward measured
in the controls — **a third independent route to the kernel accounting**, from
a real block rather than from summing isolated kernels. It lands 4.8% high,
which is the honest width of this whole accounting: three routes agree that
essentially all of a forward is the five kernels, and none of them resolves the
remainder to better than a couple of percent. The compiler experiment also
prompted the schedule-level AdaLN measurement above; that result supersedes the
earlier 4.4 ms single-table projection probe.

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

### Prefix-refresh cache — clears speed, destroys dialogue

The last cache proposal relaxed the consecutive cap while recomputing a small
current prefix on every reuse. Instead of applying the previous step's entire
50-block residual, prefix 2 ran blocks 0 and 1 on the current latent and reused
only the cached residual of blocks 2–49. The threshold and per-stream audio veto
were unchanged. Prefixes 2 and 3 were the only candidates to clear an offline
5% block-work screen; prefix 2 ran first because it has the higher ceiling.

A contemporaneous cap-5 control was necessary: the older control's full steps
were 2.2% faster than the experiment's, enough to reverse a 5% decision. With
the same build, prompt, seed and machine, the result was:

| speaker arm | reuses | full-step median | reuse median | sampling | total |
|---|---:|---:|---:|---:|---:|
| cap-5 control | 10/20 | 61.47 s | 1.24 s | 626.22 s | 707.49 s |
| prefix-2, cap 99 | 11/20 | 61.03 s | 2.50 s | 584.33 s | 678.27 s |

That is a real **7.2% sampling gain**, but only **4.3% to a finished file** on
this run. More importantly, it fails the quality gate decisively. Whisper found
the requested sentence in the control with 5/5 content keywords and 100% recall.
Prefix 2 produced repetitive Welsh-like output, 0/5 keywords, 0% recall, and
failed the repetition guard. Face and landmarks remained present in 124/124
frames, so this is not a collapsed render hiding behind an absent subject; it is
the cross-stream coherence failure the per-stream probe was meant to prevent.
The probe can veto a reuse, but refreshing two visual/audio transformer blocks
does not make an eleven-step-old shared tail coherent.

Prefix 3 was stopped during prompt processing. It has a strictly lower speed
ceiling and no mechanism that addresses the stale tail that broke prefix 2.
The environment lever and production path were removed after the run. Records:
`docs/bench/speaker-cap5-current.h3-bench.json` and
`docs/bench/speaker-prefix2.h3-bench.json`.

**So the only remaining lever is fewer FLOPs**: sparse attention (36.9% of the
forward — Sol-Attn measured and rejected on quality, axial measured and rejected
in 6D) or fewer steps (the cache, shipped at 1.79× plus cap 5's 8–24%).

### The roadmap is closed

> **Closed for speed, reopened for memory.** Section 7 measures block
> streaming, which costs 1.9–4.9% and returns 64 GB of residency. Nothing below
> is retracted — every speed lever here stays measured and struck — but "what is
> left is not a tuning knob" was written with only speed in view.

Every lever has now been measured rather than estimated:

| lever | result |
|---|---|
| cross-step cache | **shipped**, 1.79× |
| consecutive cap 3 → 5 | **shipped**, 8.4–9.5% at 20 steps, 24.5% at 40 |
| fused modulation | 0.94% — off by default |
| AdaLN schedule batching | 0.8–1.3%, 360–729 MiB and mode-dependent drift — struck |
| fused Q/K RMSNorm plus RoPE | 4.2× locally, ~1.2% render-level and 1.08 ulp — test-only, struck |
| dense-block compilation | 0.54–1.15% — struck |
| weight layout / GEMM tuning | exact 3.0–3.1% full-step gain, below gate — struck |
| GEMM efficiency in the model's own layout | model is *at* MLX's isolated rate, not 91% of it — nothing to reclaim |
| int8 / int4 matmul | 1.2% — struck |
| prefix-2 partial refresh | 7.2% sampling, 4.3% end-to-end; dialogue 5/5 → 0/5 keywords — rejected |
| Sol-Attn sparse attention | rejected on quality |
| axial-union topology | rejected on accuracy in 6D, no kernel written |

What is left is not a tuning knob. It would be a hand-written Metal GEMM that
beats MLX's, with no evidence yet that the hardware has headroom MLX is leaving;
or a fundamentally different cache whose reused state is stream-aware rather
than one shared stale tail. The bounded prefix-refresh version has now been
measured and rejected. A hand-written GEMM or a redesigned cache are projects,
not experiments, and neither should start without first establishing what the
hardware can actually do.

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

---

## Disposition, 2026-08-06

**Every lever put in for a sweep has been backed out and its answer hard-coded.**
A knob that exists to settle a question is scaffolding once the question is
settled, and leaving it costs more than the code: `fast` was a shipped profile
whose quality drifted the moment `cacheMaxSkips` moved underneath it, because
nobody was watching an option nobody used.

| removed | answer now hard-coded |
|---|---|
| `--cache-threshold` | 0.10 via `balanced`, 0 via `faithful` |
| `--cache-max-skips` | `RenderRequest.cacheMaxSkips = 5` |
| `--cache-whole-sequence-probe` | per-stream, always |
| `--quality fast`, `--quality custom` | gone; `faithful` or `balanced` |
| `H3_SOL_*` (7 variables) | Sol-Attn removed |
| `H3_FUSED_MODULATION` | fused modulation removed |
| `H3_CACHE_PREFIX_REFRESH` | prefix refresh rejected; production path removed |
| `--attention-backend sol` | registry is SDPA only |

Deleted with them: `SolAttn{Backend,Metal,Reference,Routing}.swift` and five
test files, `FusedModulation.swift` and its tests — about 2,400 lines whose only
remaining purpose was to be reachable by a flag.

**Kept, and why.** `H3_CAPTURE_*` is the only way to get real q/k/v out of a
render, and every attention measurement in this document depended on it.
`H3_BENCH_ARM` and the benchmark records are 6A's deliverable, not a lever. The
`H3AttentionBackend` seam stays with SDPA as its only implementation: it is what
let two sparse methods be measured without touching the production path, and
re-adding it later is more work than leaving it. `AxialTopology` and
`AxialReference` stay as test-only infrastructure — they are the oracle any
future topology proposal is measured against, and they have no lever, no
registry entry and no production surface.

`docs/SOL_ATTN.md` is kept with a removal notice on top: the code is gone, the
evidence for why is not. The scripts under `docs/bench/` are marked archived —
they record what produced the JSON beside them and will not run today, because
the flags they pass no longer exist.

---

## 7 — Block streaming *(MEASURED 2026-08-11; the roadmap reopens on memory)*

The roadmap above closed because every remaining lever was a speed lever and
each had been measured to nothing. This one is not a speed lever. It costs
speed and buys residency, and `docs/PERFORMANCE_GUIDE.md` already named the
category: *"Memory work remains useful even when it is not a speedup ... those
changes solve capacity, not the 95%-DiT sampling bottleneck, and should be
evaluated under a different success criterion."*

The prompt came from [`antirez/h3.c`](https://github.com/antirez/h3.c), an
independent C and Metal implementation of this model. It keeps two DiT blocks
resident and reads the rest from disk during GPU execution: 36.5 GiB of
residency becomes 2.0 GiB, at **26–84% slower**.

**That penalty is a property of their shape, not of streaming.** They measure
512x512x22, where a block is about 37 ms of compute. The control shape here
leaves 1.289 s. Same bytes, 35x the time to hide them behind. The arithmetic
said this tree sits on the opposite side of the trade, so it was measured before
anything was built.

    H3_BIG=1 swift test --filter blockStreaming

### What the checkpoint looks like from a streamer's side

| | |
|---|---:|
| block span, whole | 1291.1 MB |
| of which `adaln_proj` | 520.4 MB (**40.3%**) |
| tail — attention and MLP | 770.7 MB |
| blocks | 50, every span contiguous |

`adaln_proj` sits at the **start** of every span, bias then weight. Skipping it
therefore leaves one contiguous read rather than a scatter, so a streamer issues
a single `pread` per block either way.

### Measured

External PCIe SSD, `F_NOCACHE` throughout — this machine has 256 GB of RAM and
the checkpoint is 66 GB, so a cached read reports memory bandwidth and the
experiment becomes a tautology.

| | |
|---|---:|
| uncached sequential read | **3.17 GB/s**, flat from 1 to 4 concurrent readers |
| whole span | 0.408 s/block |
| tail only | 0.244 s/block |
| block compute, control shape | 1.289 s |

Eight blocks, double-buffered, prefetch issued before the previous block's
forward:

| arm | bytes/block | s/block | overhead | stalled |
|---|---:|---:|---:|---:|
| resident | — | 1.289 | — | — |
| tail streamed | 770.7 MB | 1.313 | **+1.9%** | 0.11 s of 1.95 s of reads |
| whole span streamed | 1291.1 MB | 1.352 | **+4.9%** | 0.25 s of 3.26 s of reads |

Residency, sixteen consecutive streamed blocks with `Memory.cacheLimit` pinned
to two blocks so the allocator cannot flatter the curve:

    865 865 865 865 865 865 865 865 865 865 865 865 865 865 865 865   MB

Flat to the megabyte — **0.00 blocks' worth of growth**. 64.6 GB of block
weights held as 0.9 GB on the MLX side plus 1.5 GB of host staging.

And the output is **bit-identical** to the resident path. It is the same bytes
reinterpreted, so anything less would have meant the offset arithmetic or the
qkv permute was wrong — both of which fail silently, which is why that check
exists rather than a tolerance.

### What this changes

Sampling residency at 864x480x124 goes from ~66 GB of weights to ~2.4 GB. Peak
was **87.2 GB**, of which activations are ~21 GB and do not move. The sampling
stage lands near **25 GB**.

**The AdaLN precompute is an optimisation, not a precondition.** That is the
result that matters for sequencing, and it was the opposite of the expectation
going in. `H3Transformer` derives `tEmb` from `plan.values` alone, so every
block's modulation is a function of the timestep schedule and could be computed
once per render — which would drop 40.3% of the bytes. But whole-span streaming
already costs only 4.9%, so precomputing buys **3 percentage points**, not
feasibility. Item 3 above struck AdaLN *batching* on a 0.8–1.3% speed argument;
nothing here disturbs that. Caching the schedule is a different operation from
batching it — same shapes, same accumulation order, so bit-identical by
construction rather than the 6.58e-6 that batching moved.

### What is not established

- **Storage dependence is linear and this was one drive.** At 1.5 GB/s the
  whole-span read is 0.86 s against 1.289 s of compute — it still hides, with no
  margin. Below about 1 GB/s it does not hide at all, and the tail-only arm
  becomes the only viable one. A USB enclosure is not a PCIe slot.
- **Nothing was measured under memory pressure.** This ran on 256 GB where the
  staging buffers, the page cache and the activations never competed. The claim
  "this makes 48 GB Macs work" needs a 48 GB Mac or a constrained run; what is
  established is that the mechanism holds its footprint flat, not that the
  machine it is for behaves.
- **Eight blocks, extrapolated to 50**, with no thermal exposure. A 20-minute
  render sustaining 3.2 GB/s is a different question from 10 seconds of it.
- **SSD wear is real and should be stated to users, not discovered by them.**
  Whole-span streaming reads 64.6 GB per full forward; a cap-5 20-step render
  runs about 11 of them, so roughly **710 GB per render** (424 GB tail-only).
- **The cross-step cache interaction is unmeasured.** Cached steps skip blocks
  1–49 entirely and therefore issue no reads at all, so the direction is
  favourable, but the figure above assumes full steps throughout.
- **Decode is untouched.** Chunked video decode remains the other half of a
  small-machine story and this does nothing for it.

---

## 8 — MLX is not the hardware *(MEASURED 2026-08-11)*

> **Superseded in part by section 14.** The 11-22% gap is real and reproduced,
> but "17.3 TFLOP/s is MLX's ceiling" was wrong: MLX had the faster kernel
> compiled in all along and its routing heuristic did not select it. Read this
> section for the gap, section 14 for what it was.

Section 6 closed on this sentence: *"a hand-written Metal GEMM that beats MLX's
... should not be started without first establishing what the hardware can
actually do."* That measurement had never been taken. It costs a minute.

It does not need a kernel. A second **vendor** implementation of the same
arithmetic answers the question: Apple's own `MPSMatrixMultiplication`, same
shapes, same machine.

    H3_BIG=1 swift test --filter hardwareCeiling

| shape | MLX bf16 | MLX fp16 | MPS fp16 | MPS/MLX |
|---|---:|---:|---:|---:|
| qkv      `[S,H]x[H,3I]` | 17.0 | 17.1 | 19.2 | **1.13x** |
| attn out `[S,I]x[I,H]` | 17.0 | 16.8 | 19.0 | **1.13x** |
| mlp fc1  `[S,H]x[H,2F]` | 17.0 | 17.1 | 19.4 | **1.14x** |
| mlp fc2  `[S,F]x[F,H]` | 15.2 | 15.3 | 18.8 | **1.23x** |
| square 8192³ | 17.0 | 17.2 | 19.0 | 1.10x |

Two independent samples agree to ±0.01x, against a 1.4% noise floor. **MLX's
bf16 and fp16 rates are identical**, which is what makes the fp16 comparison
legitimate for a bf16 production path.

**So 17 TFLOP/s was MLX, not the machine.** `gemmCeiling` showed the model at
92% of MLX's own ceiling and concluded there was nothing left to reclaim. That
conclusion was right about MLX and wrong about the hardware — the ceiling it
measured was the floor of another vendor's kernel.

### What it would be worth

From `gemmCeiling`'s per-shape times, against the 1200 ms block:

| | now | at MPS rates |
|---|---:|---:|
| qkv | 212.5 ms | 188.1 ms |
| attn out | 72.6 ms | 64.2 ms |
| mlp fc1 | 291.3 ms | 255.5 ms |
| mlp fc2 | 157.5 ms | 128.0 ms |
| **four GEMMs** | **733.9 ms** | **635.8 ms** |

**98 ms of a 1200 ms block — 8.2% of a full step**, if it survived integration.
It does not; see below. The figure is kept because it is what the kernels do, and
because it was the first thing to clear the
5% gate since the consecutive-cap sweep. Attention is a further 446.2 ms (37.8%
of the block) and was not tested; MPSGraph exposes an SDPA primitive, so the
figure is a floor rather than a total.

### The bf16 gate — passed, and it is not a numerical change

`MPSMatrixMultiplication` exposes no bf16, so the table above is fp16 and rests
on an argument. `MPSGraph` removes the argument. Same production shapes, the
model's own precision, both implementations fed **the same bytes** so any
disagreement is the arithmetic rather than the inputs:

    H3_BIG=1 swift test --filter mpsGraphBF16

| shape | MLX bf16 | MPSGraph bf16 | ratio | rel RMS | control |
|---|---:|---:|---:|---:|---:|
| qkv | 17.1 | 19.3 | **1.13x** | 0.00e+00 | 5.17e-06 |
| attn out | 16.9 | 19.0 | **1.12x** | 0.00e+00 | 3.90e-06 |
| mlp fc1 | 16.9 | 19.3 | **1.14x** | 0.00e+00 | 4.87e-06 |
| mlp fc2 | 15.3 | 18.7 | **1.22x** | 0.00e+00 | 3.88e-06 |
| square 8192³ | 17.3 | 19.2 | 1.11x | 0.00e+00 | 5.76e-06 |

**Bit-identical at every shape.** A zero that size is a claim about the
instrument as much as the kernel, so the `control` column exists: flipping one
bf16 ulp in one element of one input moves the comparison to ~4e-6. It detects a
single-bit change and reports exactly zero here.

That is consistent with both implementations accumulating in fp32 along K in
order — tiling over M and N does not reorder a reduction, and only split-K
would. So **the accumulation-order objection is gone**: this is not an
approximation, needs no equivalence class, and the fp16 argument above is
retired rather than relied on.

### Integration — measured, and it closes the item

    H3_BIG=1 swift test --filter mpsIntegration

Both arms run the same modelled block — the real `H3RMSNorm`, `modScaleShift`,
`modGate`, `SplitHalfRoPE` and SDPA, in the real order — differing only in who
multiplies. The model is checked against the real `DiTBlock` first:
**bit-identical, and 1317 ms against 1321 ms**, so it is the block.

Weights are uploaded once as graph constants and activations cross `noCopy`,
because the question is whether the win *can* survive rather than whether a
naive port loses it.

| sample | all MLX | MPSGraph GEMMs | change |
|---|---:|---:|---:|
| 1 | 1299 ms | 1339 ms | +3.1% |
| 2 | 1291 ms | 1278 ms | −1.1% |
| 3 | 1295 ms | 1308 ms | +1.0% |
| 4 | 1291 ms | 1307 ms | +1.3% |
| 5 | 1212 ms | 1252 ms | +3.3% |

**Median +1.3%, four of five slower.** The predicted −8.2% does not appear. The
block output stays bit-identical, so nothing is wrong with the arithmetic — what
fails is the boundary.

### Where the 8.2% goes, and why it is the barrier rather than the bytes

    H3_BIG=1 swift test --filter crossingBreakdown

The same four projections measured standalone, at their real shapes:

| projection | MLX | MPS+copy | MPS only | copy | out |
|---|---:|---:|---:|---:|---:|
| qkv | 224.3 ms | 209.4 ms | 190.5 ms | 18.9 ms | 677 MB |
| attn out | 75.5 ms | 68.7 ms | 64.7 ms | 4.0 ms | 169 MB |
| mlp fc1 | 292.7 ms | 268.8 ms | 252.4 ms | 16.4 ms | 902 MB |
| mlp fc2 | 172.4 ms | 134.8 ms | 137.1 ms | −2.3 ms | 169 MB |
| **total** | **765 ms** | **682 ms** | **645 ms** | **37 ms** | |

**Standalone, MPS wins by 83 ms even paying the copy back.** The copy — the one
part a future MLX API could remove, since there is no public way to hand MLX an
existing `MTLBuffer` — is only 37 ms of it. So the data movement was never the
problem, and the obvious fix would not have fixed anything.

The problem is that 83 ms of standalone advantage becomes ~17 ms of block-level
*deficit*. Roughly **100 ms per block is MLX's own overlap being destroyed**:
`asMTLBuffer` calls `eval()`, so each of the four crossings drains a lazy graph
that MLX would otherwise have kept running underneath the norms, splits, gathers,
RoPE and the silu. MLX executes a block as one asynchronous stream; four
synchronous handoffs serialise it.

### Verdict

**Closed as a drop-in.** The kernel headroom is real — 1.11–1.22x, bit-identical,
and it survives the copy — but it does not survive the synchronisation, and the
synchronisation is not removable from outside MLX. Paying it back would mean
MPSGraph work encoded into MLX's *own* command stream so no drain is needed,
which is a change inside MLX rather than a call this tree can make.

What that leaves, stated precisely so it is not rediscovered:

- **The 17.3 TFLOP/s ceiling is MLX's, not the machine's** — that stands, and it
  is new. Anything that gets kernels into MLX's stream inherits ~11%.
- Attention, 37.8% of a block, was never tested. If the barrier problem is ever
  solved, that is the next thing to measure, not more GEMM work.
- Bit-identity is five shapes on one macOS build. It is a measurement, not a
  guarantee across either.

**Nothing in sections 1–6 is retracted.** Every lever there was measured against
MLX and every one of them is still struck against MLX. What changes is that
"MLX is the ceiling" was a statement about MLX.

---

## 9 — The barrier is per-crossing, not per-runtime *(MEASURED 2026-08-11)*

Section 8 closed the MPS idea on a block that crossed four times. That was the
worst arrangement available, and the verdict was drawn from it without asking
what a crossing costs.

The prompt to ask came from `StoryForge`, a sibling tree that solved the same
problem in the opposite direction. Two mechanisms there are worth naming:

- **`MPSGraphDenoiseStep` — the "Super Fusion Kernel".** CFG, STG, modality
  guidance, variance-preserving rescale and the Euler update, fused into *one*
  MPSGraph dispatch per step, writing directly into the caller's `MTLBuffer`
  with no host round trip. Its own bottleneck report attributes **66.9% of
  denoise time to the MLX materialisation bridge** — the same barrier measured
  here, found independently.
- **`Flux2FusedKernels` — `MLXFast.metalKernel`.** Custom Metal source that MLX
  JIT-compiles and dispatches **inside its own command stream**. There is no
  `eval()`, no handoff, and therefore no barrier at all. It is present in the
  MLX revision this tree already pins.

So the question was never "MLX or MPS". It is **how much work per crossing**.

### The slope

    H3_BIG=1 swift test --filter crossingCost

Routing one, two and four projections, everything else identical:

| arm | block | vs MLX | GEMM win forgone | implied barrier |
|---|---:|---:|---:|---:|
| 1 crossing (fc2) | 1208.3 ms | **−6.6** | −37.6 | 31.0 /crossing |
| 2 crossings (fc2+fc1) | 1217.3 ms | −3.4 | −61.5 | 29.1 /crossing |
| 4 crossings (all) | 1239.2 ms | +14.9 | −83.2 | 24.5 /crossing |

**A crossing costs about 28 ms.** The four GEMMs are worth 83 ms. Everything
follows from those two numbers, on a 1224 ms block:

| design | change |
|---|---:|
| 4 crossings — what §8 measured | +30 ms, **+2.4%** |
| 1 crossing per block | −55 ms, **−4.5%** |
| 1 crossing per step, barrier amortised over 50 blocks | −83 ms, **−6.7%** |
| 0 crossings — `MLXFast.metalKernel` | −83 ms, **−6.8%**, and attention becomes reachable |

Note the first row of the table: **routing fc2 alone is already a win today**,
−6.6 ms. It is the shape where MLX is weakest (15.3 against MPS's 18.7) and it
pays only one barrier. It is also far too small to ship, and it is quoted here
because it confirms the model rather than because anyone should do it.

### What this changes about §8

Nothing measured there is retracted — four crossings really does lose. What was
wrong was the conclusion drawn from it. "Closed as a drop-in" stands; "not
removable from outside MLX" does not, because `MLXFast.metalKernel` removes it
from inside MLX, using an API this tree already has.

### Which of the three is worth doing

**`MLXFast.metalKernel` is the only one worth starting**, and the reason is risk
rather than the extra 0.1%.

- It stays a *drop-in matmul*. The equivalence classes, taps and 36 contracts
  already in this tree validate it unchanged, because nothing about the model's
  structure moves.
- The two MPSGraph designs require re-expressing a block — attention, RoPE,
  per-head norms, the packed-sequence modulation — in a second framework. That
  is a second implementation of precisely the code `FRAGILE_CONTRACTS.md` exists
  because of: a wrong packed layout or a transposed qkv keeps every shape
  correct and every number wrong. −4.5% does not buy that.
- `StoryForge` fused its *sampler step* this way, not its transformer. That is
  the same judgement, made independently.

**What it requires is a Metal GEMM that reaches MPS's rate**, ~19.2 TFLOP/s
against MLX's 17.3. That is the hard part and it should not be understated: MLX's
steel kernels are tuned, and beating them by 11% by hand is a project. What is
new is that it is no longer speculative — §8 established the hardware does 19.2
on these exact shapes, so this is matching a demonstrated number rather than
hoping for headroom.

**Attention is the reason to care.** It is 446.2 ms of a 1224 ms block, 37.8%,
and it has never been measured against anything but MLX. A custom-kernel path
reaches it; a GEMM-routing path does not. If it carries the same ~11%, the two
together are roughly 11% of the whole forward — about **1.2 minutes off an
11.4-minute render** — and that is the number that would justify the work.

### Still true, and still the constraint

The model remains compute-bound: fusing to save memory traffic is worth very
little here. fc1 writes 902 MB, the silu-gate reads it and writes 451 MB, fc2
reads that — fusing all three saves ~1.35 GB of traffic, which at 800 GB/s is
under 2 ms of a 1224 ms block. This tree has measured that three times now
(fused modulation 0.94%, fused Q/K RMSNorm plus RoPE 1.2%, AdaLN batching
0.8–1.3%). **A custom kernel here is worth writing for its arithmetic rate and
for nothing else.**

---

## 10 — The custom GEMM prototype *(MEASURED 2026-08-11)*

> **Superseded by section 14.** The conclusion here — that collecting this
> needs a multi-week kernel project starting at 61% of MLX — was answered by a
> routing patch. No kernel should be written. The prototype and its tiling
> sweep are kept because they are what established that writing one was the
> wrong layer.

Section 9 recommended `MLXFast.metalKernel` as the only path worth starting, on
the argument that it has no crossing and therefore inherits §8's 11% for free.
The mechanism works exactly as described. The performance is the whole problem.

    H3_BIG=1 swift test --filter customGEMM

A shape-specialised bf16 GEMM at the fc1 shape — `[15731,5376] x [5376,28672]`,
`simdgroup_matrix<bfloat,8,8>` fragments, fp32 accumulators, threadgroup-tiled
over K. Six tilings, **every one bit-exact against MLX**:

| tiling | acc/simdgroup | FLOP/byte | ms | TFLOP/s | vs MLX |
|---|---:|---:|---:|---:|---:|
| 64x64x32, 2x2 sg | 16 | 32 | 611.1 | 7.9 | 0.44x |
| 128x128x32, 4x2 sg | **32** | 64 | 6694.6 | **0.7** | 0.04x |
| 128x128x32, 4x4 sg | 16 | 64 | 738.7 | 6.6 | 0.36x |
| 128x64x32, 4x2 sg | 16 | 43 | 536.4 | 9.0 | 0.50x |
| 128x128x16, 4x4 sg | 16 | 64 | 680.7 | 7.1 | 0.39x |
| **256x128x16, 8x4 sg** | 16 | 85 | **439.3** | **11.0** | **0.61x** |
| MLX | | | 268.4 | **18.1** | 1.00x |
| MPSGraph (§8) | | | | **19.3** | 1.07x |

**Register pressure, not tiling, was the trap.** Rows 2 and 3 are the same
128x128 tile and differ only in how many simdgroups share it: 32 accumulators a
simdgroup spills and runs at 0.7 TFLOP/s, 16 does not and runs at 6.6 — a 9x
swing from a change that looks like scheduling. The first two attempts here moved
tile size and register count together and so measured nothing; the sweep exists
because of that mistake.

With that separated, intensity behaves as predicted: 32 → 7.9, 43 → 9.0,
85 → 11.0 FLOP/byte.

### Steel-style follow-up — 16.2 TFLOP/s, still below break-even

The first recommended follow-up was implemented in `CustomGEMMTests`: the
M3-Ultra Steel geometry (`64x64x16`, `1x2` simdgroups), 16-byte-padded
threadgroup rows, one 32-byte vector load per thread, Steel's two-float register
fragment layout, register-resident A/B fragments, a direct fp32-to-bf16 store,
and the same four-way tile-grid swizzle. It remains an `MLXFast.metalKernel`, so
there is no framework crossing.

| implementation | TFLOP/s | vs original custom | vs MLX | exact |
|---|---:|---:|---:|---:|
| original best, `256x128x16` | 11.0 | 1.00x | 0.61x | yes |
| padded/vectorised/direct-store | 14.1 | 1.28x | 0.78x | yes |
| register-resident fragments | **16.3** | **1.48x** | **0.93x** | yes |
| plus Steel grid swizzle/unroll barriers | 16.2 | 1.47x | 0.93x | yes |
| MLX | 17.4–18.0 | | 1.00x | reference |
| MPSGraph target | 19.3 | | 1.07–1.11x | yes (§8) |

The loader and epilogue changes recovered most of the gap, and the grid swizzle
did not help this JIT kernel. More importantly, inspecting the bundled Steel
loop corrected the earlier diagnosis below: Steel itself is single-buffered, so
double buffering is not the missing prerequisite for reaching MLX. The
remaining 7% custom-to-MLX gap needs GPU-counter profiling or work inside MLX's
own build; it is no longer explained by an obvious omitted mechanism.

### What the original number meant

**The competent first cut reached 61% of MLX and 57% of the target.** It was
correct, ran in MLX's own command stream, and was 1.65x away from being
worth anything at all.

That last clause is the finding. The break-even is not zero — it is MLX's 18.1,
because a kernel that merely matches MLX delivers nothing. **The project is
"start 40% behind a tuned vendor kernel and finish 6% ahead of it."**

What the original 11-TFLOP/s prototype did not do:

- **No double buffering.** Global loads and compute are serialised by a
  threadgroup barrier. The follow-up established that bundled Steel does the
  same, so this is a possible experiment rather than the primary explanation.
- **Scalar global loads.** Elements move one bf16 at a time where the hardware
  wants vectorised loads.
- **A threadgroup round trip in the epilogue**, forced by fp32 accumulators
  writing a bf16 output.
- **No software pipelining across K steps.**

The follow-up implemented the vector loader and direct epilogue and recovered
5.2 TFLOP/s. Bundled Steel does not double-buffer or pipeline across K, so those
last two are new experiments rather than missing pieces to copy.

### Verdict

**Not recommended, and now for a measured reason rather than an assumed one.**
§9 called this the best risk-adjusted option because it avoids a second model
implementation. That is still true and it is no longer sufficient: the work is a
Metal GEMM tuning project measured in weeks, whose entire payoff is the ~6.8% in
§9, and which delivers *nothing at all* until it passes 18.1.

Worth keeping in view rather than discarding:

- **The harness is the durable part.** `CustomGEMMTests` checks bit-exactness
  before it reports a rate and now retains the 16.2-TFLOP/s Steel-style control.
  Anyone resuming should start with a Metal GPU capture against MLX, not another
  speculative tiling or double-buffering rewrite.
- **MLX improving is a free win.** The gap §8 found is MLX's, and upstream
  closing it needs nothing from this tree.
- **Attention was never measured against MPS** and remains 37.8% of a block —
  a larger prize than the GEMMs and still unpriced.

---

## 11 — Attention has no headroom *(MEASURED 2026-08-11; the file closes)*

The last unpriced thing. Attention is 446.2 ms of a 1224 ms block, 7.095 TFLOP
of a forward's 961 — **36.4% of a block, a larger single prize than all four
GEMMs together** — and every measurement in this tree had compared MLX against
MLX.

    H3_BIG=1 swift test --filter mpsAttention

Production shape, B=1 H=56 N=15,731 D=128, bf16, both implementations fed the
same bytes:

| | ms | TFLOP/s | |
|---|---:|---:|---:|
| MLX SDPA | 414.8 | 17.1 | |
| MPSGraph SDPA | 409.5 | 17.3 | **1.01x** |

Three samples, 1.01x every time, relative RMS 1.89e-04 — the same attention,
computed at the same rate. Neither materialises the `[B,H,N,N]` score matrix,
which at this shape would be 27.7 GB; both are flash-style and both land on the
same number.

**So §8's gap is specific to the GEMMs and does not generalise.** Two
independent vendors agree on attention to within 1%, which is the strongest
evidence available that 17 TFLOP/s is the machine here rather than one library's
schedule. The 11-22% MLX leaves on `matmul` is a property of MLX's GEMM, not a
property of Apple silicon.

### What that means for the whole investigation

The arithmetic, weighted by where a block's time actually goes:

| | share of block | best available gap | worth |
|---|---:|---:|---:|
| four GEMMs | 733.9 ms, 60.0% | 1.11-1.22x | ~98 ms |
| attention | 446.2 ms, 36.4% | **1.01x** | ~4 ms |
| everything else | ~44 ms, 3.6% | measured repeatedly, ~1% | ~0 ms |

**The ceiling on this machine is about 8%, all of it in the GEMMs**, and §9 and
§10 established that both routes to it cost more than it is worth: routing out
of MLX pays ~28 ms a crossing, and a hand-written kernel starts at 61% of MLX
and must exceed it before delivering anything.

### The roadmap closes again, on a firmer footing

Section 6 closed it on "the model is at MLX's ceiling". That was true and
incomplete — the ceiling was MLX's. Sections 8 through 11 replace it with a
statement about the hardware:

- **Attention is at the machine's rate.** Two vendors, 1.01x.
- **The GEMMs are ~11% off it**, demonstrated, bit-identical, and worth ~8% of a
  render.
- **Nothing in this tree can collect that 8%** without either an MLX-internal
  change or a multi-week kernel project that starts 40% behind.

What remains, and is now the honest full list:

- **MLX improving** — free, needs nothing from here, and §8's table is the
  benchmark to re-run when it does.
- **Fewer steps** — the cache, already shipped at 1.79x plus cap 5.
- **Fewer tokens** — the one untried lever, from `antirez/h3.c`'s token
  reduction (24.5-28.3% there). It reduces S, so it cuts the quadratic term
  rather than making a kernel faster, and it lands on the quality-measurement
  problem 6C never solved.
- **Block streaming** (§7) — not speed, but 64 GB of residency for 1.9-4.9%.

---

## 12 — The MLX GEMM gap upstream *(researched 2026-08-11)*

§11 closed with "MLX improving is a free win — needs nothing from here". That is
still the shape of the opportunity, but it was written without checking what
upstream is actually doing. Checked, it is narrower than it sounded.

### The gap is known, open, and not ours alone

[`ml-explore/mlx#3196`](https://github.com/ml-explore/mlx/issues/3196) — *"NA/M5
addmm / matmul on MPS ~10–20% slower than PyTorch for 1280×1280 BF16"*, opened
2026-03-03, **still open**, assigned to `jagrit06`, labelled `performance`.
Reported figures: `matmul` 1.18x and `addmm` 1.09x slower than PyTorch's MPS
backend, bf16, on M5.

That is the same ratio measured here in §8 — **1.11–1.22x** — and the two
measurements are close to independent:

| | #3196 | §8 here |
|---|---|---|
| chip | M5 (has NAX) | M3 Ultra (no NAX) |
| shape | 1280x1280 | up to 15731x14336x28672 |
| compared against | PyTorch MPS backend | MPSGraph directly |

So the gap is **not** an artefact of one chip generation, one shape class, or
one comparison harness. It is MLX's GEMM.

*Not* corroboration, and worth naming so it is not cited later: the widely
linked [matmul blog post](https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/)
reports 5.5x at 128x128 fp32. At that size the measurement is dispatch overhead,
not GEMM, and it says nothing about these shapes.

### NAX Split-K is already here, and it cannot run on this machine

> **Corrected 2026-08-11.** This section first said NAX Split-K arrived in MLX
> 0.32.0 and that this tree predated it. Both halves were wrong. The claim came
> from a release-notes summary — the same fetch that misdated several mlx-swift
> releases — and it was contradicted by evidence already open in this
> investigation: `steel_gemm_splitk_axpby_nax` sits at `matmul.cpp:687` of the
> **bundled 0.31.1**, alongside `steel_gemm_splitk_nax.metal`. Verify locally
> before citing a changelog.

[`mlx#3017`](https://github.com/ml-explore/mlx/issues/3017) reports large-K
GEMMs on M5 showing high run-to-run variance and slow tail iterations, with up
to **1.62x** recovered by partitioning K.
[PR #3018](https://github.com/ml-explore/mlx/pull/3018) landed that work —
merged 2026-01-26 as `7ed2b6b`, first shipped in **v0.30.4**. This tree's MLX
0.31.1 contains it.

So the fix is present and simply cannot fire here. `mlx/backend/metal/matmul.cpp`
gives two split-K routes and our shapes take neither:

- **Case 1, SIMD split-K** needs `(_tm * _tn) <= 2048` on Max and Ultra, with
  `_tm = ceil(M/16)`. At M = 15,731 that term alone is 983, so `N` would have to
  be about 32. Our shapes score 262,144 to 1,763,328 — over the gate by two to
  three orders of magnitude. It is a path for skinny decode GEMMs.
- **Case 2, NAX split-K** is gated on `metal::is_nax_available()`. NAX is the M5
  neural accelerator; **on M3 Ultra it is false and the branch cannot be taken.**
  Even given the hardware it wants `K >= 3 * max(M, N)` — fc2 has K = 14,336
  against a required 47,193.

**So there is no upgrade that collects this**, and the reason is the hardware and
the shapes rather than the version. Both split-K paths are for K-dominant
shapes; every GEMM in this model is M-dominant.

### Where this tree actually sits

`Package.resolved` pins `mlx-swift` **0.31.6**, which vendors MLX core
**0.31.1** (`Source/Cmlx/mlx/mlx/version.h`). That already includes PR #3018, so
no upgrade is pending for this particular fix.

### What this changes

- **§11's "free win" is still true and further away.** The fix that would close
  this is general steel-GEMM work on non-NAX hardware, which is what #3196 is
  open about and nobody has landed.
- **Re-running §8's table is the right trigger**, not a version number.
  `hardwareCeiling` and `mpsGraphBF16` answer in a minute whether anything
  moved, and no release note can answer it at all.
- **Worth reporting upstream.** #3196 has M5 data at one small shape. This tree
  has M3 Ultra data at five production shapes, bit-identical outputs, and a
  clean MPSGraph-vs-MLX harness. That is a useful comment on an open issue, and
  it costs nothing.
- **Untested and now the interesting question:** whether MLX's large-K
  degradation reported in #3017 exists on non-NAX hardware at all. A K sweep at
  fixed M and N would answer it, and there is no such data upstream.

---

## 13 — Large K on non-NAX hardware *(MEASURED 2026-08-11, revised after review)*

> **Explained by section 14.** The large-M/large-K decay measured here is tile
> selection: it vanishes under the routing patch. This section deliberately
> declined to assert a mechanism, and the mechanism turned out to be one nobody
> had proposed.

> **This section was rewritten.** Its first version compared `M=4096,N=4096`
> against `M=15731,N=5376`, moved both M and N, and concluded "M-by-K
> interaction". It also reported means, while `mlx#3017` is a report about
> *variance and slow tails* — so its "does not reproduce" was unfounded, since a
> mean hides exactly the tail being claimed. And its split-K rows sat entirely
> inside the routing gate, which cannot attribute anything to split-K. The
> harness now crosses M and N independently, records every iteration, and uses
> matched pairs across the routing boundary.

    H3_BIG=1 swift test --filter gemmKSweep

15 iterations per shape, MLX and MPSGraph interleaved so ordering and thermal
drift cannot settle on one arm. Rates are TFLOP/s at the median; `worst/p50` is
the slow-tail factor.

### N does not matter. M does.

| M | K | N | MLX p50 | worst/p50 | MPS p50 | MPS/MLX |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2688 | 4096 | 17.1 | 1.16x | 18.1 | 1.06x |
| 4096 | 5376 | 4096 | 17.5 | 1.06x | 18.8 | 1.08x |
| 4096 | 14336 | 4096 | **18.9** | 1.03x | 18.9 | **1.00x** |
| 4096 | 2688 | 5376 | 17.2 | 1.04x | 18.5 | 1.08x |
| 4096 | 5376 | 5376 | 17.3 | 1.06x | 18.4 | 1.07x |
| 4096 | 14336 | 5376 | **18.7** | 1.02x | 18.8 | **1.00x** |
| 15731 | 2688 | 4096 | 18.1 | 1.31x | 19.4 | 1.07x |
| 15731 | 5376 | 4096 | 18.0 | 1.03x | 19.5 | 1.08x |
| 15731 | 14336 | 4096 | **16.7** | 1.25x | 19.6 | **1.17x** |
| 15731 | 2688 | 5376 | 18.2 | 1.02x | 19.5 | 1.07x |
| 15731 | 5376 | 5376 | 18.1 | 1.11x | 19.6 | 1.09x |
| 15731 | 14336 | 5376 | **16.4** | 1.15x | 19.4 | **1.18x** |

Reading down the two N columns at each M, they agree: **N = 4096 and N = 5376
behave identically**. Reading across M, they do not — at M = 4096 MLX *gains*
with K (17.1 → 18.9) and converges onto MPSGraph; at M = 15,731 it *loses*
(18.1 → 16.4) while MPSGraph holds ~19.5.

With N controlled at two values, the interaction is with M. Two values of N is
not a proof that no N matters anywhere, and this is one machine and one dtype.

### #3017's tail does not appear at #3017's shapes

| M | K | N | MLX p50 | worst/p50 | MPS p50 | MPS/MLX |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 4096 | 4096 | 17.6 | 1.05x | 18.5 | 1.05x |
| 4096 | 12288 | 4096 | 18.8 | 1.02x | 19.0 | 1.01x |
| 4096 | 24576 | 4096 | 18.6 | 1.02x | 18.8 | 1.01x |

Worst-of-15 within 2-5% of the median, out to K = 24,576, and MLX ties
MPSGraph. The slow tail #3017 describes is not present on this hardware at these
shapes — now measured as a tail rather than inferred from an average.

**Tails do show up in the large-M rows** (MLX 1.15-1.31x, and MPSGraph 1.68x at
one point), so variance exists at this model's shapes and is not MLX's alone.
Characterising it is a separate job from this sweep.

**Caveat on the statistic.** At 15 samples the 95th percentile *is* the maximum,
so `p95` and `worst` are one number, not two. Distinguishing them needs a few
hundred iterations per shape; what is reported is "worst of 15 against the
median", which detects a gross tail and cannot describe its distribution.

### Split-K is what MLX is winning with — matched pairs

`_tm*_tn <= 2048` puts N = 1024 inside the Case-1 gate (32x64 = 2048) and
N = 1088 outside it (32x68 = 2176). A 6% change in N changes the kernel MLX
picks and nothing else:

| M | K | N | MLX path | MLX p50 | MPS p50 | MPS/MLX |
|---:|---:|---:|---|---:|---:|---:|
| 512 | 32768 | 1024 | **split-K** | **13.0** | 11.8 | 0.91x |
| 512 | 32768 | 1088 | steel | **8.3** | 11.1 | 1.34x |
| 512 | 8192 | 1024 | **split-K** | **10.6** | 10.3 | 0.97x |
| 512 | 8192 | 1088 | steel | **7.4** | 9.4 | 1.27x |

**MLX drops 36% across a 6% change in N**, from beating MPSGraph to losing to it
by a third, while MPSGraph moves 6%. The advantage tracks the routing decision,
not the shape.

### What is established, and what is not

Established:

- MLX's rate **decays with K at M = 15,731** and rises with K at M = 4096, with
  N held constant across both — so the effect follows M, not N.
- **No slow tail at #3017's shapes on non-NAX hardware**, measured per
  iteration.
- **Split-K causes MLX's advantage** where it routes there, by matched pair.

Not established, and not to be written as though it were:

- **That extending split-K to the large-M corner would help.** Every split-K
  point measured is a skinny shape (M = 512). Nothing here shows the kernel
  helps at M = 15,731, and testing it needs a patched MLX with the gate widened
  — which is the actual next experiment, not a conclusion.
- **A mechanism.** "Loses reuse down a long K loop" is #3017's explanation for
  M5 and is not evidence for what M3 Ultra does at large M.
- **That the tails at large M mean anything.** MPSGraph shows a larger one than
  MLX at the same point.

### For upstream

What is worth posting to [`mlx#3196`](https://github.com/ml-explore/mlx/issues/3196)
is the observation, not the theory: on M3 Ultra, bf16, MLX matches MPSGraph at
M = 4096 for K >= 12288 but falls 11-18% behind at M = 15,731 with K = 14,336,
with N controlled and outputs bit-identical. That is a shape-dependent gap in
the steel path, reported as such, with the harness attached and the mechanism
left open.

---

## 14 — The optimized kernel was already in MLX *(MEASURED 2026-08-11)*

The 16.2-TFLOP/s custom-kernel follow-up in §10 established that another JIT
rewrite was the wrong layer. The next experiment patched only MLX's routing and
swept every non-NAX Steel shape already compiled into its metallib. No shader,
arithmetic or framework integration changed.

At the fc1 production shape, three samples each:

| Steel route | TFLOP/s | result |
|---|---:|---:|
| default `64x64x16, 1x2sg` | 18.0–18.1 | baseline |
| `64x64x16, 2x2sg` | 19.6 | +8% |
| `64x32x32, 2x2sg` | 19.3 | +7% |
| **`32x64x16, 1x2sg`** | **19.7–19.8** | **+9%** |
| `32x32x16, 2x2sg` | 18.3 | +1% |
| `64x32x8, 4x1sg` | 15.7 | −13% |

The best route then ran across the whole §8 table:

| shape | tuned MLX bf16 | MPS | MPS/MLX |
|---|---:|---:|---:|
| qkv | 19.7 | 19.7 | 1.00x |
| attention output | 19.6 | 19.7 | 1.00x |
| mlp fc1 | 19.8 | 19.8 | 1.00x |
| mlp fc2 | 19.4 | 19.4 | 1.01x |
| square 8192³ | 19.5 | 19.3 | 0.98x |

The bf16 gate also passed at all five shapes: **bit-identical to MPSGraph**, with
the existing one-ULP controls detecting changes. This is especially decisive
for fc2: the former 15.3-TFLOP/s outlier reaches 18.6–19.4 depending on the
correctness-versus-throughput harness, tying MPSGraph rather than trailing it by
22%.

### Full-block result

With the tuned route applied to the ordinary MLX `matmul`, the modeled block ran
in **1179 ms**, against the previous roughly 1290-ms range: about **8.6% faster**.
The block remained bit-identical. Routing the GEMMs to MPSGraph now took 1240 ms
and lost 5.1%, as expected once its kernels no longer have an arithmetic-rate
advantage but still pay the framework crossing.

### The actual change

For Ultra-class devices, large bf16/fp16 matmuls with `M >= 8192` and reasonable
K should route from `64x64x16, 1x2sg` to the already-shipped
`32x64x16, 1x2sg` Steel kernel. The conservative M gate preserves the measured
M=4096 regime where the original route already tied MPSGraph.

The upstream-ready patch is `patches/mlx-m3-ultra-large-m-gemm.patch`. The
SwiftPM dependency checkout was restored after measurement; making this durable
in the application requires that patch in an MLX/mlx-swift fork or an upstream
release. No custom kernel should ship.

### Independently reproduced *(2026-08-11)*

Patch applied to the SwiftPM checkout, measured, and the checkout restored
byte-identical afterwards (`diff` clean, baseline fc2 back to 16.6 TFLOP/s).
Two things were verified that the original run did not record.

**fp16, which the patch also changes.** The branch is
`out.dtype() != float32`, so half takes the same route and no fp16 figure had
been recorded — the one gap that could have made this a regression for other
callers. It is not:

| shape | MLX bf16 | MLX fp16 | MPS fp16 | MPS/MLX |
|---|---:|---:|---:|---:|
| qkv | 19.6 | 19.7 | 19.7 | 1.00x |
| attention output | 19.6 | 19.6 | 19.6 | 1.00x |
| mlp fc1 | 19.7 | 19.8 | 19.8 | 1.00x |
| mlp fc2 | 19.3 | 19.3 | 19.5 | 1.01x |
| square 8192³ | 19.6 | 19.7 | 19.1 | **0.97x** |

fp16 improves identically to bf16, and at the square MLX now *beats* MPS.
`mpsGraphBF16` still reports **bit-identical at all five shapes**, with both
controls live.

**The whole large-M plane, not only the production shapes.** Re-running
`gemmKSweep` patched:

| M | K | N | MLX before | MLX after | MPS/MLX after |
|---:|---:|---:|---:|---:|---:|
| 15731 | 2688 | 4096 | 18.1 | **19.6** | 1.00x |
| 15731 | 5376 | 4096 | 18.0 | **19.7** | 0.99x |
| 15731 | 14336 | 4096 | 16.7 | **19.6** | 1.00x |
| 15731 | 2688 | 5376 | 18.2 | **19.6** | 1.00x |
| 15731 | 5376 | 5376 | 18.1 | **19.7** | 1.00x |
| 15731 | 14336 | 5376 | 16.4 | **19.3** | 1.01x |

Every large-M cell ties MPSGraph, at both N and all three K. **§13's large-K
decay was tile selection all along** — it disappears entirely, which is a
better explanation than the "loses reuse down a long K loop" mechanism that
section declined to assert.

And the gate is as conservative as claimed: M = 4096 rows are unchanged
(17.2–19.1, ratios 1.00–1.08x, matching the unpatched sweep), and the Case-1
split-K pairs at M = 512 are untouched.

### What is still unverified before upstreaming

- **`GEMM_TPARAM_MACRO` has three call sites**, not one:
  `steel_matmul_regular_axpby` (measured), **`gather_mm`** and
  **`segmented_mm`** (not). This model exercises the first; MoE and segmented
  callers take the same tile change unmeasured.
- **The threshold boundary is verified at its endpoints only** — 4096 below and
  8192 at the gate. Nothing between 4096 and 8192 was measured, and 8192 is
  where the discontinuity now sits.
- **Transpose variants.** The `2 * max(M,N) > K` branch is transpose-agnostic
  and applies to nn/nt/tn/tt alike; only the combinations production uses were
  measured.

None of these block the result for *this* application, where the affected path
is the one measured and the outcome is bit-identical. They are what an upstream
reviewer will ask for.
