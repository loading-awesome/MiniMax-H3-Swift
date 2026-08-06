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
control-cached-2/3      0.979    0.995    0.973   0.0001   <- same config, noise
fused-cached-1          1.000    1.003    0.998   0.0008
control-dense-1         1.039    1.084    1.069   0.0089
```

**A quality difference below roughly 2% on `accel` or 0.5% on `detail` cannot
be attributed to a configuration change.** Two useful calibrations fall out:
dense carries 8.4% more high-frequency detail than cached, which is the cache's
actual quality cost; and fused modulation sits at eight times the noise floor
on `dssim` but within it on detail — a different render, not a worse one.

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
1. Consecutive-cap sweep: 4 -> 5 -> 6            <- 6C, below
2. Full coherence gates on any faster arm
3. AdaLN schedule batching
4. MLX dense-block compilation
5. Canonical weight layout and GEMM profiling
6. Selective resident quantized matmul research
```

Items 3–6 are untried and, unlike 6B, are not bounded above by activation
traffic — 6B measured 1% because a step at this shape is bound by attention and
the large GEMMs, which is where 5 and 6 aim.

---

## 6C — Consecutive-cap sweep *(next)*

Improve the existing cache **without changing its residual semantics.**

### The constraint that shapes everything

There is one cached residual for the entire packed multimodal stack. Audio and
video **cannot skip independently** — the reuse either happens for the whole
stack or not at all. What can be independent is the *admission test*:

```
reuse only when
    videoDelta < videoThreshold(sigma)
    AND
    audioDelta < audioThreshold(sigma)
```

Independent thresholds and independent vetoes. Not independent reuse.

### Work

1. Use the measured target-video range rather than the whole packed sequence.
   6A records it already; 6C is where it gets a vote.
2. Pass the real video sigma into the policy. `TimestepPlan.sigmaVideo` carries
   it now.
3. Separate `videoThreshold(sigma:)` and `audioThreshold(sigma:)`.
4. Conservative hysteresis: a raw delta spike **always** forces a full step; an
   EMA may confirm a skip but may never hide a spike. First-step, final-step,
   history and consecutive-skip protections all survive unchanged.
5. Stateful filtering stays out of the MLX-free scalar policy.
6. Every decision and threshold in the receipt — already true.
7. The current constant threshold survives as a reproducible legacy profile.

### What the controls actually show *(measured 2026-08-06, 3 runs)*

```
step  sigma   whole   video   audio   decision
   1  0.996   0.216   0.237   0.120   full    videoAboveThreshold
   2  0.991   0.137   0.147   0.112   full    videoAboveThreshold
   3  0.986   0.115   0.124   0.096   full    videoAboveThreshold
   4  0.980   0.096   0.101   0.079   reused
   7  0.957   0.065   0.067   0.056   full    consecutiveCap
  10  0.923   0.059   0.061   0.060   reused
  11  0.908   0.062   0.064   0.065   full    consecutiveCap
  15  0.800   0.087   0.090   0.108   full    consecutiveCap
  16  0.750   0.106   0.109   0.120   full    audioAboveThreshold
  18  0.571   0.159   0.163   0.196   full    audioAboveThreshold
  19  0.387   0.250   0.255   0.246   full    cooldown
```

**"Early sigma allows aggressive reuse" is wrong for this checkpoint.** Steps
1–3 carry the second-largest deltas of the whole render, 0.216 falling to 0.115.
The curve is **U-shaped in step index** with its minimum around step 10 at
0.059, rising at both ends. A monotone sigma curve would be fitted backwards.

**Splitting whole-sequence from video buys almost nothing — but not nothing.**
Video runs a consistent ~3% above whole-sequence, unsurprising once stated,
since video is 95.1% of the rows. It disagrees about a decision **exactly once**
in twenty steps: at step 4, whole is 0.096 and admits reuse while video is 0.101
and would refuse. That single crossing is the entire empirical case for giving
the video probe a vote, and it points the wrong way — it costs a step rather
than saving one. Keep collecting it; do not promote it without a separate
quality argument.

**The audio stream crosses over.** Early it is far quieter than video (0.120
against 0.237 at step 1); late it is louder (0.196 against 0.163 at step 18),
crossing over around step 9.

**But `audioAboveThreshold` in the headline does not mean audio vetoed
anything.** At steps 16–18 the whole-sequence probe is *also* over threshold and
would have forced a full step by itself; audio is reported only because it is
the larger number. The step where audio is genuinely the sole objection is
**15** — whole 0.087, audio 0.108 — and at the shipping cap of 3 that is
invisible, because the cap fires there first. This is why the trace now records
every active constraint rather than one headline.

**From step 4 to step 15 the cache is running on the consecutive cap, not on
its threshold.** The deltas in that window peak at 0.096 against a threshold of
0.10, so the threshold never fires; every refusal there is `consecutiveCap`, at
steps 7, 11 and 15. The knob that governs the middle of the schedule is
`maxConsecutiveSkips`, and nobody has swept it.

### What raising the cap is actually worth

**Not what was first claimed.** The first reading said three capped steps could
be removed for ~26%. That is wrong: the skip counter **resets at every refusal**,
so a larger cap relocates the refresh points rather than deleting them. Replayed
through the real policy against the measured deltas
(`StepCachePolicyReplayTests`):

| cap | reuses | sampling | vs cap 3 | refreshed by cap |
|---:|---:|---:|---:|---|
| 3 | 9 | 658.5 s | 1.00× | 7, 11, 15 |
| 4 | 9 | 658.5 s | **1.00×** | 8, 13 |
| 5 | 10 | 600.9 s | 1.10× | 9, 15 |
| 6 | 10 | 600.9 s | 1.10× | 10 |
| unbounded | 11 | 543.3 s | 1.21× | — |

**Cap 4 is a refresh-placement experiment, not a speed experiment.** Same nine
reuses, same wall clock, refreshes moved. It is worth running precisely because
it isolates residual-age placement from step count.

Caps 5 and 6 buy one step, about 9.6% of sampling. Unbounded buys 21%, not 26%,
and carries the greatest stale-residual risk.

**Unbounded is not unbounded.** With no cap the cache reuses steps 4–14 and is
then stopped by the audio probe at step 15. Once the cap is relaxed, audio is
the only thing standing between the cache and the late high-change region —
which is the argument for keeping the per-stream probe, and it cannot be seen at
cap 3 at all.

These are offline projections against fixed deltas. Changing the cap changes the
trajectory and therefore every subsequent delta, so **rendered arms remain the
only authority**; the projection is for deciding which arms are worth an hour of
GPU each.

Design thresholds *after* that, and shape them U-wise in step index rather than
monotone in sigma — if any threshold-controlled region turns out to be worth
fitting at all. The current trace says the cache already finds the useful middle
window on its own. **The unresolved question is not where to put a threshold; it
is how old a reused full-stack residual can get before coherence breaks.**

### Protocol

Threshold stays at 0.10, per-stream probing on, fusion off. The only thing that
varies is the cap.

1. **Cap 4** — refresh placement at constant step count. Purely a coherence
   probe; if it looks different from cap 3, residual *age* matters independently
   of how many steps are skipped, which is the thing worth knowing.
2. **Cap 5** — only if 4 is visually coherent.
3. **Cap 6** — only if 5 is coherent *and* materially faster.
4. **Unbounded** — only if 4→6 establish a safe trend. It is explicitly a
   cliff-finding experiment and should be labelled as one.

Reject an arm immediately on visible pulsing, geometry drift or warping. Those
are what the Sol-Attn work found by watching output after every tensor metric
said the render was fine.

### Promotion gate

**At least 5% end-to-end over the 1.79× cached control** — not over dense.
Beating dense is not an achievement here; the cache already does it.

Nothing is promoted from one prompt and one seed. A qualifying arm needs
multiple seeds and: localized temporal acceleration, Laplacian detail, speech
WER, audio spectral and envelope correlation, lip-sync margin, face and landmark
stability, and blinded playback. Faithful mode stays completely cache-free.

---

## 6D — Axial-union oracle *(not started)*

Research, isolated from the product surface.

### Deterministic topology

- Prefix queries attend every key densely, and every query keeps the complete
  prefix as keys.
- Video queries attend: all tokens in the same latent frame; the same spatial
  position across latent frames; deduplicated global landmark frames.
- Audio stays in the dense prefix.
- Blocks 0 and 49 dense. First and last sampling steps dense.
- No entropy routing, no per-step content-dependent Top-K.

### The reference must stream one softmax

**Sliced SDPA cannot serve as the oracle.** Separate spatial, temporal and
prefix results cannot be combined afterwards because each was normalised by its
own softmax denominator. The reference has to run one block-streamed online
softmax over the union of allowed keys.

Operate on captured real H3 q/k/v. Prove that selecting every key reproduces
dense. Report prefix, audio and video relative RMS **separately** — §8 of
`SOL_ATTN.md` found they degrade differently, and an average hides it.

Do not optimise the oracle. Its job is correctness and early rejection.

**Stop condition:** if the deterministic topology already produces unacceptable
prefix or audio error, or errors that are unstable between consecutive captured
steps, do not build the Metal backend.

---

## 6E — Correctness-first Metal topology backend *(gated on 6D)*

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
