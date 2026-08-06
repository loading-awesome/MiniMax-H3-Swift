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

### Acceptance

Repeated control runs — `control-cached` and `control-dense`, same prompt, same
seed, same shape — establish the timing variance every later claim has to clear.
**Do not run anything else on the GPU during a control sweep.** Test runs taken
beside the first repeat moved its step time from 25 s to 29 s, which is larger
than most of the gains this roadmap is chasing.

---

## 6B — Exact fusion *(implemented; speed not yet measured)*

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

### Remaining gates

- [ ] Median production-step improvement ≥ 5%, against a 6A control from the
      same machine. Below that the custom kernel does not earn its maintenance
      surface and should be reverted, not kept "for later".
- [ ] No increase in peak memory.
- [ ] CUDA tap conformance inside its established equivalence class.
- [ ] Warnings-as-errors build, full suite green.

Separately evaluable and not yet attempted: a fused Q/K RMSNorm plus split-half
RoPE. Kept separate because the oracle already measured the unfused attention
path as bit-identical to the reference's fused kernel, so this one is a pure
traffic argument with no correctness upside.

`H3_FUSED_MODULATION=0` forces the readable path at run time, for bisecting a
suspected difference on a real render without a rebuild.

---

## 6C — Sigma-aware dual-stream cache policy *(not started)*

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

### Measure before designing the curve

"Early sigma allows aggressive reuse" is plausible and **unproven for this
checkpoint's back-loaded schedule.** The 6A traces carry per-step sigma against
all three deltas; read them before drawing a curve.

### Promotion gate

Faster than the current balanced profile, and no worse on detail, speech, audio
spectrum, face stability or lip-sync within measured control variance. Faithful
mode stays completely cache-free.

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
