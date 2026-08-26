# ANE acceleration: where this stands

Last measured: 2026-08-26 · M3 Ultra, macOS 25F84, ANE architecture `h15g`

**Nothing is wired into the model.** Every number here comes from probes in
`Tools/ANE/` and tests in `Tests/H3ModulesTests/`, and the production path is
unchanged.

## The position in three lines

Every hardware objection to putting DiT linears on the Neural Engine has been
measured and discharged — rate, dual-die execution, bandwidth, precision,
end-to-end coherence. What remains is a single MLX scheduling question, and it
is worth **1.078x against a 1.15x gate if the answer is no, and 1.214x if it is
yes**.

The question: *can the engine be handed its input without draining MLX's lazy
graph?*

## Discharged

| objection | verdict | where |
|---|---|---|
| Is the engine fast enough? | **Yes** — 3.65–3.69 TFLOP/s per die, ~7.4 across both, beside the GPU's 16 | `ANE_OVERLAP_RESULTS` |
| Do both dies run at once? | **Yes**, but concurrency engages the second die, **not** `kANEFAneInstanceHint` | `ANE_OVERLAP_RESULTS` |
| Can Metal and ANE share memory? | **Yes**, IOSurface, zero copy, error 0 | `ANE_OVERLAP_RESULTS` |
| Is there bandwidth for both? | **Yes** — peak DRAM read 192 GB/s against ~800 available, lowest bin ≥85% of every arm | `ANE_OVERLAP_RESULTS` |
| Is fp16 accurate enough? | **More accurate than bf16** — 7e-5 to 5e-4 per projection against the bf16 GPU path's 1.66e-3, on real captured tensors | `ANE_PRECISION_RESULTS` |
| Does that survive 1000 block evaluations? | **Yes** — a full fp16 render passes every gate the tree owns | `ANE_PRECISION_RESULTS` |
| Is there a better lowering to find? | **No** — static and dynamic paths are bit-identical; the arithmetic is fixed hardware | `ANE_DIFFERENTIAL_RESULTS` |
| Can results get back into MLX cheaply? | **Yes** — a Metal write lands in an MLXArray's own backing, so nothing is copied home | `MLXSeamTests` |

## The one live constraint

A hybrid block saves **232 ms**. Four crossings cost **149 ms**, and
`crossingBreakdown` splits that:

| component | cost | removable |
|---|---:|---|
| copying the result home | 29 ms | **yes**, by aliasing |
| fusion and scheduling loss | **120 ms** | only by not draining MLX |

The 120 ms exists only in block context — it is the norms, modulation, RoPE and
gating that MLX fuses around a projection and gives up when the graph has to
materialise mid-block. In isolation there is no barrier at all.

| design | block | speedup |
|---|---:|---:|
| barrier as measured | 1102 ms | 1.050x |
| copy removed by aliasing | 1073 ms | **1.078x** |
| fusion loss also removed | 953 ms | **1.214x** |

Gate is 1.15x, so it turns on the last row.

## Next experiment

Narrow, and not "build the bridge":

1. Can `MLXFast.metalKernel` — which compiles into MLX's own command stream
   with no `eval` — write an activation into an IOSurface?
2. Can a Metal shared event be signalled from inside that stream, so the engine
   starts off the kernel rather than off a full graph drain?

Both are cheap and both are decisive.

## Two things that must hold before any routing

**Saturation is a cliff, not a gradient.** A running partial that reaches 2^15
makes the engine return **zero** — not inf, not NaN, nothing downstream can
detect it. Block 49's `fc2` breaches it on real data and destroys 0.02% of its
outputs, taking that projection from 1.0e-4 to 0.101. A power-of-two operand
scale is exact in fp16 and buys 15x headroom at no precision cost. Every
enabled projection needs a measured `max|interior partial|` bound with margin,
per block. Published figures are maxima over samples, not full tensors, so the
real margin is tighter.

**This path cannot be gated against the bf16 goldens.** Any arithmetic change
reselects the diffusion sample, so a comparison against them always fails and
always for the wrong reason. Conformance has to be independent quality
measurement plus self-consistency, against goldens regenerated under this
path's own arithmetic.

## No longer on the critical path

**Static resident weight banks and the ~4 GiB per-instance address window.** The
design supplies weights dynamically through IOSurface, so residency is moot —
and since the instance hint does not choose a die, per-die residency would be a
correctness hazard rather than a scheduling one.

**Further engine characterisation.** The fingerprint in
`ANE_REVERSE_ENGINEERING` is complete enough to build against.

## The argument against, which is not a measurement

MLX gained **67 ms a block** between the roadmap's `crossingCost` and today's
re-run. Every such gain shrinks the engine's share and leaves the barrier
untouched, so this route gets worse with time rather than better. Weigh that
against a 1.214x best case reached through undocumented private APIs that break
on OS updates.

## Documents

| | |
|---|---|
| `ANE_ACCELERATION_PLAN` | the design, its gates, and the corrections to both |
| `ANE_REVERSE_ENGINEERING` | machine fingerprint, ABI, numerics of `h15g` |
| `ANE_DIFFERENTIAL_RESULTS` | weight binding, layout invariants, static vs dynamic |
| `ANE_OVERLAP_RESULTS` | concurrency, per-die energy, DRAM bandwidth |
| `ANE_PROJECTION_SPIKE` | one production QKV projection, split three ways |
| `ANE_PRECISION_RESULTS` | real tensors, the saturation cliff, the fp16 render |

## Findings that outlived the question

Work here turned up defects unrelated to the engine, all fixed and committed:

- Three of the four external quality oracles could not build. `LipSyncCheck`
  imported `H3Core`, a module the package has not had for some time, and both
  Swift tools were loose files with no target so nothing compiled them. They are
  targets now.
- The lip-sync drift check measured *spread*, which mixes drift with the noise
  of estimating a lag from a one-second window. It fits a weighted trend now.
- Render receipts do not record the DiT's compute dtype, so two renders from
  the same seed and checkpoint that produce different videos are
  indistinguishable in their receipts. **Not yet fixed.**
