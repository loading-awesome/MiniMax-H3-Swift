# ANE acceleration: where this stands

Last measured: 2026-08-26 · M3 Ultra, macOS 25F84, ANE architecture `h15g`

Three of a DiT block's four linear projections — `qkv`, `fc1` and `attn out` —
run partly on the Neural Engine behind `H3_ANE=experimental`. The default
production path is unchanged. The bridge validates the private ABI by selector
and refuses unrecognised machines, so an OS update becomes a fallback rather
than a crash mid-sample; `H3_ANE_ALLOW_UNVALIDATED=1` overrides that for
research.

## The position in three lines

**A production block runs 1162.7 ms unrouted and 1047.5 ms routed: 1.110x**,
against a 1.15x gate. Every hardware objection is discharged and the remaining
gap is one projection, `fc2`, which is held off the engine because its partial
sums cannot be proven clear of the 2^15 saturation cliff on the data that
exists.

That is not a scheduling problem, and this document said for some time that it
was. See below.

## Discharged

| objection | verdict | where |
|---|---|---|
| Is the engine fast enough? | **Yes** — 3.85–3.87 TFLOP/s per die, 7.64 across both under `h3_ane_run_pair` | `ANE_REVERSE_ENGINEERING` |
| Do both dies run at once? | **Yes** — two evaluations in flight cost the same as one (19.90 ms vs 19.91), and `kANEFAneInstanceHint` has nothing to do with it | `ANE_REVERSE_ENGINEERING` |
| Can Metal and ANE share memory? | **Yes**, IOSurface, zero copy | `ANE_REVERSE_ENGINEERING` |
| Is there bandwidth for both? | **Yes** — peak DRAM read 192 GB/s against ~800 available | `ANE_REVERSE_ENGINEERING` |
| Is fp16 accurate enough? | **More accurate than bf16** — 7e-5 to 5e-4 per projection against the bf16 GPU path's 1.66e-3, on real captured tensors | `ANE_PRECISION_RESULTS` |
| Does that survive 1000 block evaluations? | **Yes** — a full fp16 render passes every gate the tree owns | `ANE_PRECISION_RESULTS` |
| Is there a better lowering to find? | **No** — static and dynamic paths are bit-identical | `ANE_REVERSE_ENGINEERING` |
| Can results get back into MLX cheaply? | **Yes** — `MLXArray(rawPointer:)` maps an IOSurface at the same address, so the engine's output is adopted rather than copied | `ANELinearPrimitive` |
| Can the GPU work while the engine runs? | **Yes** — `asyncEval` on its own stream: 20.2 ms overlapped against 39.9 ms serial | `ANELinearPrimitive` |

## The scheduling question, retired

This page carried a table projecting **1.078x if the engine could not be fed
without draining MLX's lazy graph, and 1.214x if it could**, resting on a
`crossingBreakdown` attribution of 29 ms to copying results home and **120 ms to
fusion and scheduling loss** — MLX giving up the norms, modulation, RoPE and
gating it would otherwise fuse around a projection.

**The 120 ms is not there.** Measured directly, comparing an input that is
already materialised against one that arrives as a lazy elementwise chain, which
is what a real block hands over:

| input to the routed projection | plain | routed | |
|---|---:|---:|---:|
| already materialised | 196.1 ms | 161.5 ms | 1.214x |
| lazy upstream chain | 198.6 ms | 163.2 ms | 1.216x |

No difference. Materialising the activation for the engine costs essentially
nothing beyond the work MLX had to do anyway, so there is no barrier for a
double-buffering scheme to hide and nothing to win by feeding the engine from
inside MLX's stream. The two seam questions this page used to pose are answered
anyway, and in opposite directions: `MLXFast.metalKernel` allocates its own
output buffer and cannot be pointed at an IOSurface, while `MLXArray(rawPointer:)`
aliases one exactly — which is the direction that mattered, and is what the
routed path uses.

What actually decided the number was four much duller things, listed under *What
had to be right* below. The largest was a strided `asData` that turned a 22 MB
copy into an element-wise gather and held the whole route at **0.003x**.

## Current integration evidence

Three of a block's four linears are routed. Measured in isolation at production
sequence length, S=15731:

| projection | N | plain | routed | |
|---|---:|---:|---:|---:|
| `qkv` | 21504 | 196.4 ms | 150.6 ms | **1.304x** |
| `fc1` | 28672 | 262.7 ms | 197.0 ms | **1.334x** |
| `attn out` | 5376 | 66.1 ms | 58.8 ms | **1.124x** |

`attn out` returns least because it is the one projection whose contraction
axis is larger than its output — 7168 in, 5376 out — so the fixed cost of
converting and copying the activation is spread over a smaller GEMM.

A whole block at production width, `CompiledBlockTests.productionBlock`, median
of 11:

| | |
|---|---:|
| routing off | 1162.7 ms |
| routing on | **1047.5 ms** |
| | **1.110x** — 5.8 s off a 50-block forward |

**This is short of the 1.15x gate, and `fc2` is what is left — and `fc2` is not
coming.** `Tools/ANE/saturation_bound.py` settles it.

### The saturation bound

The earlier figure sampled a 64x64 corner and measured a cumulative sum in index
order. Both are the wrong instrument. Index order is one ordering out of `K!`,
and the engine's is undocumented — it tiles the contraction across its MAC array
in a pattern nobody here has reverse-engineered — so a claim resting on index
order is a claim about a computation the hardware does not perform. What holds
under *any* order is

    |any partial sum| <= sum_k |a_k * w_k|

which is one GEMM on the magnitudes. Run over all 1024 captured rows and all `N`
output channels of all nine production oracles, against the real checkpoint
weights, at the 1/16 operand scale this path applies (so unscaled partials must
stay under 524,288):

| projection | worst bound | block | headroom | verdict |
|---|---:|---:|---:|---|
| `mlp fc1` | 2,428 | 49 | **216x** | proven under any order |
| `qkv` | 5,132 | 49 | **102x** | proven under any order |
| `attn out` | 69,912 | 49 | **7.5x** | proven under any order |
| `mlp fc2` | 978,586 | 49 | **0.5x** | **not proven — exceeds by 1.9x** |

The three routed projections are safe with margin, and safe for a stronger
reason than "we measured it and it was fine": no accumulation order exists that
could saturate them. `fc1` and `qkv` would clear even unscaled.

`fc2` does not clear, and the depth trend is why it will not be argued around.
Its bound runs 43,746 at block 0, **10,252** at block 24, and 978,586 at block
49 — non-monotonic, and rising two orders of magnitude over the last half of the
stack. A 1/64 scale would prove it on this data at 2.1x. That is not enough:

- The oracles cover **three of fifty blocks**. Blocks 25 through 48 are
  unmeasured, and the one thing the measured points establish is that this
  quantity does not move smoothly with depth.
- They cover **1024 of 15,406 sequence positions**, 6.6%.
- The failure returns **zero** with nothing downstream able to see it.

A 2.1x margin, extrapolated across 24 unmeasured blocks, against a silent
failure, to gain 33 ms a block. No.

**So 1.110x is where this route stands until oracles exist for every block.**
That is a capture job, not a code job, and it is the honest prerequisite for
reconsidering `fc2`. Reaching a gate by guessing at a silent-corruption
threshold is not reaching it.

### What it costs in memory

Weights are uploaded to the engine as fp16 copies and kept, so they are
duplicated against the bf16 originals MLX already holds:

| | session surfaces | weights per block | across 50 blocks |
|---|---:|---:|---:|
| `qkv` | 346 MB | 63 MB | 3.08 GB |
| `fc1` | 407 MB | 84 MB | 4.10 GB |
| `attn out` | 261 MB | 21 MB | 1.03 GB |
| | **1015 MB** | | **8.20 GB** |

**9.2 GB**, against a render that already peaks at 87.2 GB. Nothing evicts
these while the model is alive.

### What had to be right

Four things, each worth more than the engine itself:

- **Contiguity.** `transposed().asType()` leaves a strided view, and
  `asData(.noCopyIfContiguous)` silently drops out of its no-copy path into an
  element-wise gather. Both the activation and the weight upload hit this. With
  `MLX.contiguous` in front, a 22 MB copy went from 8792 ms to 0.4 ms — the
  route was running at **0.003x** until this was found.
- **Ordering.** The activation must be materialised *before* the GPU shard is
  submitted. Draining for it afterwards waits on the whole GPU matmul, so the
  engine starts only once the GPU is done and the costs add: 39.9 ms serial
  against 20.2 ms overlapped, for work that is 19.2 and 19.8 ms alone.
- **Sequence padding.** The engine sustains 3.87 TFLOP/s a die at s=14336 and
  s=16384 but 2.45 at s=15731, which is prime. Compiling for the next multiple
  of 64 and slicing the surplus off is 1.215x against 0.789x.
- **Shard balance.** The GPU sustains ~18.6 TFLOP/s here, not the 16 the
  roadmap quotes, so a share derived from 16 over-loads the engine and leaves
  the GPU waiting. Swept instead: 3072 columns a die, where both sides land
  within 1 ms of each other. 1.459x at the balance point against 1.303x at the
  derived split.

The suite runs by default in `swift test`, with no environment variable, on any
machine the bridge validates — and it carries **a timing assertion**, which is
the one that matters. Every correctness test here stayed green through that
400x slowdown.

This is operator conformance, not release conformance. The route has not yet
cleared the plan's full-block 1.20x gate, a complete 50-block forward, repeated
forward leak/thermal testing, or trajectory-level quality acceptance. It must
remain experimental until those gates pass.

## Two things that must hold before any routing

**Saturation is a cliff, not a gradient.** A running partial that reaches 2^15
makes the engine return **zero** — not inf, not NaN, nothing downstream can
detect it. Block 49's `fc2` breaches it on real data and destroys 0.02% of its
outputs, taking that projection from 1.0e-4 to 0.101.

This is now enforced rather than noted. `Tools/ANE/saturation_bound.py` proves a
bound that holds under any accumulation order, and no projection is routed
without clearing it — see *The saturation bound* above. `fc1`, `qkv` and
`attn out` clear at 216x, 102x and 7.5x; `fc2` does not clear and is not
routed.

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

MLX gets faster, and every gain shrinks the engine's share of the work: the GPU
already sustains ~18.6 TFLOP/s on these shapes rather than the 16 this project
assumed, which is why the shard split had to be retuned. A route that buys
1.110x today buys less against a faster MLX tomorrow, and it buys it through
undocumented private APIs pinned to one OS build.

Against that: the arithmetic is measured, the safety bound is proved rather than
sampled, and the failure mode is a fallback rather than a wrong number. What it
does **not** yet have is a full 50-block forward, leak and thermal behaviour over
a real render, or trajectory-level quality acceptance — and it is short of its
gate. It should stay experimental.

## Documents

| | |
|---|---|
| `ANE_REVERSE_ENGINEERING` | machine fingerprint, ABI, numerics of `h15g` |
| `ANE_PRECISION_RESULTS` | real tensors, the saturation cliff, the fp16 render |

| tool | |
|---|---|
| `Tools/ANE/saturation_bound.py` | order-free partial-sum bound per projection per block |
| `Tools/ANE/underflow.py` | denormal flush and the engine datapath model |
| `Tools/ANE/counters.h` | IOReport per-die energy and DRAM bandwidth |

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
