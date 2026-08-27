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
against a 1.15x gate. Every hardware objection is discharged. `fc2` stays off
the engine (saturation).

**Overlapping the two CFG branches was the last idea with a multiple in it, and
it is measured and closed.** It works — the schedule is bit-identical and
recovers real overlap — but it never beats the plain routed path in absolute
time. The engine is too slow relative to the GPU for the columns it would have
to take. See *The CFG overlap ceiling*.

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

**This is short of the 1.15x gate, and `fc2` is not coming** without oracles
for every block. `Tools/ANE/saturation_bound.py` settles it.

## The CFG overlap ceiling

Classifier-free guidance is two independent forwards, and run back to back the
engine is idle for the 37.5% of each block that is GPU attention. Overlapping
them was the remaining idea. It is implemented, it is correct, and it does not
pay. All of the following is measured on this machine.

**The hardware cooperates completely.** An engine evaluation and a GPU burst
submitted together cost `max`, not `sum` — 134.7 ms of engine fully hidden
inside 341.1 ms of GPU, 340.1 ms together against 475.7 serial, 100.7% of the
smaller job recovered (`ANEFormTests.engineBesideGPU`). Nothing about the
silicon is in the way.

**The engine's rate is the wall.** Swept across the whole decomposition — cut by
output column at full length, by sequence tile at full width, and the rectangles
between — the private path holds 7.7–7.95 TFLOP/s across both dies and never
exceeds it (`ANEFormTests.aneForm`):

| s | n | matmul TF/s | conv 1x1 TF/s |
|---|---|---|---|
| 15744 | 3072 | 7.69 | 3.90 |
| 15744 | 10752 | 7.9 | 3.9 |
| 15744 | 21504 | 4.44 | refused |
| 8192 | 10752 | **7.95** | 3.96 |
| 2048 | 3072 | 7.65 | 3.90 |

Expressing the projection as a **1x1 convolution** — the engine's native form,
and what Core ML's own harness measured at 5.46 TFLOP/s a die — is bit-identical
and **2.6x slower** here. That gap was the one candidate for a faster lowering
and it is refuted. Core ML's figure does not transfer: it was measured on a
different decomposition, and the shapes that produced it are ones the ANE
compiler refuses through this path (`C5376H1W21504` is outside its supported
range). The engine wants spatial reuse; a linear over a `1 x S` image has none,
which is also why the public 18.6 TOPS fp16 figures come from 64x64
convolutions and why projects that measure this hardware honestly report 5–9%
of peak.

**So the arithmetic closes.** The GPU sustains 19.0–19.4 TFLOP/s on every
projection in the block and 16.7 on attention (`GEMMCeilingTests`), against the
engine's 7.9. Per CFG pair at production width, 2339.0 ms of GPU-equivalent
work, of which 1050.4 ms is routable. Moving a fraction of the columns to an
engine at 0.41 of the GPU's rate, and paying 180–255 ms a pair to convert,
upload, copy back and re-join, gives:

| engine share | serial | pipelined |
|---|---:|---:|
| unrouted | 2339.0 ms | 2339.5 ms |
| **0.286 (shipping)** | **2219.8 ms** | 2226.8 ms |
| 0.33 | 2318.8 ms | 2244.4 ms |
| 0.38 | 2464.8 ms | 2285.7 ms |
| 0.45 | 2617.2 ms | 2341.0 ms |
| 0.60 | 3069.7 ms | 2546.6 ms |

The pipeline does what it claims — 1.205x against a serial block at the same
share — and the minimum of the table is still the configuration that already
shipped. Raising the engine's share buys overlap and loses more than it buys.

`H3_ANE_CFG_OVERLAP=1` enables the schedule; it is off by default. The receipt
records `aneCFGOverlap` when it ran.

### What the schedule is, since it is kept

Two threads was the wrong shape and measured 0.987x. mlx-swift serialises every
`eval` behind one global recursive lock, so a thread blocking on the engine
holds the GPU's turn and the two halves take turns. There is one MLX thread; the
engine has its own thread and never touches MLX (`ANELinearBackend.Engine`), and
projections are submitted with `begin` and collected with `value` as late as the
data dependency allows. The branches run half a block out of phase so every
engine job has GPU work already queued to hide behind. Two ordering rules were
each worth over 200 ms a pair and are easy to get backwards:

- The engine's activation upload goes on **its own stream**. On the default
  stream it queues behind the attention it was supposed to run beside.
- A branch's post-attention elementwise must be queued **before** the other
  branch's attention is submitted. Metal runs a stream in order, so an `fc1`
  whose `h2` sits behind 441 ms of unrelated attention cannot start until that
  attention ends, cancelling exactly the overlap it was meant to provide.

### Query tiling, measured and closed

Tiling the queries opens a seam the channel split does not have: everything
after attention in a block is row-wise, so tile `i-1`'s `out` and `fc1` can run
on the engine while the GPU attends tile `i`. Softmax is over the key axis, so
splitting queries and keeping the whole KV is arithmetically free — **measured
bit-identical at T = 2, 4, 8, 16** (`TiledAttentionTests`), and the tiled block
is bit-identical to the untiled one at the same share (`QueryTilingTests`).

It works, and it is not enough. One production block, one configuration per
process, each interleaved against the untiled block at the same share:

| post share | T | untiled | tiled | tiling recovers |
|---|---:|---:|---:|---:|
| **0.286 (shipping)** | — | **1082.0 ms** | — | — |
| 0.286 | 4 | 1108.9 ms | 1130.1 ms | −21 ms |
| 0.35 | 4 | 1155.6 ms | 1136.1 ms | +20 ms |
| 0.40 | 4 | 1216.4 ms | 1170.0 ms | +46 ms |
| 0.45 | 8 | 1260.3 ms | 1209.4 ms | +51 ms |
| 0.45 | 16 | 1270.8 ms | 1255.8 ms | +15 ms |

Same-share tiling does nothing, as predicted — it costs the 21 ms of tiling tax
and recovers nothing, because at 0.286 the GPU shard and the engine shard are
already balanced and there is nothing exposed to hide. Tiling recovers 15–51 ms
once the share is raised, which is the right order for the ~58 ms the Gantt
predicted. **But every increase in the share costs more than tiling gives
back**, monotonically, so the minimum of the table is the configuration that
already ships. Best tiled is 1136.1 ms against a 1011 ms gate.

The reason is one number. Moving the share from 0.286 to 0.45 removes 54 ms of
GPU work and adds **144 ms** of engine work, because the engine runs at 0.373 of
the GPU's rate on this shape. Tiling can only hide engine time inside the
attention window; it cannot make it smaller.

**The lever's perfect-execution ceiling is the gate itself.** With 100% engine
duty for the whole window, the post-attention phase costs only its GPU shard:
161 ms at the largest share whose engine time still fits inside 448 ms of
attention, against 252 ms today — a 91 ms saving, landing at **~991 ms against
a 1011 ms gate**. Two percent of margin, requiring the engine to be busy every
millisecond the GPU attends while one MLX thread also runs sixteen uploads and
every gather and norm in the block. Measured duty is about half that, which is
the 1136 ms above. There is no version of this that clears the gate with room.

### Native pack and merge, measured and closed

The tiling result above left one hypothesis alive: that the engine sat at about
half duty because of the per-tile MLX seam — a bf16→fp16 convert, a transpose, a
CPU-visible `asData`, a memcpy in, and a three-way MLX join out, all multiplied
by T. Removing that seam is the only version of query tiling with a path to the
gate, so it was built: a Metal kernel packing straight into the engine's
activation IOSurface, and merge kernels reading both output surfaces. Both are
bit-identical to the seam they replace.

**The seam was not the cost.** Paired in one process, alternating the two paths
sample by sample because the effect is smaller than cross-process drift:

| route | CPU seam | native | |
|---|---:|---:|---:|
| untiled, 0.286 (shipping) | 1035.1 ms | 1046.0 ms | 0.990x |
| tiled T=4, 0.286 | 1094.5 ms | 1092.1 ms | 1.002x |

Native I/O is very slightly *slower* on the route that ships and does nothing on
the tiled one. Against the gate, with the shipping control measured before and
after the candidate and agreeing to better than 2%:

| candidate | control | candidate | gate |
|---|---:|---:|---|
| tiled T=4, post 0.40, native | 1053 ms | 1138.2 ms | misses |
| tiled T=8, post 0.45, native | 1054 ms | 1184.6 ms | misses |

The `tilingTax` measurement already said why: at T=8 the whole per-call overhead
is 5% of the projection — 290.3 ms whole against 305.0 for eight calls. A native
seam can attack that 5%. The missing duty is not there. It is the dependency
chain: attention tile `i` must land before `out i` can be packed, and `out i`
must land before `fc1 i` can be packed, and no amount of faster packing removes
a serial edge.

**Even giving up exactness does not clear it.** The fused SwiGLU merge is the
only kernel that removes real work rather than seam — it emits `[tileS, ffn]`
directly instead of materialising the 28,672-column `fc1` result — and it is
the one kernel that does not reproduce MLX bit-for-bit, so it cannot ship. Run
anyway as a ceiling probe, with the control drifting 0.2–0.4%:

| candidate | control | candidate | gate |
|---|---:|---:|---|
| tiled T=4, post 0.40, fused | 1054.0 ms | 1130.9 ms | misses |
| tiled T=8, post 0.45, fused | 1068.0 ms | 1190.4 ms | misses |

Seven milliseconds, and still 120 above the gate. That is the ceiling of the
direction, measured with the numerical constraint removed.

The native pack and the linear merge stay behind `H3_ANE_NATIVE_IO=1` because
they are exact and correct; the fused merge stays behind
`H3_ANE_NATIVE_FUSED_SWIGLU=1` and must not be enabled until it has an oracle.
Neither is on a production path.

### What would move it

Nothing on this path. The CFG overlap and the query tiling were the two
schedules the arithmetic allowed; both are measured and closed, and so is the
native seam that was the last hypothesis for why tiling underdelivered. The engine would have to reach ~9.3 TFLOP/s — 18% above
anything measured here — before overlapping beats not overlapping, and 1.5x on
a guided step needs the pair under 1560 ms, which is below the GPU-only floor of
1288.6 ms plus any engine share at all. The levers that remain all change the
output: the cross-step cache (shipping, 1.9x, an approximation), sparse
attention over the 37.5% that is attention, or int8 weights against a
bf16-pinned contract.


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
- Render receipts did not record the DiT's compute dtype, so two renders from
  the same seed and checkpoint that produced different videos were
  indistinguishable in their receipts. Receipt schema 4 adds a `compute` block
  carrying the dtype and the projections the engine actually took — observed
  during the render rather than read from configuration, and recorded on
  failures too. It distinguishes a run that declined a projection from one that
  never offered it, because those are different renders.
