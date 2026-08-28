# ANE acceleration: where this stands

Last measured: 2026-08-26 · M3 Ultra, macOS 25F84, ANE architecture `h15g`

Three of a DiT block's four linear projections — `qkv`, `fc1` and `attn out` —
run partly on the Neural Engine behind `H3_ANE=experimental`. The default
production path is unchanged. The bridge validates the private ABI by selector
and refuses unrecognised machines, so an OS update becomes a fallback rather
than a crash mid-sample; `H3_ANE_ALLOW_UNVALIDATED=1` overrides that for
research.

> **Only builds on the gate's audited list route to the engine.** Earning a
> place needs the selector audit (`Tools/ANE/abi-check.m`) and a clean
> `Tools/ANE/pair-stress.m` watch — see *Machine safety*. macOS 26.5.2 and
> earlier carry a driver defect that hard-locks the machine and are not on it.

## The position in three lines

**The 1.15x gate is cleared: 1.172x on a block, 1.179x on a real render.** A production block runs 1136.1 ms
unrouted, 1041.0 ms on the previous routed path, and **969.4 ms** with the
contraction split and the partial join fused. `fc2` stays off the engine
(saturation), and the two overlap schedules stay closed — the win is entirely in
how the engine is fed.

**Overlapping the two CFG branches was the last idea with a multiple in it, and
it is measured and closed.** It works — the schedule is bit-identical and
recovers real overlap — but it never beats the plain routed path in absolute
time. The engine is too slow relative to the GPU for the columns it would have
to take. See *The CFG overlap ceiling*.

### Persistent MLP island: production topology executes

The next tensor-parallel form keeps a device's matching `fc1` gate/up neurons,
SwiGLU activation, and corresponding `fc2` rows together, returning one
hidden-width partial instead of joining the 28,672-column `fc1` result. The
first bounded spike establishes two different boundaries:

- A single four-input MIL graph containing both `fc1` matmuls, SwiGLU, and
  `fc2` **compiles and loads but is rejected at inference** with ANE status
  `0x1d`. Replacing native `silu` with `sigmoid` and `mul` does not change the
  refusal (`Tools/ANE/mlp-island-spike.mm`). Compiler acceptance is therefore
  not evidence that this runtime can execute a multi-weight island.
- Chaining the bridge's proven two-input matmul programs through retained
  IOSurfaces **executes correctly**. One Metal dispatch now reads the separate
  fp16 gate/up surfaces, restores the `fc1` operand scale, applies
  `SiLU(gate) * up`, applies the downstream scale, and transposes `[S,F]` into
  the `[F,S]` surface consumed directly by row-sharded `fc2`. At the safe
  `S=64, K=128, F=256, H=128` fixture the complete warmed chain takes 1.06 ms
  serially and returns relative RMS `4.12e-4` against an fp32 reference
  (`Tools/ANE/mlp-island-chain-spike.mm`).

`ANENativePackMergeTests.nativeSwiGLUTransposeFeedsNextANEProjection` varies
both row and column, so it proves the transpose as well as the arithmetic and
pins exact power-of-two scale/unscale behavior across the seam. Nothing from
this spike is on the render path.

The inline probe now measures that exact candidate when explicitly requested:

    H3_ANE_BOUND=/path/mlp-island-bound.json \
    H3_ANE_BOUND_MLP_ISLAND=1 \
    h3 render ...

It records per-block `fc2 island bNN ane0 split4` and `ane1 split4` entries over
every row and faithful step, including the largest power-of-two operand scale
that retains 2x saturation headroom. The initial plan assigns the first half of
the 14,336 SwiGLU neurons to Metal and one 3,584-neuron quarter to each ANE;
each ANE quarter is split into four 896-wide contractions.

The final 50-block calibration is recorded at
`/Volumes/scratch_disk/h3-gates/mlp-island-2048-calibration.json` for the exact
`GPU=10240, ANE0=2048, ANE1=2048` partition. Of 100 ranges, 76 need no scaling,
13 need `1/2`, three `1/4`, three `1/8`, one `1/16`, two `1/32`, one `1/64`,
and one `1/256`. The outlier is block 39 / ANE1, whose four-piece order-free
bound is 3,364,614 at progress 0.80; `1/256` leaves 2.49x cliff headroom.
`Tools/ANE/mlp_island_scales.py --neurons-per-die 2048` validates completeness,
the exact contraction ranges, power-of-two scales, minimality, and headroom
before emitting the Swift table.

A second identical faithful render captured 1,024 evenly spaced rows of the
actual block-39 SwiGLU activation at that exact step. Scoring only the final
ANE1 `12288..<14336` range with 256 output channels gives **4.09e-4 relative
RMS** at `1/256`, versus 1.66e-3 for the bf16 GPU reference class. Despite
14.51% flushed subnormal products, the scaled maximum partial is only 28.1 and
no output is zeroed. The one-shot capture is opt-in:

    H3_CAPTURE_MLP_ISLAND=/path/captures \
    H3_CAPTURE_MLP_BLOCKS=39 H3_CAPTURE_MLP_AFTER=0.8 \
    h3 render ...

and `Tools/ANE/underflow.py --k-range 12288:14336` scores the actual shard.
A global `1/256` remains invalid: early-block oracle captures measured
`3.6e-3` to `4.6e-3`. The calibrated per-block/per-die table is therefore part
of the arithmetic contract.

The full topology now executes at production dimensions in
`Tools/ANE/mlp-island-dual-spike.mm`: padded `S=15488`, `K=7168`, `H=7168`,
gate pair, up pair, one paired Metal seam, then fc2 pair. The sequential control
for two 3,584-neuron islands is 1,042.0 ms; the paired schedule is **520.3 ms**
(median of three), so both dies deliver the expected 2.00x concurrency without
a lock or timeout. That result also rejects the original 50/25/25 ownership:
the full GPU MLP is about 435 ms, so giving half its neurons to a 520 ms ANE
critical path is slower.

Rate balancing lands near 29% total ANE ownership. Direct, unsplit diagnostics
locate that balance, but they are not the saturation-safe execution contract:

| neurons per ANE | total ANE share | unsplit paired island | estimated GPU remainder |
|---:|---:|---:|---:|
| 3,584 | 50.0% | 520.3 ms | 217.7 ms |
| 2,176 | 30.4% | 320.9 ms | 303.2 ms |
| 2,112 | 29.5% | 311.8 ms | 307.2 ms |
| **2,048** | **28.6%** | **286.8 ms** | **311.0 ms** |

The safe version is four 512-neuron pieces per die. A production piece takes
79.16 ms as a paired gate/SwiGLU/fc2 chain, so four cost **316.6 ms**. The native
final join—GPU bf16 partial plus eight scaled ANE fp16 partials into bf16—takes
**19.69 ms** at production shape and has an exact conformance test. With the
estimated 311.0 ms GPU remainder running concurrently, the MLP critical path is
therefore about **336 ms**, or **1.29x** over the 435 ms full-GPU MLP. The
join can likely be partly hidden by accumulating each completed piece while the
next ANE pair runs, but that must be measured in the integrated scheduler.

This cannot make the whole block 1.5x by itself—the MLP is only about 38% of
the block—so attention remains necessary for that target.

Saturation, worst-case precision, dual-die execution, and the final join are now
green for the exact 10,240/2,048/2,048 split. The implementation gate is to
build the four-piece block scheduler and measure GPU/ANE overlap rather than
summing isolated timings. One prompt's
full bound is not a universal input proof, so a production route also needs
either a broader calibration corpus or a runtime range guard with GPU fallback.

### Integrated scheduler result (2026-08-27) — measured on the wrong shape

**Every number in this section is from a synthetic MLP the model does not
contain.** See *The island had never routed*. It is kept because the
implementation findings — the 49-entry table, the fc1 scale — are real and were
found here. The timings are not about this model.


The persistent island is now wired into `H3MLP` behind both
`H3_ANE=experimental` and `H3_ANE_MLP_ISLAND=1`. It owns persistent compiled
programs and weight surfaces, submits the ANE work on its own queue, computes
the Metal-owned neurons concurrently on a separate MLX stream, and uses the
native one-pass partial join. A post-submit failure quarantines the mutable
session and recomputes the complete MLP on Metal.

Conformance found two implementation errors before timing:

- The first table transcription contained 49 entries, shifting every block
  after the omitted row; block 39 received `1/2` instead of its proven
  `1/256`. The checked table is now the exact 50-row generator output, with a
  test pinning its count and block-39 value.
- Reusing the generic projection's `1/16` fc1 input scale produced 1.3% error
  in a one-piece identity-down oracle. fc1's faithful bound peaks near 72 and
  does not require scaling. Leaving fc1 unscaled reduces the complete hybrid
  MLP error to 0.00362 relative RMS against monolithic bf16; the hybrid result
  is 0.00444 from the fp32 oracle against 0.00430 for the monolithic GPU path.
  The independently calibrated per-block/per-die fc2 scales remain unchanged.

The integrated timing also invalidates the sum-of-isolated-parts projection.
Four active 512-neuron pieces per die with four-way fc1 contraction splitting
measured only **1.061x** (536.7 ms GPU, 506.1 ms hybrid). One unsplit fc1
contraction fell to **0.911x** under simultaneous GPU load even though it is
faster in isolation: shared execution changes the ANE critical path. Returning
one already-calibrated 512-neuron piece per die to Metal is the best measured
balance so far. With three active pieces per die it measures **1.102x**
(537.1 ms GPU, 487.4 ms hybrid) at `S=15,461`, including the final join.

This is a correct experimental backend, not the route to a 1.5x deliverable.
At the measured 38% MLP share, 1.102x on the MLP alone implies only about
**1.036x** for a block if everything else is unchanged. The next performance
work must move independent deliverable work onto the ANE while the GPU executes
attention (or another stage), rather than assigning more of the same MLP and
driving both engines into the shared-fabric critical path.

### The wavefront rejects the island, and the dies are idle 76% of the block (2026-08-27) — mislabelled

**The island declined every block in this run too**, so what the table below
compares is not the island. The engine-duty measurements stand; the attribution
does not. See *The island had never routed*.


The island was then wired into the CFG wavefront as a split-phase submission:
`beginMLP` submits the whole MLP and returns, the other branch's attention is
submitted next, and `finishBlock` collects — so the ANE tail is covered by GPU
attention instead of waited on. In the run below, `qkv` and `attn out` go back to Metal while the island holds
the dies, on the belief that four concurrent evaluations on the private runtime
was the load the spikes hard-locked under. **That belief is withdrawn** — see
*The concurrency constraint is withdrawn* — so this measures the island against
the projections rather than beside them.

Measured against the same wavefront without the island, interleaved in one
process at production width, depth 4 (`cfgStackIsland`):

| schedule | per block-pair | engine busy |
|---|---|---|
| pipelined | 1928.7 ms | 1870 ms of 7682 ms wall (**24.3%**) |
| pipelined + island | 2082.9 ms | 905 ms of 8286 ms wall (**10.9%**) |

**0.926x — the island is a 6% regression in the integrated schedule**, and the
two runs agree (0.943x on the first). It is not close, and the reason is on the
right-hand column rather than in the MLP arithmetic: the island puts *less* work
on the dies than the projection routing it displaces (905 ms against 1870 ms of
die time), while pushing `qkv` and `attn out` — the work it evicted — back onto
the contended GPU. Hiding the join cannot recover this. The whole join is
19.7 ms per MLP, 39 ms a pair, against a 154 ms deficit.

The engine-busy column is the number that should drive the next decision, and
nothing before this measured it. `ANELinearBackend.EngineMeter` times what the
engine queues actually spend inside an evaluation. In the best schedule the
dies are working **24.3% of the block** — 3.55 TFLOP a pair, against the ~22.7
TFLOP a branch-block of arithmetic the stack contains, so under 8% of the work.
Saturating both dies for the whole block is what the 1.48x ceiling assumes.
Raising duty from 24% to near 100% cannot come from more MLP: it needs the
largest stage the ANE does not touch, which is attention.

**The MLP island stays opt-in and stays off.** It is correct, its calibration
holds, and — as measured here, evicting the projections — it is the wrong place
to spend the dies. Whether it is the wrong place *beside* them is a different
question, and one the eviction rule prevented anyone from asking.

### The concurrency constraint is withdrawn (2026-08-27)

The island evicted `qkv` and `attn out` from the engine because four concurrent
evaluations — the island's pair plus a projection's pair — were believed to be
what hard-locked the machine. Nothing measured that. It was inferred from a
single dual-evaluation lock, and the sleep race is a better explanation of that
event: dual mode is simply the first thing to touch a die that has been idle
long enough to sleep.

The inference does not survive its own evidence:

- The lock **reproduced with no concurrency at all** — serial mode, one
  compiled model, one head, S=512. Concurrency is not necessary for it.
- **Two evaluations in flight were already measured as free**: 19.90 ms against
  19.91 ms for one, which is the whole basis of `h3_ane_run_pair`. The dies have
  been running concurrently in the shipping path all along.
- Four in flight has **never been run**. It was refused, not measured.

What this reopens, in the order it matters:

1. **The island beside the projections.** Eviction cost 1870 ms of projection
   die time to buy 905 ms of island die time, which is most of the 0.926x. The
   two together are the only configuration that can raise duty above 24.3%
   without new kernels. `CFGOverlap.islandEvictsProjections` now defaults to
   sharing; `H3_ANE_ISLAND_EVICTS=1` restores eviction for comparison, and
   `cfgStackIsland` measures all three.
2. **The attention spike's dual modes**, which were gated on the same belief.
3. **A block-level schedule with several ANE queues in flight** — the
   rate-balancing and command-DAG steps of the plan both assume it, and both
   were closed by an argument that has now been withdrawn.

None of this is a claim that four evaluations are safe. It is a claim that the
question is open and cheap to answer, where before it was closed by something
that was never true. The prerequisite is the power bracket: the hazard that
actually exists is the sleep race, and it applies to one evaluation as much as
to four.

### The island had never routed (2026-08-27)

`ANEMLPIslandBackend.expectedHidden` was written as **7,168**. That is
`innerDim` — the attention width, 56 heads x 128 — and not the MLP's hidden
size, which is **5,376** (`H3Config.hiddenSize`). A real block arrives as
`x [S, 5376]`, `fc1 [28672, 5376]`, `fc2 [5376, 14336]`, so the guard
`hidden == expectedHidden` refused it, every time, in every configuration.

**The island has never executed a single production block.** Everything
attributed to it has to be reassigned:

| reported | what it actually measured |
|---|---|
| island MLP at 1.102x, 0.00362 rel-RMS | a synthetic MLP with hidden 7,168, which this model does not contain |
| wavefront island at 0.926x, rel-RMS 0.0126 | evicting `qkv` and `attn out` from the engine — `fc1` stayed routed via `H3MLP.begin`, which has no engine gate |
| island die time 905 ms | `fc1` alone |

The reassigned reading of the wavefront run is still worth having, because it
measures something real: **moving `qkv` and `attn out` off the engine costs 4.3%
and halves die time** (1906 ms to 914 ms). That is a clean result about
allocation. It is simply not a result about the island.

Two things kept this invisible, and both are worth naming:

- **The conformance test built its fixture from the constant under test.**
  `let hidden = ANEMLPIslandBackend.expectedHidden` — so the island was checked
  against its own assumption and would have passed at any value. Six tests, all
  green, all blind. The fixtures now come from `H3Config`, so a drifting
  constant fails them.
- **A decline before submission leaves nothing on the receipt.** That is correct
  — it is not a routing failure — but it meant a schedule that had quietly
  stopped routing looked identical to one that never tried. `beginRoute` now
  reports the first decline under `H3_ANE_ISLAND_TRACE=1`, and the benchmark
  fails rather than passes when both sides come out bit-identical.

The constants are now derived from `H3Config` rather than restated. With that
one change the island routes on a real block for the first time, at
**0.00036 rel-RMS** against the plain schedule over one block-pair.

### Four evaluations in flight work, and utilisation is not the lever (2026-08-27)

With `expectedHidden` corrected, the island routes and the three allocations can
finally be compared. Depth 4, production width, interleaved in one process:

| schedule | per block-pair | vs baseline | engine work | wall |
|---|---|---|---|---|
| projections only | 2083.3 ms | — | 2009 ms | 8406 ms |
| island + projections | 2175.2 ms | **0.958x** | **5967 ms** | 8822 ms |
| island, evicting | 2267.7 ms | 0.919x | 3303 ms | 9210 ms |

Conformance over four blocks: 0.00091 and 0.00086 rel-RMS.

Two results, and the second matters more than the first.

**Four concurrent evaluations are fine.** The island's pair and a projection's
pair ran together for five minutes across three schedules, and engine work
nearly tripled — 2009 ms to 5967 ms of die time. The machine did not lock. The
constraint that closed this direction was never real, and it can be dropped from
the planning. (`EngineMeter` sums both queues, so 5967 ms is die-seconds across
two queues, not a duty cycle. What it establishes is that the work landed, not
how it was distributed.)

**And it does not help.** Nearly tripling the engine's work made the block-pair
**slower**, by 4.2%. That is the finding that should redirect the plan.

The reason is that ANE work is not free to the GPU. Every piece the island takes
costs Metal at the seams: the fp16 transpose and upload of the activation, the
SwiGLU/transpose seam between gate/up and down, and the eight-partial join at
the end. The island removes 3,072 of 14,336 `fc1` neurons from the GPU — 21% of
the MLP — and hands back three Metal passes over `[S, 5376]` and `[S, 512]`
surfaces to get it. The GPU is the critical path, so work that shortens the ANE
queue while lengthening the Metal queue moves the block the wrong way.

This contradicts the utilisation premise the roadmap rests on. "GPU 16 TFLOP/s
plus 7.6 on the dies gives a 1.48x ceiling" assumes the dies' share arrives free.
It does not: it arrives with a seam, and the seam is GPU work. **The ANE only
pays where its seams cost less than the GPU work it removes.** By that test:

- `qkv`, `attn out` and `fc1` pay. One upload, one join, no intermediate seam —
  which is why the projections-only schedule is still the best measured.
- The MLP island does not. Three seams per piece against 21% of one stage.
- **The planned attention island probably does not either.** The design is
  `QK^T` on the ANE, softmax on Metal, `PV` on the ANE — two Metal seams per
  query tile per head, over `S x S` score planes. That is far more seam traffic
  per unit of arithmetic than the island that just lost 4.2%.

The attention direction worth testing is therefore the **fused** graph — the
question `Tools/ANE/attention-spike.mm` was written to ask, whether the compiler
accepts `QK^T -> softmax -> AV` as one program — because a fused graph has one
seam at each end and nothing in the middle. A three-stage attention island has
already been measured by proxy, and the proxy lost.

## macOS 27.0 (26A5421a): the ABI survives, and the failure mode changed

The machine was upgraded from 26.5.2 (25F84, xnu-12377) to **27.0 (26A5421a,
xnu-13432)** — a major kernel bump — after a day in which the ANE path hard
reset it four times. What the upgrade changed, checked rather than assumed:

| check | result |
|---|---|
| package + tests build (Swift 6.3.3, Xcode 26.6) | clean |
| `h3 doctor` — Metal, model, memory | no problems |
| private ABI, 12 selectors (`Tools/ANE/abi-check.m`) | **all resolve** |
| in-memory MIL accepted by the new compiler | yes, compiles and loads |
| `kANEFAneInstanceHint` still selects a die | yes — 3,865 pairs/s against 3,536-3,806 on 25F84 |
| `-[_ANEClient beginRealTimeTask]` | still entitlement-gated, unchanged |

The version gate now carries a list of audited builds rather than one string.
`26A5421a` is on it, and the comment says exactly what that asserts: selectors
resolve. The trailing letter says GM seed, so the shipping 27.0 build will
differ and has to be re-audited rather than assumed.

### Intermittent submission now fails loudly instead of silently

This is the change that matters. The sequence that panicked 25F84 twice —
`pair-stress 240 2000`, one pair every two seconds — now behaves completely
differently:

| | 25F84 | 27.0 |
|---|---|---|
| 2000 ms gap, 120 pairs | **0 failures**, then a DART panic at +504 s / +583 s | **114 failures**, machine alive |
| no gap, ~38k pairs | 0 failures | 1 failure |

The failures are `ANEProgramProcessRequestDirect() Failed with status=0x12 :
statusType=0x9: Request cancelled`. The watch ran to **+1236 s with zero
ANE/DART events** — more than double both prior reproductions — and the machine
has taken no resets since the upgrade. The driver now **cancels** a request that
arrives during a power transition instead of letting it proceed into an
inconsistent DART mapping. That is the fix the three panic logs were asking for,
and it independently confirms the characterisation built from those logs: the
cancellations track the idle gap exactly, which is what the sleep race predicted
and what the deferred fault was a symptom of.

**The reconstructed root cause.** The old driver's own log names it: a request
arriving during a driver-initiated power transition was met with
`enqueueActionBlock: Skip ... fSleepInProgress: 1`, which discarded the work item
that completes the transition and establishes the address translations — and yet
the request was **admitted**. It then proceeded against a DART whose mappings
were not installed or were being torn down, which produced two symptoms with one
root:

- the gated power-on blocked forever waiting for a completion that had been
  discarded, parking a workloop thread while holding the command gate every ANE
  client needs — the three watchdog resets;
- the device translated against a mapping that was not valid, and the driver
  faulted when it next exercised that DART — the three panics, whose
  *secondary* error is a fault raised while handling a fault, meaning the
  recovery path found the state already inconsistent.

That is: **work admitted onto a die whose translations were mid-teardown.** It
explains the deferral, since the corruption is installed at submission but only
observed when the DART is next exercised — a later power cycle, a page recycle,
another client — which is why it fired while idle with nothing of ours running,
and why no clean run could ever have proven safety. It also explains why
continuous submission was safe on the old OS: with no idle gap the timer never
expires and the window never opens.

27.0 fails closed where 25F84 failed open. The old driver reported all 120
submissions successful and then killed the machine; the new one refuses 114 of
them, and the refusal rate tracks the idle gap exactly.

**This is inference, not a changelog.** It is reconstruction from the old
driver's log lines, the failure timing and the new error code — consistent, but
Apple's source is not visible. A competing reading cannot be excluded: 27.0 may
have more aggressive power gating as a side effect of unrelated work, making the
defect *unreachable* rather than *fixed*. The outcome is the same here; the
difference decides whether it is still reachable by another path, which is a
reason to file it rather than not.

Two consequences:

- **The engine now requires sustained feeding.** It refuses ~95% of work at a
  2 s cadence and 1 in 38,654 at full rate. Irrelevant to a render, which
  submits densely — but it rules out ever adding a low-rate heartbeat, which
  would now be almost pure refusal.
- **A graceful refusal is testable in a way a silent wedge was not.** The bridge
  already falls back to the GPU on a failed submission, so correctness holds and
  the only cost is speed — and the receipts can count it.

### The cancellation is transient, and retrying it removes it entirely (2026-08-27)

**This corrects the two consequences drawn in the section above.** They were
read off a failure rate without asking what happens on a second attempt.

Measured back to back in one session, the documented worst case — one pair every
2000 ms, the cadence that refused 114 of 120 before:

| retry budget | pairs | submission failures |
|---|---:|---:|
| 0 (previous behaviour) | 120 | **108** |
| 8, 3 ms apart | 120 | **0** |

Not reduced — **eliminated**. A cancelled request is refused *before admission*:
nothing was submitted and no surface was touched, which is exactly what "fails
closed" means. The power transition it collides with lasts 8-15 ms, so a handful
of 3 ms retries lands after it. Nothing here forces admission — every refusal is
still honoured, and the retry is an ordinary resubmission — so this does not
reopen the 25F84 panic path.

Two claims above are therefore withdrawn. The engine does **not** "require
sustained feeding", and a low-rate heartbeat is **not** ruled out: a 2 s cadence
is completely usable once the transient is absorbed.

**What this was actually costing.** The linear route died in every render, and
not because renders submit sparsely. At process start the engine is cold, so the
*first* submissions are the ones cancelled — and the slot pool treats any failure
as evidence that the runtime may still own its surfaces, calling `quarantine`,
which sets `poisoned` permanently. Six cancellations inside the first five
seconds retired the route before it had completed a single evaluation, and the
receipts recorded a render with no ANE linear work rather than a failure. The
quarantine is right for a failure that may have left work in flight; it was
being applied to the one failure that provably has not.

### The GPU baseline is unchanged on 27.0

Re-measured because every number in this document was from 25F84 and the engine
share was swept against a GPU rate that a new Metal driver could have moved.
Same recipe as the render gate — 864x480, 124 frames, 20 steps, `balanced`,
single pass:

| | 25F84 | 27.0 |
|---|---:|---:|
| per step, mean | 28.22 s | **28.03 s** |
| full step, median of the 10 that ran the stack | 54.54 s | **54.84 s** |
| sampling | 564.5 s | **560.6 s** |
| whole render | 10m 56s | **10m 46s** |
| branch-steps reused | 10/20 | 10/20 |

Within half a percent, with identical cache behaviour. The GEMM ceiling agrees:

```
qkv 19.3   attn out 19.3   fc1 19.5   fc2 19.1   square 8192^3 19.5  TFLOP/s
attention S=15731 H=56 D=128   417.4 ms   17.0 TFLOP/s
```

against the ~18.6 TFLOP/s the share was tuned to, so the balance point has not
moved and the tuning stands. **`fc1` at 19.5 TFLOP/s on the large-M shape is
also a live check on `patches/mlx-m3-ultra-large-m-gemm.patch`**: that shape is
exactly where unpatched MLX falls 11-18% behind, and `Scripts/check-mlx-patch.sh`
reports it present after the Xcode upgrade re-fetched the SwiftPM checkout —
which is precisely the scenario that would otherwise have dropped it silently.

One methodological note, recorded because it wasted a render: the first attempt
at this baseline was run at `--cfg-scale 5.0` and came out at 114.7 s a step,
which looked like a 2.1x regression and was reported as one. It was two forward
passes per step against the gate's one. The header says
`guidance 5.00 (two forward passes per step)` and it was read past twice. The
gate's numbers are single-pass; anything compared against them must be too.

### What the upgrade does not change

`fc2`'s bound, the seam economics, the 1.478x hardware ceiling, and the fact
that `ComputeUnitKind` in the new `CoreAI` framework is a single `.neuralEngine`
with no die granularity — see *CoreAI is not a route*. Every performance number
in this document is also from 25F84 and is now stale: the engine share was swept
against a GPU sustaining ~18.6 TFLOP/s, and a new Metal driver moving that moves
the balance point. **Nothing here should be quoted as a 27.0 result until it is
re-measured.**

### The multi-weight MLP island executes; 0x1D was a sizing error (2026-08-27)

**This corrects the finding at the top of this document.** The four-input MIL
graph containing both `fc1` matmuls, SwiGLU and `fc2` was recorded as
"compiles and loads but is rejected at inference with ANE status `0x1d`", and
the conclusion drawn was that *"compiler acceptance is therefore not evidence
that this runtime can execute a multi-weight island"*. That conclusion is wrong.

macOS 27 decodes `0x1D`:

```
Code=42 "Inference failed — IOSurface smaller than the model expects
         (re-check inputBufferSize/outputBufferSize from the load reply)"
         underlying=0x1D
```

It is a **buffer-size error**, not an architectural rejection. Oversizing every
surface in `Tools/ANE/mlp-island-spike.mm` first made the same graph run:

| surface multiplier | result |
|---|---|
| 1x, 2x, 3x | `0x1D` |
| **4x** | **`evaluate=OK`** |

That observation was real; its first explanation was not. The allocation-only
`Tools/ANE/surface-layout.m` probe calls the runtime's own
`createIOSurfaceWithWidth:pixel_size:height:bytesPerElement:` factory. It returns
exactly the packed sizes for all three tensors: 16 KiB for `[1,128,1,64]` and
64 KiB for both `[1,128,1,256]` and `[1,256,1,128]`. There is no 256-element
minor-axis floor.

The load reply was already present as `modelAttributes`. It gives both the
answer and the root cause:

```
input symbols: down_weight=0, gate_weight=1, up_weight=2, x=3
sizes:         65536          65536          65536        16384
```

The compiler canonicalized the inputs; it did not preserve the MIL function's
textual `x, gate_weight, up_weight, down_weight` order. The request bound that
textual order to indices `0,1,2,3`, putting the 16 KiB `x` surface in the 64 KiB
down-weight slot. Padding `x` to 64 KiB hid the validation error and evaluated
with `x` and `down_weight` swapped, which explains both the exact 4x boundary
and the uncorrelated output. Binding by the compiled symbol order evaluates at
ordinary 1x allocation.

Numerical conformance is now a separate scale question. The original tiny
fixture drives the fused fp16 intermediates into the engine's flush-to-zero
region; increasing all fixture operands shows the transition cleanly:

| fixture scale | rel-RMS | cosine |
|---:|---:|---:|
| 0.5 | 1.0 (zero output) | 0 |
| 1 | 0.248 | 0.9688 |
| 2 | 0.0575 | 0.9984 |
| 4 | 0.0127 | 0.9999 |
| 8 | **0.00160** | **0.999999** |

The fused dynamic-weight graph therefore executes and produces conformant
arithmetic away from underflow *at the tiny shape*. The old architectural and
layout conclusions are both retired.

**At production width it is the same underflow, and the fixture was the
problem.** Widening the contraction to `hiddenSize` 5,376 with a bounded 512-
neuron island reads rel-RMS **0.0395** at the default fixture — ten times
outside the 0.003 class. `H3_MLP_INTERNAL_SCALE`, which is arithmetically
neutral, moves that only from 0.03953 to 0.03934, and an earlier version of this
section concluded from that a second, unidentified fault. That was wrong: the
internal scale was swept, the *operand magnitudes* were not. Sweeping
`H3_MLP_FIXTURE_SCALE` at width:

| fixture scale | reference max abs | rel-RMS | cosine |
|---:|---:|---:|---:|
| 0.25 | 1.7e-4 | 1.0 (zero output) | 0 |
| 0.5 | 1.1e-3 | 0.205 | 0.979 |
| 1 | 6.8e-3 | 0.0395 | 0.9992 |
| 2 | 4.0e-2 | 0.00648 | 0.99998 |
| **4** | 0.21 | **9.06e-4** | 1.0 |
| **8** | 3.9 | **4.35e-4** | 1.0 |

Monotonically decreasing in operand magnitude, which is the flush-to-zero
signature; saturation would run the other way. It is the same fault as the tiny
shape, shifted, and there is no second defect.

What this exposes is the fixture. At scale 1 the largest reference output is
**0.0068** — not a magnitude any real DiT activation takes. At scale 4 to 8 the
fused island is inside the equivalence class. The blocking question is therefore
**not** "why is it wrong at width" but "what magnitude do real MLP inputs
have", and a captured production tensor answers it. Until that capture exists
the chained island stays the production path, but the fused island is a live
candidate rather than a closed one.

### Fused attention works with an explicit softmax (2026-08-27)

**This supersedes the section below it.** The fused graph's 97% error is the
MIL `softmax` op, not the arithmetic, and replacing it with an explicit stable
reduction fixes it at production scale.

Two experiments, in order. First, `Tools/ANE/scores-spike.mm` emits `QK^T`
alone — no softmax, no AV — so a wrong answer would mean the *matmul* fails at
attention's N and rebuilding softmax would be wasted work:

| keys x queries | matmul rel-RMS |
|---|---:|
| 512 | 2.572e-4 |
| 2,048 | 2.569e-4 |
| **15,744** | **2.568e-4** |

Flat across a 30x range, at fp16 round-off. The matmul is exact at the
production plane. (Two things that cost time and are worth recording: the score
plane comes back **transposed** relative to what the MIL types declare, so both
readings must be checked — an uncorrelated `sqrt(2)` rel-RMS is the signature of
the wrong one; and the runtime requires all inputs to share a shape, so a graph
with `q` at [T,D] and `k` at [S,D] is refused with 0x1D whatever N is.)

Second, the softmax itself, replaced by:

```
m = reduce_max(scores, key axis)      w = exp(scores - m)
d = reduce_sum(w, key axis)           y = matmul(w, v) / d
```

| | S=512 | S=15,744 |
|---|---:|---:|
| fused `softmax` op | 0.0217 | **0.974** |
| **explicit reduction** | **8.75e-4** | **8.99e-4** |
| bf16 GPU reference class | 1.66e-3 | 1.66e-3 |

**Better than the GPU's own bf16 path, at the production sequence.** The
reductions do not tile wrongly; only the fused op does. The earlier diagnosis —
"a softmax normalised per tile" — was right about the mechanism and wrong to
treat it as unfixable.

Cost: 73.96 ms a head against 62.70 for the fused version, 1.716 TFLOP/s, and
compile rises from 199 ms to 3.8 s (one-time per shape, cacheable). Balancing
against the GPU's 7.45 ms a head moves about **9 of 56 heads** onto the dies,
attention 417 ms to ~347 ms, and the render from 1.179x to roughly **1.28x**.

It also reopens the ceiling. `A/G` is no longer fixed, so the branch-local model
`forward = A/G + L/(G+R)` no longer bounds this, and the coarse two-branch
pipeline is worth costing.

### The head-count cliff, and what it caps the route at (2026-08-27)

The projection above — "about 9 of 56 heads" — is wrong, and the reason is a
hard boundary rather than a balance point. A graph's cost per head is flat up to
four heads and then collapses:

| heads in one graph | S=15,744 | S=11,000 |
|---|---:|---:|
| 1 | 1.70 TFLOP/s | 0.49 |
| 2 | 1.88 | 0.48 |
| **4** | **1.90** | 1.69 |
| 5 | **0.20** | 1.63 |
| 8 | — | 0.25 |

Confirmed independently through the integrated backend against a real capture:
5, 6 and 7 heads a die take 7.7 s, 7.7 s and 11.2 s a call against 374 ms at
four.

**It is not a capacity limit, and the obvious hypothesis is disproven.** A 2 GiB
score-plane ceiling predicts the S=15,744 column exactly (4 heads = 1.847 GiB
fast, 5 = 2.309 GiB slow) and then fails: 1.803 GiB at S=11,000 is *slow* while
1.847 GiB at S=15,744 is *fast*. Nor is it monotone in heads — at S=11,000 one
and two heads are slow and four and five are fast. It is a per-shape tiling
decision inside the compiler, and no rule covering both columns is established.

### The render gate for routed attention: 1.042x end to end (2026-08-27)

Both arms rendered fresh in one session, same prompt, seed, shape and binary
lineage: 864x480x124, 20 steps, seed 0, cache threshold 0.1.

| arm | per step | full step | sampling | vs GPU |
|---|---:|---:|---:|---:|
| `sdpa`, GPU only | 28.34 s | 55.28 s | 566.8 s | — |
| `ane` attention, 4 heads a die | 27.20 s | 51.97 s | 544.0 s | 1.042x |
| `ane` attention, repeat | 27.21 s | 52.08 s | 544.3 s | 1.042x |
| **attention + linear** | **24.48 s** | **46.28 s** | **490.6 s** | **1.158x** |

Attention alone is **1.042x end to end, 1.063x on the ten steps that run the
stack**, reproducing to 0.04% across two renders. With the linear route also
alive — which it had never been in a render before the cancellation fix — the
pair is **1.158x end to end and 1.194x on full steps**, with zero submission
failures and `aneRoutedProjections = [attn out, fc1, qkv]`, nothing declined.

Attention alone is far below the 1.16-1.18x that route measures in isolation,
and it should be. Attention is about 37% of DiT FLOPs and the head cliff caps the dies at 8 of
56 heads, so Amdahl gives `1/(0.63 + 0.37/1.156)` = 1.053x against a measured
1.063x on full steps. The route behaves as its own micro-benchmarks predict:
there is no hidden loss here, and no hidden upside either.

Routing was verified live rather than inferred. `H3_ANE_ATTENTION_TRACE=1`
reports every hundredth routed call and the first decline of each kind; the
render logged calls 1 through 400 with **zero declines and zero evaluate
failures**. An earlier arm was discarded before it was believed: it had been run
with `H3_ANE=experimental`, which also enables the linear primitive, and that
route failed six times with `Request cancelled` and poisoned its session. The
attention gate is now `H3_ANE_ATTENTION` alone, because these two routes share
no code and coupling them silently confounded the measurement.

**The linear route failing closed under render load is the larger finding here.**
It is the path carrying the banked 1.179x, and it needs its own investigation.

### The sequence is a lottery, and 64-alignment is not the ticket (2026-08-27)

The head count is not the only shape term that decides throughput. At a fixed
H=1, sweeping the sequence:

| S | TFLOP/s | S | TFLOP/s |
|---:|---:|---:|---:|
| 11,000 | 0.49 | 14,000 | 1.52 |
| 11,008 | 0.49 | 15,000 | 1.43 |
| 11,072 | 0.49 | 15,360 | 1.72 |
| 12,000 | 1.60 | **15,731** | **0.41** |
| 12,288 | 1.62 | **15,744** | **1.71** |
| 13,000 | 0.94 | | |

**Thirteen elements of sequence separate a 4.1x difference.** Nothing tidy
predicts it:

- **Not 64-alignment.** This was the standing explanation for 15,731 against
  15,744 and it is refuted directly: S=11,008 and S=11,072 are exact multiples
  of 64 and run at 0.49, indistinguishable from unaligned S=11,000's 0.49
  (125.79 ms against 125.64 ms). Padding an unaligned sequence to a multiple of
  64 buys nothing by itself.
- **Not capacity.** Larger sequences are *faster*: 15,744 beats 11,072 by 3.5x
  on a plane twice the size.
- **Not the power of two dividing S.** 11,072 = 64x173 is slow; 12,000 = 32x375
  is fast.

What this means for production is a **risk, not a tuning knob**. Any change that
moves the sequence — resolution, duration, a different padding rule — can land
on a slow shape with no warning and no way to predict it from the shape alone.
15,731 is exactly such a shape, and it is a plausible unpadded production
sequence.

The defensible response is not to explain the compiler but to **measure at
session creation**. The backend already pays a one-time per-shape compile; a
calibration evaluation against the GPU's expected cost for the same heads would
let it either accept the shape, pad to a *measured-fast* neighbour and mask, or
decline to SDPA. That rehabilitates padding as a remedy chosen by measurement,
rather than as a rule derived from divisibility — which the table above shows
does not hold. It is not built yet; today the route is validated only at the
shapes recorded here.

What production needs is settled: **four heads a die at the production
sequence**, measured in two harnesses. That caps the dies at 8 of 56 heads —
14% of attention — which is why the route lands where it does and why it does
not tune higher. `H3_ANE_ATTENTION_HEADS` re-sweeps it if the shape or the OS
moves; nothing else should touch the constant.

### Superseded: the fused attention graph is accepted, and numerically wrong

The last open approach. Attention is 37% of the block and the one term the
ceiling model cannot overlap — `forward = A/G + L/(G+R)` — so it is the only
place left that could move the number that dominates. A fused
`QK^T -> softmax -> AV` graph was the only shape worth testing, because its
seams are one at each end rather than one between every stage.

At production S the graph **compiles, loads and evaluates**:

```
compile_load=ACCEPTED   evaluate=SANE   single_ms=62.703   2.025 TFLOP/s
explicit_io=15.375 MiB  hypothetical_exposed_scores=0.462 GiB
```

The seam economics are genuinely good, and they are what the island lacked:
15.4 MiB crosses per head while the 0.462 GiB score plane never leaves the
engine. Throughput is poor — 2.02 TFLOP/s a die against 5.56 on linears, and
8.4x slower per head than the GPU — but balancing heads would still have moved
~11 of 56 onto the dies, taking attention from 417 ms to ~337 ms and the render
from 1.179x to roughly **1.30x**, using capacity that is otherwise idle.

**It is wrong.** Against an fp32 reference over the same tensors:

| | S=512 | S=15,744 |
|---|---:|---:|
| key_axis rel-RMS | 0.0217 | **0.974** |
| query_axis rel-RMS | 0.0300 | **0.974** |

Both orientations, so it is not a transposed lowering. The reference is
validated: at S=512 it reproduces the spike's own scalar `ReferenceError` to six
figures (`Tools/ANE/attention_reference.py`, chunked fp32, which also lifts the
S<=512 cap that left this unmeasured).

**The cause is not the engine's arithmetic.** ANE linears measure 7e-5 to 5e-4,
better than the bf16 GPU path's 1.66e-3. Precision decay would sit near 1e-4 and
grow gently with contraction depth; this jumps 45x for a 31x change in `S`, with
the same graph and the same arithmetic, to a value that is uncorrelated rather
than imprecise. That is a wrong **normalisation**. A 0.462 GiB score plane must
be tiled, and a softmax normalised per tile instead of across the full row
produces exactly this — while at S=512 the row fits one tile, normalisation is
correct, and 2.2% is just fp16 `exp` with no 1/sqrt(d) scale.

So attention is closed from both directions, and the two failures are
complementary rather than coincidental:

| design | seams | softmax |
|---|---|---|
| fused | cheap — the score plane never crosses | **broken** once the plane tiles |
| three-stage, Metal softmax between | correct | **seam cost**, measured negative by proxy at 0.958x |

Correct normalisation or cheap seams; not both. Nothing here is tunable into a
win, and `A/G` stays on the GPU.

One process note. `evaluate=SANE` passed while the output was 97% wrong: that
check tests finite and non-constant, nothing more. This is the third time in
this document that wrong answers have presented as healthy output — after the
saturation cliff's silent zeros and an `expectedHidden` that made the island
decline every block while six conformance tests stayed green.

### CoreAI is not a route

macOS 27 ships `/System/Library/Frameworks/CoreAI.framework`, an umbrella
re-exporting `CoreAIDelegates` over six subframeworks (`CoreAIRuntime`,
`CoreAICompiler`, `CoreAIDelegates`, `CoreAICommon`, `CoreAICache`,
`CoreAIAsset`). It is pure Swift — no ObjC classes, no exported C symbols — and
its API is a real generation beyond CoreML: `AIModel` loading **named
functions**, `SpecializationOptions`, `CompilationDelegates` with third-party
`delegatePlugins`, a `Profiler`, an `IntermediateLogger`.

It is not usable here, for two reasons:

- **Not linkable.** Nothing in the SDK — no header, no `.tbd`, no
  `.swiftinterface`, and none on the system either. Its Info.plist reads
  `DTSDKName = macosx27.0.internal`.
- **Wrong granularity.** `ComputeUnitKind` is `cpu | gpu | neuralEngine`. One
  neural engine, not two dies. That is the same abstraction level CoreML gave
  us and the one this work has to go beneath, since everything here depends on
  `kANEFAneInstanceHint` running both dies concurrently alongside Metal.

The delegate plugin architecture is worth watching. If Apple opens it, it is the
sanctioned version of what this bridge does by reverse engineering.

## Machine safety: fixed on 27.0, real on 25F84 and earlier

**Scope: this hazard applies to macOS 26.5.2 (25F84) and earlier. It does not
reproduce on 27.0.** The full forensic record — four hard resets, three kernel
panics, the hypotheses tested and eliminated — is in the git history around
2026-08-27 rather than here, because it describes a defect that current builds
do not have and would misread as a live warning.

**What it was.** A request submitted to a die during a driver-initiated power
transition was *admitted* rather than rejected. The driver's own log names the
moment: `enqueueActionBlock: Skip ... fSleepInProgress: 1` discarded the work
item that completes the transition and establishes the address translations, and
the request proceeded anyway against a DART whose mappings were absent or being
torn down. One root, two symptoms — a gated power-on blocking forever on a
discarded completion (kernel hang, watchdog reset), and a translation fault
whose *secondary* error means the recovery path found the state already
inconsistent (`sptm_t8110dart_clear_err: dart-ane0 ... 0x80080008`).

The fault was **deferred eight to ten minutes** and fired while the machine was
idle, because the corruption is installed at submission but only observed when
the DART is next exercised. That is why no clean run could ever have proven
safety, and why continuous submission was always safe: with no idle gap the
five-second timer never expires and the window never opens.

**What changed.** 27.0 fails closed where 25F84 failed open. The same sequence
now returns `ANEProgramProcessRequestDirect() ... Request cancelled` for ~95% of
submissions at a 2 s cadence, and 1 in 38,654 at full rate — the refusal rate
tracks the idle gap exactly. See *Intermittent submission now fails loudly*.

Whether that is a targeted repair or more aggressive power gating making the
defect unreachable is not something the outside of the driver can distinguish.
Either way it is incidental to us, which is the reason for the rule below.

### The rule this leaves

- **Re-run the regression test on every OS bump.** `Tools/ANE/pair-stress.m`
  submits one pair every two seconds for four minutes; then watch for twenty.
  Reproductions landed at +504 s and +583 s, so nothing may be called clear
  before +900 s — a watch was stopped at +553 s once and the machine panicked
  30 seconds later.
- **Only audited builds route.** The version gate carries a list, and a build
  earns its place by the selector audit *and* a clean pair-stress watch. A GM
  seed's build string differs from the shipping one and has to be re-audited.
- **Sustained submission is the safe pattern and the fast one.** Intermittent
  poking was what broke 25F84 and is what 27.0 refuses. A render submits
  densely; nothing should add a low-rate heartbeat.

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

### Two shipping constants were calibrated on a machine that no longer exists (2026-08-28)

Re-swept end to end on the fixed arm once attention was also routed. Both
defaults moved, and the reason is the same in each case: attention now occupies
both dies for about 276 ms a block, so the engine has far less spare capacity
than when either constant was chosen.

**The engine's share of the columns**, `postShare`:

| share | per step | full step | ANE busy |
|---:|---:|---:|---:|
| 0.30 | 24.69 s | 45.83 s | 160.9 s |
| **0.375** | **23.95 s** | **44.30 s** | 176.4 s |
| 0.45 *(old default)* | 24.22 s | 44.66 s | 194.8 s |
| 0.52 | 25.44 s | 47.07 s | 221.0 s |

Engine busy time keeps climbing across that range while wall time turns around:
past the balance, each extra column costs more in GPU-side fp32 partial summing
than it saves on the engine. That is the same shape the original calibration
found, and 0.30 and 0.52 are firmly outside it.

**But the apparent winner here does not survive being paired with the split, and
this table is why the two must be swept together.** The sweep above ran at 8
pieces. At 4 — the split that now ships — the order reverses:

| share, at split 4 | full step |
|---:|---|
| **0.45** | **43.43, 43.43** |
| 0.375 | 43.58, 43.52, 43.61 |

Non-overlapping across five runs, so 0.45 stands and **the default was right all
along**. Fewer pieces means less partial summing per column, so the engine can
profitably take more of them: the optimum share rises as the split falls. An
earlier version of this section swept the two independently, combined the
winners, and moved the share to 0.375 on the strength of it. That is not a valid
composition, and only the split change survives.

**The contraction split**, at share 0.375:

| split | per step | full step | ANE busy |
|---:|---:|---:|---:|
| **4** | **23.95 s** | **43.98 s** | 176.5 s |
| 8 *(old default)* | 23.95 s | 44.30 s | 176.4 s |
| 16 | 31.81 s | 59.32 s | 293.9 s |
| 32 | 54.89 s | 103.86 s | 595.4 s |

**16 and 32 are worse than not routing at all**, and not because of reduction
cost: engine busy time triples and then quadruples for identical work, so the
engine itself slows down. At 16 pieces `qkv` and `fc1` contract over `k=336`,
too thin for the reuse this hardware needs. This is the third shape cliff
recorded here — heads per attention graph, sequence length, and now contraction
pieces — all with one signature: flat, then catastrophic, with no rule that
predicts the edge.

`qkv` keeps its own share but no longer wants a different value. It was 0.286 on
the reasoning that nothing covers it, since it gates attention; with attention
itself partly on the dies that no longer holds, and the full step rises
monotonically as it drops — 43.55 s at 0.375, 44.38 at 0.286, 45.10 at 0.20.

**On the precision of these numbers.** Two runs of one config gave 23.97 and
23.27 s a step, so the per-step mean carries about 3% of run-to-run spread and
the 0.375-against-0.45 margin sits inside it. The full step repeats far better
(43.58 against 43.52) because it excludes the cached steps, so it is the metric
to compare on. What the sweep establishes firmly is the direction and the
cliffs, not the third significant figure.

## What the whole route can be worth, and why it is not 1.5x

Measured on the fixed arm — 864x480x124, 20 steps, seed 0, cache 0.1, one
prompt — decomposed from the arms themselves rather than assumed:

    block, GPU only        1105.6 ms  =  attention 401.8  +  linear 703.8
    attention routed floor  344.4 ms     (GPU keeps 48 of 56 heads)

The model is calibrated, not fitted: it predicts an attention saving of 57.4 ms
a block against 64.0 measured.

The engine's rate is the term that decides everything, and it is already known:
`ANEFormTests` swept the decomposition by output column, by sequence tile, and
the rectangles between, and found **7.7 to 7.95 TFLOP/s across both dies, never
exceeded**. The one candidate for a faster lowering — the 1x1 convolution, the
engine's native form — is bit-identical and 2.6x slower. So `R = 7.9` is a wall,
not a setting.

| R (both dies) | ideal linear | full step | overall |
|---:|---:|---:|---:|
| **7.9 (measured wall)** | 497.9 ms | 42.1 s | **1.303x** |
| 9.0 | 478.4 ms | 41.1 s | 1.332x |
| 11.1 | 445.1 ms | 39.5 s | 1.387x |

**So the ceiling on this arm is about 1.30x, and 1.5x is not reachable by moving
work to the dies.** Perfect sharing of every routable projection still lands
short, because 1.5x would need a 36.4 s full step against a best-possible 42.1.
Reaching it would take a faster engine path, a cheaper GPU baseline, or work
that is genuinely parallel rather than merely relocatable — at `cfgScale 1`
there is no second branch to pipeline, which is what the CFG section below
assumes.

Against that ceiling the shipping route is at **1.176x**, having captured
**69%** of the ideal linear saving (142 of 205.9 ms a block). The remaining
64 ms a block is worth about 1.31x, and it is scheduling and seam cost rather
than engine rate.

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

### The contraction axis, which nothing had swept

Every measurement on this page cut the engine's work by **output column** or by
**sequence tile**. Tensor parallelism splits either of those *or the
contraction*, and the contraction is the one axis this bridge had never tried.
It is where the engine was hiding:

| k | TF/s, both dies | ms |
|---|---:|---:|
| 1344 | **20.61** | 12.6 |
| 2688 | 14.93 | 34.8 |
| 5376 (`qkv`, `fc1`) | 7.80 | 133.4 |
| 7168 (`attn out`) | 7.81 | 177.6 |
| 14336 (`fc2`) | 3.98 | 697.1 |

The rate roughly halves each time `k` doubles past ~2688, which is the
signature of an engine spilling its accumulator and re-streaming rather than one
that is compute-bound. So do the split by hand — chunk the contraction, run each
chunk, sum the partials. Both operands are `k`-major in the engine's own layout,
so a chunk is a contiguous row range of each and costs no gather.

Checked against an fp32 MLX reference, at every production contraction:

| k | chunks | ms | TF/s | rel RMS vs fp32 |
|---|---:|---:|---:|---:|
| 5376 | 1 | 134.7 | 7.72 | 1.695e-04 |
| 5376 | 4 | **50.3** | **20.69** | **8.743e-05** |
| 7168 | 1 | 178.0 | 7.79 | 2.526e-04 |
| 7168 | 8 | **55.7** | **24.89** | **9.162e-05** |
| 14336 | 1 | 695.5 | 3.99 | 2.522e-04 |
| 14336 | 8 | **135.5** | **20.47** | **9.033e-05** |

2.7x to 5.1x faster, and **more accurate every time it is split** — summing
partials in fp32 is a better accumulation than one long fp16 chain, so the error
falls as the chunks shrink. At 9e-05 the routed path is roughly 19x closer to
fp32 than the bf16 GPU path it replaces.

**This inverts the premise of everything above.** The engine was 0.41 of the
GPU; split, it is 20–25 TFLOP/s against the GPU's 19.4, which makes the two
devices peers. The heterogeneous ceiling that closed the CFG overlap and the
query tiling — `(rate_GPU + rate_ANE) / rate_GPU`, 1.41x — is not 1.41x any
more.

Recomputing the block, with attention and the elementwise pinned to the GPU:

| | ceiling | |
|---|---:|---:|
| today, 7.8 TF/s, `fc2` excluded | 833 ms | 1.40x |
| split-k at 20 TF/s, `fc2` excluded | 643 ms | 1.82x |
| split-k at 20 TF/s, `fc2` included | **575 ms** | **2.03x** |

Two things that were settled are now open again, and both should be re-run
before anything is built:

  * **`fc2` is still not routable**, and splitting does not change that. The
    tool now computes the split bound, but the oracle captures it needs are not
    in this checkout, and the coverage objection that kept `fc2` out survives
    the better margin anyway. See *Splitting the contraction does not settle
    `fc2` either*.
  * **The native merge kernels may still be useful.** Split-k produces `c`
    partials per projection to sum, which is what an accumulating merge kernel
    is for. Measured, the summation is not currently the binding cost — a split
    `qkv` projection is 122.4 ms against an engine time of 50.3 — so this is an
    optimisation and not the gate.

None of this is implemented in the routed path. What is measured is the engine's
rate and its arithmetic; the block ceiling above is arithmetic, not a
measurement, and the two schedules already closed on this page are a standing
reminder of the distance between those.

### Split-k, implemented and measured

The three routed projections cut their contraction: `qkv` and `fc1` into four
pieces of 1344, `attn out` into eight of 896. Both operands are already
`k`-major, so a piece is a contiguous byte range of the packed activation and of
the uploaded weight shard — an offset per piece, not a gather — and the
activation surfaces total exactly what one contraction needed. Partials are
summed in fp32 before the operand scale comes off, which is why the split is
*more* accurate as well as faster.

**The share had to move with it, and neither half works alone.** 0.286 is the
point where the GPU shard and the engine finish together *for an engine at 0.40
of the GPU's rate*. Split, the engine is at parity and the balance moves. One
production block, one configuration per process:

| configuration | block | |
|---|---:|---:|
| GPU only | 1133.6 ms | — |
| whole contraction, share 0.286 — the previous default | 1037.6 ms | 1.093x |
| whole contraction, share 0.45 | 1287.5 ms | 0.88x |
| split-k, share 0.286 | 1068.8 ms | 1.06x |
| **split-k, share 0.45 — the new default** | **~1010 ms** | **~1.12x** |
| split-k, share 0.52 | 1061.5 ms | 1.07x |

Row three is the control that matters: **raising the share without splitting
costs 250 ms.** The gain is the split, not the retune, and 0.286 looked like a
wall for as long as it did because moving it required a faster engine that
nobody had gone looking for.

The plateau runs 0.40 to 0.50 — 1005.7, 1007.4, 1010.6, 1015.8 — with a cliff
just past it. The default is 0.45, mid-plateau, and it **follows the split**:
`H3_ANE_SPLIT_K=1` restores 0.286 along with the whole contraction, because a
split contraction left at 0.286 is slower than the unsplit path it replaced.

At the projection the win is far larger than at the block: `qkv` at share 0.5 is
122.4 ms split against 245.7 whole and 196 on the GPU alone — 1.60x. The block
sees only ~1.12x because **the engine can overlap nothing but its own GPU
shard.** Attention, `fc2` and the elementwise are 643 ms of the block it cannot
touch.

### Both schedules stay closed, now for a better reason

Query tiling and the CFG overlap were closed on the grounds that the engine was
too slow to fit in the window it was offered. That excuse is gone — at post
share 0.85 the engine's `out`+`fc1` work is 261 ms against a 434 ms attention
window — so both were re-run against a split engine at parity:

| schedule | best | against untiled split-k |
|---|---:|---:|
| query tiling, T=4, post 0.50 | 1042.0 ms | +3.6% |
| query tiling, T=4, post 0.70 | 1047.9 ms | +4.2% |
| query tiling, T=4, post 0.85 | 1109.1 ms | +10.3% |
| CFG pipeline, share 0.50 | 2050.0 ms a pair | +2.0% |
| CFG pipeline, share 0.65 | 2172.8 ms a pair | +8.1% |

Both still lose, and the CFG pipeline's 1.111x at share 0.65 is again a gain
over a serial arm that is itself worse. **So the limit was never the engine's
rate.** It is that one MLX thread has to run every upload, gather, norm and
collect between engine jobs, and the dependency chain inside a block gives it
nowhere else to be. A faster engine made the schedules cheaper to feed and did
not make them pay.

### Fusing the join, and the gate

Splitting leaves `splits` partials per die to sum. Written as ordinary MLX that
is `2*splits` elementwise operations and MLX materialises every one: at a share
of 0.45 each fp32 intermediate is 306 MB, and the chain moves about 5 GB a die
per projection against the 0.77 GB the arithmetic requires. Across both dies and
three projections, 30 GB a block — most of what splitting had won.

Compiled, the accumulate, the unscale and the cast fuse into one pass: read
`splits` fp16 partials, write one bf16 result, accumulate in fp32 inside the
kernel so the precision the split buys is kept. `qkv` at share 0.5 went from
122.4 ms to **108.0 ms**, and the block from ~1010 to **969.4 ms**.

Two passes of the three-way comparison, one configuration per process:

| | pass 1 | pass 2 | |
|---|---:|---:|---:|
| GPU only | 1133.1 ms | 1139.1 ms | — |
| whole contraction, share 0.286 | 1038.5 ms | 1043.5 ms | 1.091x |
| **split-k + fused join, share 0.45** | **972.2 ms** | **966.5 ms** | **1.172x** |

The gate wants 987.9 ms against this harness's 1136.1 ms baseline. 969.4 clears
it with 19 ms in hand. The share plateau moved with the cheaper join and now
runs 0.45 to 0.52 — 968.5 and 960.7 — falling off at 0.60 (1012.1) and 0.68
(1043.0), so 0.45 keeps the margin from the cliff.

### Routing `fc2`: per-block bounds, and the machinery for a per-block scale

`fc2` was refused on a single global bound — 4,761,873 at block 45, 9.1x over
the cliff — and that refusal was right for a single global scale. It is the
wrong instrument for routing, because the bound is not remotely uniform across
blocks.

The probe now records a per-block entry alongside the global worst, from the
same GEMM (`SaturationProbe.record`, `alsoLabel:`). A partial run — 3 of 20
faithful steps, preserved at `docs/bench/fc2-bound-partial.json` — already shows
an **813x spread** across blocks, and the required scales fall out as:

| required operand scale | blocks |
|---|---:|
| 1 (none needed) | 15 |
| 1/2 to 1/8 | 17 |
| 1/16 — the shipping scale | 9 |
| 1/32 to 1/128 | 9 |

**41 of 50 blocks need 1/16 or milder**, which is no more aggressive than what
`qkv`, `attn out` and `fc1` already ship with. That is the case for routing
`fc2` per block rather than refusing it wholesale: scaling everything by what
the worst block needs would push the quiet blocks into the fp16 denormals the
next section pins, and scaling everything by 1/16 leaves the loud blocks
breaching the cliff as silent zeros. Per block, both ends are safe.

**These numbers are a lower bound, not a calibration.** They come from 3 steps
of 20, and the completed run in *`fc2` is refused* reached 4,761,873 where this
partial one is at 2,069,736. A shipping table needs the full render. The
refusal rule matters more than the table: any block whose required scale is more
aggressive than the underflow floor allows stays on the GPU, and a partial bound
can only understate what that is.

The plumbing is in and unmeasured:

- `ANELinearBackend.start(scale:)` threads a per-call operand scale through the
  upload, the join, and the reducer cache — which is now keyed on the scale, so
  a reducer compiled for one block cannot silently apply the wrong magnitude to
  another.
- The native merge path is gated on the default scale. It has no unscale and the
  native pack has no scale, so the pair is self-consistent only there; a
  per-block scale taking that path would have its upload scale dropped and
  breach the very bound it was chosen to clear, silently.

What is still missing: the table generator, the routing entry point with its
refusal rule, a conformance test per routed block, and any measurement at all.

### The operand scale has an underflow floor

Chasing the split's accuracy claim turned up something older. Every routed
activation is multiplied by 1/16 before the engine sees it, to move the
partial-sum envelope off the 2^15 cliff. fp16's smallest normal is 6.1e-05, so
scaling down far enough pushes the **products** into denormals and the engine
loses them:

| operand sigma | engine sees | typical product | bf16 GPU | routed |
|---:|---:|---:|---:|---:|
| 1.00 | 0.0625 | 6.3e-02 | 1.66e-03 | 1.68e-03 |
| 0.20 | 0.0125 | 2.5e-03 | 1.66e-03 | 2.13e-03 |
| 0.05 | 0.0031 | 1.6e-04 | 1.66e-03 | **3.10e-02** |

This is the **shipping path**, not the split: whole and split degrade together
and identically, and at production magnitudes the split is very slightly the
better of the two. `ANE_PRECISION_RESULTS` measured 7e-05 to 5e-04 on real
captured activations, which sit far above the floor — what was missing is any
statement of where the floor is, or any test holding it there. A projection
whose activations ran near sigma 0.05 would be quietly wrong by 3%, and nothing
downstream in this tree would catch it.
`operandScaleHasAnUnderflowFloor` pins it.

### The render gate, run end to end

Two full renders, same prompt, same seed, same recipe — 864x480, 124 frames, 20
steps, `balanced` (the shipping cache at threshold 0.1). The only difference is
`H3_ANE=experimental`.

| | GPU only | split-k engine | |
|---|---:|---:|---:|
| **full step, median of the 10 that ran the stack** | **54.54 s** | **46.28 s** | **1.179x** |
| mean seconds a step | 28.22 s | 23.90 s | 1.181x |
| sampling | 564.5 s | 479.0 s | 1.179x |
| whole render | 10m 56s | **9m 27s** | 1.157x |

**1.179x on the figure the block gate predicted 1.172x for** — the synthetic
harness called the real render to within half a percent. End to end it is
1.157x, diluted by a decode that does not change and a prompt pass that does not
either. Both runs reused 10 of 20 branch-steps, so the cache did the same thing
on both sides.

The receipts tell them apart, which is what recording arithmetic is for:

    gpu   routed []                     split 0
    ane   routed [attn out, fc1, qkv]   split 8

Quality is judged by the oracles, not by goldens — the arithmetic differs by
construction, so a golden comparison would mean nothing:

| oracle | GPU only | split-k engine |
|---|---|---|
| face / flash | stable, 0 flash events | stable, 0 flash events |
| speech | PASS, WER 0.00, 5/5 keywords | PASS, WER 0.00, 5/5 keywords |
| lip sync | **FAIL**, offset -250 ms, margin +0.015 | **FAIL**, offset -250 ms, margin -0.010 |
| coherence against the GPU render | reference | dssim **0.0051**, accel 1.029, detail 0.979 |

Read the dssim first: 0.0051 is below the 0.01 line, so the two arms rendered
the same scene and the other ratios compare quality rather than content. accel
1.029 is under the 1.1 that means warping; detail 0.979 is over the 0.95 that
means blur.

**Lip sync fails on both arms, identically, and that is not a regression.** It
is what a 20-step balanced render of this prompt does. It is recorded because an
oracle that fails on the control cannot be used to pass the treatment, and
quietly dropping it would throw away the reason for having oracles at all.


### `fc2` is refused, and the oracles were optimistic by 4.9x

The captures are on `big_daddy` — all nine, `b{00,24,49}_c{000,013,019}` — so
the real bound ran, with `--splits`, against the real checkpoint. Unsplit it
reproduces the published figures exactly, `fc2` at 978,586 and 0.5x of headroom.
Split eight ways it clears with room:

| projection | whole `k` | split 8 |
|---|---:|---:|
| `qkv` | 5,132 (102x) | 729 (720x) |
| `attn out` | 69,912 (7.5x) | 19,358 (27x) |
| `fc1` | 2,428 (216x) | 366 (1432x) |
| **`fc2`** | **978,586 (0.5x, NOT PROVEN)** | **140,221 (3.7x, PROVEN)** |

On the captured blocks, splitting turns `fc2` from refused into proven. **It is
still refused, because the captured blocks are not the problem.**

The tool's own closing note is that it holds three blocks of fifty and 1024 of
15,406 rows, and that a bound over a sample is evidence rather than proof.
Nothing has to be captured to close that: what the bound needs is one GEMM on
the magnitudes, and every activation it wants exists at the moment the
projection runs. `DiTBlock.SaturationProbe` computes it inline —
`H3_ANE_BOUND=path h3 render`, engine off — over **every block, every row, every
step**. It is checked against hand arithmetic first: with unit operands the
whole-`k` bound is exactly `k` and a piece is exactly `k/8`.

One render, 50 blocks, 15,406 rows, 20 faithful steps:

| projection | whole `k` | split 8 | worst block |
|---|---:|---:|---:|
| `qkv` | 6,406 (82x) | 915 (573x) | 46 |
| `attn out` | 133,365 (**3.9x**) | 54,411 (9.6x) | 44 |
| `fc1` | 2,475 (212x) | 393 (1334x) | 49 |
| **`fc2`** | **4,761,873 (0.1x)** | **4,520,180 (0.1x)** | **45** |

**`fc2`'s real worst is 4,761,873 at block 45 — 4.9x worse than anything the
oracles ever saw, at a block they never captured.** It exceeds the cliff by 9.1x
whole and by 8.6x split, because splitting only helps when the magnitude is
spread across `k`: at block 49 it bought 7x, at block 45 it buys 1.05x. That is
the adversarial case, and it is the one that decides.

So `fc2` stays on the GPU, the original refusal was right, and the reason it was
right is not the reason anyone gave. The three-block sample was optimistic by
almost a factor of five and missed the worst block entirely.

Two things fall out of the same run. `attn out` — which *is* routed — has only
**3.9x** of headroom whole, not the 7.5x the oracles reported; it is still
proven under any accumulation order, and **splitting nearly triples it to
9.6x**. The contraction split is therefore a safety argument as well as a speed
one, on the projection that sits closest to the cliff.

### `fc2` routes, on a per-block scale, and is worth 1.6% (2026-08-27)

The refusal above was right, and the condition it named — "a measured per-block
bound rather than an operand scale that is merely probably enough" — is now met.
A faithful bound render on this arm puts `fc2`'s worst at **4,573,078 at block
45**, 9x past the cliff at the shipping 1/16 scale.

What makes a scale work where splitting does not is the spread: the bounds vary
**790x across blocks**, from 5,781 at block 09 to 4,573,078 at block 45. Only one
block needs 1/512 and fourteen need no scaling at all, so `ANEFC2Scales` is per
block. That also conditions the arithmetic rather than merely permitting it —
each entry being the largest power of two holding `bound * scale` in a fixed
window puts every block's typical product in **0.10 to 0.28**, about 1,650x above
the fp16 denormal floor. One global scale would be harsh on the quiet blocks and
push them toward it.

Audited against the GPU's own answer for the same input, every block and every
step, `H3_ANE_FC2_VERIFY=1`:

| extra halvings | cliff margin | worst block | worst rel-RMS | spurious zeros |
|---:|---:|---|---:|---:|
| 0 | 2x | b45 at 1/512 | 1.548e-3 | 8.0e-6 |
| **1** | **4x** | **b45 at 1/1024** | **1.205e-3** | 8.2e-6 |
| 2 | 8x | b45 at 1/2048 | 1.808e-3 | 1.006e-5 |

**No saturation.** Spurious zeros sit near 1e-5 and rise gently as the scale
shrinks, which is fp16 rounding on near-zero outputs; saturation zeros outputs in
bulk, not one in a hundred thousand.

**The margin is non-monotonic in accuracy, and 4x is the optimum.** Both hazards
are live at once: at 2x the partial sums run near the cliff, at 8x the operands
run toward the denormals, and 4x is the only setting that is not paying one of
them. It is strictly better than either neighbour — more accurate than 8x *and*
more accurate than 2x while keeping twice its headroom — so it is the default.
An earlier version of this section recorded 2 halvings as the default on the
reading that extra margin was nearly free in accuracy. It is not: 8x costs 50%
more error than 4x.

What does not move is the floor. `fc2` contracts over `k=14336`, by far the
longest accumulation in the model, and at every margin it remains the least
accurate arithmetic in the route — 1.2e-3 against 7e-5 to 5e-4 for the other
projections. That part is fp16 over a long `k`, not the scale.

**It is worth 1.6%** — 24.48 s a step to 24.10, and 1.176x end to end with all
four projections routed. That is a poor trade against being the least accurate
arithmetic in the route, which is why `H3_ANE_FC2` is a separate opt-in from
`H3_ANE=experimental` rather than riding along with it. The reason it is small
is structural: the engine is slower per FLOP than the GPU, so relocating work
pays only through overlap, and with three projections already routed the dies
are no longer idle enough to absorb a fourth.

### Fusing the modulation buys nothing

`modScaleShift` and `modGate` are four elementwise operations around a gather,
each materialising a 169 MB `[S, hidden]` intermediate, and they are most of the
78 ms a block spends outside its GEMMs and attention. Compiling them is exactly
what took the partial join from 122.4 ms to 108.0, so the same trade was tried
here and it does not repeat:

| | GPU only | split-k |
|---|---:|---:|
| unfused | 1226.6 ms | 1034.2 ms |
| compiled | 1232.2 ms | 1045.4 ms |

Within noise, and slightly the wrong way. The join won because it had `splits`
fp32 intermediates of 306 MB to eliminate; these chains are two or three
operations on bf16 and MLX is evidently already close to the floor on them, or
the gather dominates and will not fuse. The projected 58 ms is not there.

**A caution about how that was nearly mismeasured.** The first comparison put
the compiled version 8% slower — and so was the *control*, measured minutes
after a 45-minute render. It was heat, not code. Re-measuring both arms in the
same thermal state is what turned an 8% regression into a 1% one, and no
performance claim on this page should be read from arms measured an hour apart.


### Sub-process dispatch: the GPU is already saturated by one process

The schedules all failed on the same shape of limit — one MLX thread runs every
upload, gather and collect between engine jobs, and mlx-swift serialises every
`eval` behind one process-wide recursive lock. Both of those are *intra*-process,
so the obvious question is whether separate processes escape them.

They do escape them, and it does not matter. Two processes running the same
8192³ bf16 GEMM, in independent build trees so nothing serialises them:

| | ms | TFLOP/s |
|---|---:|---:|
| one process alone | 59.5 | **18.5** |
| two concurrent, process A | 124.2 | 8.8 |
| two concurrent, process B | 124.4 | 8.8 |

Each runs at half the rate; aggregate is 17.6 TFLOP/s against 18.5 for one.
**Concurrency returns 0.95x, not 2x.** A single MLX process already saturates
this GPU, so there is no idle compute for another process to reclaim — the
contention costs about 5%.

That closes the direction for compute-bound work, which is 92% of a block. The
only thing sub-processes could still reach is the GPU-*idle* window while the
engine runs, which is exactly what query tiling and the CFG pipeline attacked
from inside one process, and the floor there is 906 ms — 1.254x — whoever fills
it.

**Two ways this was nearly measured wrong**, both worth remembering:

- Running `swift test` N times in parallel does not run N processes in parallel.
  SwiftPM locks the build directory, so four "concurrent" runs each reported the
  same 1203 ms per block and looked like 4x throughput. Wall clock gave it away:
  41.1 s for four against 10.5 s for one. They ran one after another. The second
  build tree above is what makes the comparison real, and it needs
  `bootstrap-metal.sh` run against it or MLX dies with a missing metallib.
- Two full renders cannot be used for this at all. The runtime refuses them —
  `H3-4002`, one render at a time, because two 66 GB models cannot be safely
  oversubscribed against a 98.7 GB planned peak. That refusal is correct and it
  is not the thing being measured here.


### What would move it

The contraction split, above. Every schedule closed on this page was closed by
the same number — an engine at 0.41 of the GPU's rate — and that number was an
artifact of contracting 5,376 or 14,336 deep in one call rather than a property
of the hardware. The engine would have to reach ~9.3 TFLOP/s — 18% above
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

#### Splitting the contraction does not settle `fc2` either

Split-k changes what the bound is *about*. A split projection never accumulates
across a piece — each piece is its own evaluation and the partials are summed
afterwards on the GPU in fp32 — so the quantity that must clear the cliff is the
worst single piece. `saturation_bound.py --splits N` computes exactly that, and
its split logic is checked against brute force including the adversarial case
where all the magnitude sits in one piece and splitting correctly buys nothing.

**It could not be run.** The nine oracle captures the bound needs
(`parity/goldens/oracle_prod_matrix/b{00,24,49}_c{000,013,019}`) are not in this
checkout, and there is no in-tree generator — they come from the reference run.
A bound over invented activations is not a bound, so no number was produced.

What the real checkpoint *can* answer is the weight half. `fc2`'s weight L1 is
close to flat across the contraction at block 49 — over all 5,376 output
channels, at eight pieces, the median piece carries 0.1285 of the row's total
against an ideal 0.1250, and the worst channel carries 0.1419:

| pieces | k each | ideal | median | worst channel |
|---:|---:|---:|---:|---:|
| 2 | 7168 | 0.5000 | 0.5023 | 0.5156 |
| 4 | 3584 | 0.2500 | 0.2534 | 0.2655 |
| 8 | 1792 | 0.1250 | 0.1285 | 0.1419 |
| 16 | 896 | 0.0625 | 0.0655 | 0.0777 |

So on the weight side the mechanism works. If the activation magnitudes were
flat across `k` as well, block 49's bound of 978,586 would fall to about
138,861 at eight pieces — 3.8x of headroom where there is now 0.54x, which is
the difference between "exceeds by 1.9x" and "clears".

**That projection is not the bound and must not be used as one.** `fc2`'s input
is the SwiGLU output, which is gated: a large fraction of its channels are near
zero and the surviving magnitude may well concentrate rather than spread. The
weight profile cannot see that. Only the captures can.

And the margin was never the whole objection. The captures cover three of fifty
blocks and 6.6% of sequence positions, the quantity moves 95x across the three
blocks that were measured, and the failure returns zero with nothing downstream
able to see it. Splitting the contraction improves the margin; it does not
improve the sampling. Even at 3.8x, extrapolating across 24 unmeasured blocks of
a quantity that has already been seen to move by 95x is the same bad trade in
better clothes.

**`fc2` remains off the engine, and the prerequisite is unchanged: oracles for
every block.** Still a capture job, not a code job.

### What it costs in memory

Weights are uploaded to the engine as fp16 copies and kept, so they are
duplicated against the bf16 originals MLX already holds — and the split adds a
second cost, because every piece keeps its own `[s, perDie]` partial until the
join sums them. From the shard plan at the shipping share of 0.45:

| | per slot | 2 slots | weights a block | across 50 blocks |
|---|---:|---:|---:|---:|
| `qkv` | 1.39 GB | 2.79 GB | 0.10 GB | 5.0 GB |
| `fc1` | 1.78 GB | 3.56 GB | 0.14 GB | 6.9 GB |
| `attn out` | 0.81 GB | 1.61 GB | 0.03 GB | 1.6 GB |
| | | **8.0 GB** | | **13.8 GB** |

**21.7 GB against 9.2 GB before the split**, on a render that peaked at 87.2 GB
— so roughly 110 GB. Two things grew: the output surfaces, by `splits`, and the
resident weights, because a share of 0.45 hands the engine 45% of every
projection's columns instead of 28.6%.

Measured against the process's resident set rather than derived, for the session
surfaces plus one block of weights — MLX's own `mlxPeakBytes` reports 87.2 GB
whether the engine ran or not, because IOSurfaces are outside its allocator:

| | measured |
|---|---:|
| whole contraction | 3.0 GB |
| split-k | **9.0 GB** |

Comfortable on the 256 GB machine this was measured on, and not comfortable
everywhere. Nothing evicts these while the model is alive, and
`H3_ANE_SPLIT_K=1` returns the whole contraction and the 0.286 share together,
which is the 9.2 GB configuration at 1.091x.

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
