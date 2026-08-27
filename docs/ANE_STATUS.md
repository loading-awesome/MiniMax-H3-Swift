# ANE acceleration: where this stands

Last measured: 2026-08-26 · M3 Ultra, macOS 25F84, ANE architecture `h15g`

Three of a DiT block's four linear projections — `qkv`, `fc1` and `attn out` —
run partly on the Neural Engine behind `H3_ANE=experimental`. The default
production path is unchanged. The bridge validates the private ABI by selector
and refuses unrecognised machines, so an OS update becomes a fallback rather
than a crash mid-sample; `H3_ANE_ALLOW_UNVALIDATED=1` overrides that for
research.

> **Before running anything that touches the engine, read *Machine safety: the
> driver sleep race*.** A one-head, S=512, single-threaded evaluation hard-locked
> this machine on 2026-08-27. The hazard is a five-second driver sleep timer, not
> the size of the work or the number of evaluations in flight, and the bridge
> currently has no defence against it.

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
  submits densely; fatal to the keep-alive idea, which would now be almost pure
  refusal. That idea is dead on its own terms and stays off.
- **A graceful refusal is testable in a way a silent wedge was not.** The bridge
  already falls back to the GPU on a failed submission, so correctness holds and
  the only cost is speed — and the receipts can count it.

### What the upgrade does not change

`fc2`'s bound, the seam economics, the 1.478x hardware ceiling, and the fact
that `ComputeUnitKind` in the new `CoreAI` framework is a single `.neuralEngine`
with no die granularity — see *CoreAI is not a route*. Every performance number
in this document is also from 25F84 and is now stale: the engine share was swept
against a GPU sustaining ~18.6 TFLOP/s, and a new Metal driver moving that moves
the balance point. **Nothing here should be quoted as a 27.0 result until it is
re-measured.**

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

## Machine safety: the driver sleep race, which is not concurrency

**A hard lock is reachable from a single-threaded, one-head, S=512 evaluation.
Size and concurrency are not what makes it dangerous.** On 2026-08-27 a default
`Tools/ANE/attention-spike.mm` run — serial mode, one compiled model, the
configuration its own header calls the safe default — wedged the machine into a
hard shutdown. No panic report was produced, because nothing faulted.

The unified log stops dead. Its final line, and the boot 48 seconds later:

```
11:01:26.262 [ERROR] ANE1: enqueueActionBlock: Skip enqueueActionBlock \
                          fSleepInProgress: 1 fDriverInitiatedSleep: 1
11:01:26.262         ANE1: isANEActive: fIsPowered: 0, fSleepInProgress: 1
11:01:26.262         ANE1: ANE_PowerOn_gated: Powering on ANE. blocking: 1
11:01:26.262         ANE1: ANE_PowerOn_gated: Wait until ANE gets powered up \
                          for client <private> retries=1
11:01:26.262         ANE1: setPowerState:
11:02:14.017 === system boot ===
```

Immediately before those lines the driver is transferring buffer ownership to
`(name:h3-ane-attention pid:15415)` — the spike creating its program on ANE1.

The mechanism the log describes:

1. The program create arrives while that die's driver-initiated sleep is in
   progress.
2. The driver **refuses to enqueue** the action block because sleep is underway
   — the completion that would finish the transition is discarded.
3. The client still needs the die, so the driver calls `ANE_PowerOn_gated` with
   `blocking: 1`.
4. That wait sits inside the IOKit command gate, waiting for a power-up
   notification only the action block from step 2 could deliver.
5. The gate is never released. Every other ANE client — `aned`,
   `localspeechrecog`, `naturallanguaged`, `mediaanalysisd`, `replayd` — is a
   system daemon, and they serialise behind the same gate. Nothing crashes; a
   kernel workloop thread is parked forever holding a lock the system needs.

### The window is five seconds, and other processes keep it open

The driver-initiated sleep timer, measured from the log:

```
11:04:19.724  ANE0/ANE1: ANE_PowerOff_gated: Client requesting power off: mediaanalysisd
11:04:24.725  ANE0/ANE1: DriverInitiatedSleepTimerTimeOut
```

**5.001 seconds of idle.** macOS runs the ANE constantly for its own features,
so both dies cycle awake → 5 s idle → sleep transition all day, driven by
processes that have nothing to do with this repo. Arriving during a transition
is not a coincidence to be avoided by being careful; it is a dice roll that any
process touching the ANE after a pause is taking.

The same `enqueueActionBlock: Skip` error appears in runs that survived — at
10:58:26 during an identical spike run three minutes earlier. It survived
because the dies were hot from a benchmark, so the blocking power-on returned
immediately. Whether the skipped action is fatal depends on whether anything
actually has to wait for it.

The idle **second** die is the one that gets hit. Serial workloads keep ANE0 fed
and give ANE1 nothing — the driver's own statistics after a boot read
`ANE0: WorkSubmitted: 34` against `ANE1: WorkSubmitted: 0` — so ANE1 sleeps on
its timer while the process is still running, and the next two-die program
create lands in its transition. That, not concurrency, is why the earlier
dual-evaluation lock happened: dual mode is simply the first thing to touch a
die that has been idle long enough to sleep. **The header note in
`attention-spike.mm` attributing the lock to concurrency is wrong**, and it sent
the mitigation in the wrong direction.

### What the runtime offers that this bridge never used

System clients bracket their usage explicitly — the log lines are
`Client requesting power on` and `Client requesting power off`. The private
runtime exposes that protocol:

```
_ANEClient:            beginRealTimeTask, endRealTimeTask,
                       loadRealTimeModel:options:qos:error:,
                       unloadRealTimeModel:options:qos:error:,
                       evaluateRealTimeWithModel:options:request:error:
_ANEDaemonConnection:  beginRealTimeTaskWithReply:, endRealTimeTaskWithReply:
```

`H3ANEBridge.m` uses none of it. It drives `_ANEModel` directly —
`compileWithQoS:`, `loadWithQoS:`, `evaluateWithQoS:`, `unloadWithQoS:` — with
no power bracket and no keep-alive anywhere. A model is loaded once and then
evaluated at whatever interval the schedule happens to produce, and the driver
is free to begin sleeping between any two submissions.

`_ANEStrings` lists no real-time-specific entitlement — only
`restrictedAccessEntitlement` (`com.apple.aned.private.allow`) for the private
mach service, `aggressivePowerSavingEntitlement`, the compiler-service and
memory-unwire ones. That the bracket is reachable over the ordinary
`com.apple.appleneuralengine` service is therefore **plausible and unverified**:
`_ANEClient` has `initWithRestrictedAccessAllowed:` and `_ANEDaemonConnection`
has both `daemonConnection` and `daemonConnectionRestricted`, and which one
carries the real-time selectors has not been established. Establishing it means
calling it, which is the operation that costs a reboot when it is wrong.

### The bracket is entitlement-gated, and the keep-alive is what is left

`-[_ANEClient beginRealTimeTask]` is the supported way to say the engine is in
use. Measured 2026-08-27, it is not available to this process:

| connection | `allowRestrictedAccess` | `beginRealTimeTask` |
|---|---|---|
| `sharedConnection` | no | **false**, 0.2 ms, three attempts |
| `sharedPrivateConnection` | yes | **false**, 1.0 ms, three attempts |

Sub-millisecond refusals on a connection that had just been activated, with no
message from `aned` at all, on both the ordinary and the restricted connection.
This is an entitlement check, not a failed operation and not a cold-connection
ordering problem. `_ANEStrings` lists no real-time entitlement, so which one it
wants is unknown; `com.apple.aned.private.allow` is the candidate, and it is an
Apple-private entitlement that an ad-hoc signature cannot carry without
disabling AMFI — a boot-arg change to the machine, which is a worse trade than
the problem.

Requiring the bracket therefore means refusing all ANE work, which was the
bridge's behaviour for about twenty minutes and is not the right answer.

What is left is to keep the timer from ever firing. The transition, not the
sleep, is what is dangerous:

```
11:00:59.796  ANE0: setPowerStateGatedPriv: ANE sleep initiated
11:00:59.811  ANE0: power_off_hardware: Powering off... done      (15 ms)
11:03:08.109  ANE0: setPowerStateGatedPriv: ANE sleep initiated
11:03:08.117  ANE0: power_off_hardware: Powering off... done      ( 8 ms)
```

**8 to 15 ms.** A fully-asleep die powers on correctly — that happens at every
boot — so the exposure per create is that window, not the whole idle period.

`h3_ane_keepalive_is_running` submits a 64x64x64 pair to both dies every two
seconds, starting on the first `h3_ane_program_create` and running for the
lifetime of the process. Both dies, because an idle die sleeps on its own timer
while the other works, and the idle second die is the one that got hit. The
keep-alive's own create goes first, so the real creates that follow — one per
new shape, dozens across a benchmark — all happen with the timer held off.

This is a reduction in exposure, not a proof of safety. What remains is the
keep-alive's own first create, plus any real create that races it from another
thread before the timer is running. Without the entitlement there is no way to
close that, because there is no way to ask the driver to wait.

Measured over 31 s of holding with no other work in the process
(`Tools/ANE/power-bracket-check.m`, one create then idle):

| signature | occurrences |
|---|---|
| `DriverInitiatedSleepTimerTimeOut` | **0** |
| `setPowerStateGatedPriv: ANE sleep initiated` | **0** |
| `enqueueActionBlock: Skip` | **0** |
| driver events, ANE0 / ANE1 | 177 / 177 |

All three signatures that preceded the hard lock are absent, and the two dies
are equally active — the idle second die, which is the one that got hit, is
being fed.

What this does **not** do is pin the engine powered. The log shows it still
power-cycling about once a tick, but through `ANE_PowerOff_gated: Client
requesting power off` rather than `fDriverInitiatedSleep: 1`. Power-on from
fully-off is the normal path and is what happens at every boot; the wedge needed
a driver-initiated sleep in progress. Suppressing that timer is what the
keep-alive does, and it is all it does. Pinning the engine powered is what the
bracket would do, and that needs the entitlement.

### Three watchdog resets, and what the hang actually is

By the end of 2026-08-27 this machine had hard reset three times: 08:02, 11:02,
11:49. It had never done so before that day. All three fell during ANE testing.

The reset is **not** a power fault, and the owner's account settles it: hard
shutdown followed by **self-recovery**, with nobody touching the power button.

| evidence | reading |
|---|---|
| self-recovery, no human | the reset was automatic |
| no panic report, ever | nothing faulted — a stuck thread does not panic |
| `panicmedic-telemetry` in NVRAM | the OS was unresponsive and was reset for it |
| 35 s, 48 s, 61 s from last log line to boot | not a power drop, which reboots in seconds; this is lost log tail plus watchdog timeout plus early boot |
| `wdog` in all three `ResetCounter` records | consistent, though identical strings across three same-class events prove little on their own |

The mechanism is a kernel workloop thread parked forever holding a gate, which
stops the kernel making forward progress, which the watchdog answers by
resetting the machine. For the 11:02 reset it is caught in the act — the log
stops mid-sequence inside `ANE_PowerOn_gated: Wait until ANE gets powered up`,
with the skipped enqueue two lines above. The other two have no ANE line at the
end, which is weaker evidence than it looks: the unified log writes
asynchronously and a wedged kernel never flushes its tail.

The 11:49 reset came **2.5 minutes after** the benchmark process exited, with
the last ANE event being a clean, normal sleep. That is the important detail for
what to do about it. Once a process using the engine exits, the dies idle, five
seconds later a transition opens — and macOS touches the ANE constantly for its
own features. The access that lands in the window does not have to be ours, and
in that case there is nothing of ours in the log at all.

### Holding the engine for a session, and what that does not buy

`Tools/ANE/ane-hold.m` runs for a whole work session and suppresses the idle
timer for its duration. It creates one 64x64x64 program through the shipping
bridge — which starts the bridge's own keep-alive — so what it holds is exactly
what a render holds rather than a second mechanism that could drift.

The scope is the point. A per-process keep-alive dies with its process, which is
precisely when the window opens. One holder across a session collapses many
transitions into one.

**It does not pin the power plane, and nothing available to this process can.**
`-[_ANEClient beginRealTimeTask]` is entitlement-gated and refuses on both
connections. The engine still power-cycles under client-requested power-off —
about eleven times in a 31 s measurement — and whether *that* transition path
can race has not been established. This narrows the window; it does not close
it.

### The wired memory was not a leak

`swiftpm-testing-helper` was observed holding 702 MB wired with 4 programs open,
which invites a leak diagnosis. It is not one. After the process exited the
driver reports `Programs Open:0 Wired-Memory:0` for it, with
`ANE_UserClientClose_gated_block_invoke` on both dies. The island holds
persistent compiled programs and weight surfaces by design; four open programs
with weights wired is the intended steady state, and it is released on exit.

### It is a DART fault, and the keep-alive is a suspect (2026-08-27)

The fourth failure of the day produced a panic log, and it says something none
of the previous three could:

```
panic(cpu 2 caller ...): sptm_t8110dart_clear_err:
  dart 0xfffffdc018445bc8 (dart-ane0:46): DART instance 1:
  Unrecoverable secondary error 0x80080008
Kernel Extensions in backtrace:
  com.apple.sptm
  com.apple.driver.AppleT8110DART  (dependency: IODARTFamily)
```

**DART is the IOMMU, and `dart-ane0` is ANE die 0's.** This is a device address
translation fault, caught by the Secure Page Table Monitor and declared
unrecoverable. It is not the software gate wedge this section was written
around. "Secondary error" means a second fault arrived while the first was being
handled.

What was running: **only `Tools/ANE/ane-hold.m`**. No benchmark, no render, no
test. It started at 12:07:33 and the machine panicked at 12:08:45 — **72
seconds** — with nothing on the machine touching the engine except a 64x64x64
pair submitted every two seconds.

That inverts the story this document told an hour earlier. The keep-alive was
built to prevent a hang and is now the best-evidenced trigger of a panic. The
plausible mechanism is the one thing it uniquely does: because it cannot pin the
power plane, the engine still power-cycles under client-requested power-off
between ticks — about once every three seconds, measured — so a keep-alive at
2 s manufactures DART teardown and restore at a rate Apple's own stack never
produces. Rapid power-cycling with DMA mappings live is a plausible way to reach
a secondary translation fault, and it is not a pattern any normal client makes.

It is therefore **off unless `H3_ANE_KEEPALIVE=1`**, in the bridge and in
`Tools/ANE/differential.m`. Do not set that variable to make `ane-hold` work.

This also revises the earlier reading of the three watchdog resets. A DART fault
that wedges the fabric rather than reaching the panic path would hang the kernel
exactly as observed — no panic, no log tail, watchdog reset. The 11:02 wedge
inside `ANE_PowerOn_gated` may be a *symptom* of a device that stopped answering
because its DART had faulted, rather than a pure software deadlock. The sleep
race remains a real and documented hazard; it may not be the whole story, and it
is no longer clear it is the main one.

**Nothing about this is understood well enough to keep running the engine.**
Four failures in one day on a machine with no prior history: three watchdog
resets and one DART panic. Two mitigations have now been tried and one of them
made things worse.

### One surface object, two dies, two DARTs — the leading hypothesis

`h3_ane_job_submit_pair` dispatches its two evaluations **concurrently**, with
instance hints 1 and 2 — two dies, each behind its own DART. Every caller passes
the *same input tensor* to both: the island (`handles.xs[split]` to gate0 and
gate1), the shard path, and the keep-alive. Inside `h3_ane_run` each thread then
built its `_ANERequest` from `x->object` — one cached `_ANEIOSurfaceObject`
shared between two in-flight requests against two different IOMMUs.

If that private class keeps per-evaluation mapping state — which IOVA the
surface is currently mapped to for the requesting device — two concurrent uses
race on it, and one request is programmed with an address not valid in its own
DART. That is the fault the machine panicked with.

The class is undocumented, so whether sharing is legal cannot be settled by
reading. Wrapping the same IOSurface once per evaluation is cheap and removes
the question, and `h3_ane_run` now does that. **The fix is untested against the
hardware**, because the rule below says nothing runs.

Two things make this the leading candidate rather than one of many:

- It is the only structure found that can hand the device an address belonging
  to a different mapping. Allocation size (`rows x align64(width x 2)`, floored
  at 16 KB), `write_prefix` bounds, and padded-sequence agreement between the
  compiled program and the allocated surface were all audited and are
  conservative.
- It scales with exactly what changed. The sharing is longstanding, but today
  was the first day of sustained high-rate pair submission — dual-die spikes all
  morning, the island routing for the first time, then a keep-alive doing it
  every two seconds indefinitely.

**And the IOMMU is not the safety net it first appears.** DART maps at 16 KB
page granularity, so a device access that overruns a surface but stays inside
the same page is never faulted — it silently reads or writes whatever else is
there, and within one DART the neighbouring mappings belong to other ANE
clients, which are Apple's. The panic is the page-crossing tail of that
distribution. Silent corruption of another client's compute is a plausible
outcome of the same defect, and is what the machine's owner suspected before
this was found.

### Two hypotheses tested and eliminated, and where that leaves it

`Tools/ANE/pair-stress.m` submits `h3_ane_run_pair` in a loop with the same
input and weight surfaces bound to both dies — the worst case, and what
`ane-hold` was doing when the machine panicked 72 s in. It takes a gap in
milliseconds, so the submission rate is the variable.

| configuration | pairs | outcome |
|---|---|---|
| shared surface object, no gap | 671,438 in 180 s | clean |
| per-request surface object, no gap | 636,432 in 180 s | clean |
| shared, 200 ms gap | 879 in 180 s | clean |
| shared, 2000 ms gap | 120 in 240 s | clean |
| `ane-hold`, ~2000 ms gap | ~36 in 72 s | **DART panic** |

**The shared `_ANEIOSurfaceObject` is not the fault.** Head to head at 3,700
pairs a second it is indistinguishable from wrapping per evaluation, so the
default stays shared and the alternative is available under
`H3_ANE_PER_REQUEST_SURFACE_OBJECT=1`.

**Submission volume is not the fault either.** 1.3 million pair submissions
across six minutes did nothing. Thirty-six submissions across seventy-two
seconds killed the machine. Whatever this is, hammering the engine is safer than
poking it.

**And a 2 s gap alone does not reproduce it.** Four minutes at exactly the rate
that panicked the machine, clean.

So there is **no reliable reproduction**, which is the most important fact for
anything that comes next. Four failures in roughly five hours of intermittent
heavy use is a base rate against which a four-minute clean run is worth almost
nothing — no mitigation can be validated by absence of failure at this
timescale, including the two already committed.

What survives, unproven: the failure needs something we are not controlling.
The machine's owner framed it as a lock CoreML takes that we cannot — and the
shape of the evidence fits. We register as a client (the driver tracks our
`Programs Open` and `Wired-Memory`) but we cannot register *intent*:
`beginRealTimeTask` is the "I am holding this die" declaration and it is
entitlement-gated. Without it the driver is free to act on the die — power it
down, tear down and restore DART mappings — on its own schedule, while our
mappings are live, and with no way to tell us. Continuous submission does not
take a lock; it denies the driver the opportunity. That would explain why
sustained work is safe, intermittent work is dangerous, and neither is
deterministic: it needs the driver, or another client, to act in the window.

Testing that needs a soak with per-client driver logging and a failure to
examine, not another four-minute run.

### The panic reproduces, with the fault deferred ~8-10 minutes (2026-08-27)

Three DART panics, all byte-identical:

```
sptm_t8110dart_clear_err: dart (dart-ane0:46): DART instance 1:
  Unrecoverable secondary error 0x80080008
backtrace: com.apple.sptm, com.apple.driver.AppleT8110DART
```

| # | boot | panic | preceding ANE work | delay |
|---|---|---|---|---|
| 1 | 11:48:53 | 12:08:45 | `ane-hold` started 12:07:33 | +72 s |
| 2 | 12:09:10 | 12:48:37 | `pair-stress` ended 12:40:13 | **+504 s** |
| 3 | 12:48:59 | 13:08:25 | `pair-stress` ended 12:58:42 | **+583 s** |

**Panics 2 and 3 are the same experiment, run twice.** The second was a
deliberate replication of the first: `pair-stress 240 2000`, then the `probe`,
then idle. It reproduced. That makes this a deterministic mechanism with a
variable delay, not the random hardware event it looked like.

Identical error codes are the other half of that argument. Random corruption —
a stray write, an electrical fault — gives varied failure addresses and varied
codes. The same DART, the same instance, and `0x80080008` three times means the
hardware takes the same fault path every time.

**The fault is deferred and fires while the machine is idle.** At the moment of
panics 2 and 3 there had been no ANE driver activity for minutes; the last
flushed log lines are ordinary system chatter. Whatever the work leaves behind
surfaces eight to ten minutes later, with nothing running.

Two methodological lessons, both learned the expensive way:

- **A clean run proves nothing.** Every clean result recorded above —
  1.3 million pairs, the 200 ms sweep, the 2 s sweep — was measuring a window
  that closes before the fault arrives. They are not safety evidence.
- **The replication was nearly called a negative.** The watch was stopped at
  +553 s because the first panic came at +504 s. The machine panicked 30 s
  later. Anything watching for this needs to run well past 600 s before
  concluding anything, and the variance is not yet bounded.

The fused attention graph is **not** implicated: it never executed in any of the
three, exiting 127 on a missing binary both times it was invoked. Its apparent
correlation with panics 2 and 3 was coincidence — the deferred clock was already
running from the preceding stress run.

### The rule this leaves

- **Nothing runs on the engine.** As of the DART panic there is no configuration
  of this bridge known to be safe on this machine, and the two mitigations tried
  so far are a partial fix and a suspect. `ane-hold` refuses to run.
- The keep-alive, in the bridge and in the spikes, is off unless
  `H3_ANE_KEEPALIVE=1`.
- **No ANE work of any size runs outside the keep-alive.** The toy shapes are
  not the safe subset; there is no safe subset. `h3_ane_program_create` starts
  the keep-alive before it creates anything, and returns NULL if it cannot.
- It covers **both** dies, at two seconds against the driver's five.
- Anything that talks to the private runtime without going through
  `H3ANEBridge` needs the same guard. `Tools/ANE/differential.m` has it, and the
  five spikes that include it inherit it; the two island spikes reach the engine
  through the bridge and inherit it there.
- The residual first-create exposure is real and unclosable without the
  entitlement. It is one window of 8-15 ms per process, against roughly one
  create — not one per shape, which is what it was when this machine went down.

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
