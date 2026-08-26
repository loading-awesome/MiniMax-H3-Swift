# ANE acceleration plan for the H3 DiT

- Status: research plan; no production ANE backend exists yet
- Written: 2026-08-26
- Target machine: M3 Ultra, macOS 15+
- Initial scope: the four large linear projections in each DiT block
- Explicit non-goal: moving dense attention to ANE in the first implementation

Reverse engineering of the installed compiler and runtime is a prerequisite,
not an implementation detail. See `docs/ANE_REVERSE_ENGINEERING.md` for the
machine fingerprint, recovered ABI surface, differential-test protocol, and
open questions that must be resolved before the bridge is designed.

## Decision in one paragraph

Build an opt-in, direct ANE execution path that splits each large DiT linear by
output channels between Metal and both physical ANE instances. Keep attention
and all existing model orchestration on MLX/Metal. Compile a small set of
fixed-shape MIL programs once, supply one layer's weights dynamically through
IOSurface, and join the independently produced output columns without sending
either result through the CPU. The work proceeds through isolated probes with
hard correctness, concurrency, copy-cost, and end-to-end gates. If dynamic
weights require a per-dispatch staging copy, GPU and ANE do not overlap, or the
projected full-forward gain falls below 1.20x, stop before production
integration.

## Why this is worth testing

The current production forward is limited by GPU execution, not by the small
elementwise passes around it. Existing measurements in this tree put one full
forward at approximately 961 TFLOP and 60 seconds:

| Work | TFLOP | Current owner |
|---|---:|---|
| Dense attention | 355 | GPU |
| Four block linears | 606 | GPU |
| Total | 961 | GPU |

`H3_BIG=1 swift test --filter aneCeiling` measured the production linear shapes
on one M3 Ultra ANE placement:

| Projection | Shape `[K,N]` | ANE time at `S=2048` | Achieved |
|---|---:|---:|---:|
| QKV | `[5376,21504]` | 87.3 ms | 5.42 TFLOP/s |
| Attention output | `[7168,5376]` | 48.7 ms | 3.24 TFLOP/s |
| MLP fc1 | `[5376,28672]` | 115.2 ms | 5.48 TFLOP/s |
| MLP fc2 | `[14336,5376]` | 174.4 ms | 1.81 TFLOP/s |

Weighted across the real projection mix, that is about 3.7 TFLOP/s on one
physical ANE instance and an ideal 7.4 TFLOP/s on both. If the GPU sustains its
measured 16 TFLOP/s and all three engines overlap, the balanced split sends
about half of linear FLOPs to ANE:

```text
ANE time  = f * 606 / 7.4
GPU time  = (1 - f) * 606 / 16        <- linears only
forward   = 355 / 16  +  606 / (16 + 7.4)

balanced f ~= 0.32
forward time ~= 48 seconds
ceiling ~= 1.25x
```

> **Corrected 2026-08-26.** This block previously solved `f * 606 / 7.4`
> against `(355 + (1 - f) * 606) / 16`, giving `f ~= 0.50`, 41 seconds and
> 1.46x. That form lets attention overlap the ANE's linear work, and the
> dependency graph forbids it: a block is a chain — qkv, attention, attention
> output, fc1, fc2 — and block `i+1` consumes block `i`, so attention is a
> serial GPU-only phase with both dies idle. `docs/ANE_OVERLAP_RESULTS.md`
> had the correct form all along and the two documents disagreed.
>
> Two consequences. The balanced ANE share is 32%, not 50%, which needs about
> 6.1 GB of INT8 linears resident rather than 9.7 GB — easier, not harder.
> And **attention alone caps this whole exercise at 2.7x**: 355 TFLOP at
> 16 TFLOP/s is 22.2 seconds no matter how fast the engine is.

This is a ceiling, not a forecast. It excludes integration barriers, layout
conversion, driver scheduling, output joining, and contention. The purpose of
the plan is to price those terms before changing the production model.

## Constraints established by research

### The approximately 4 GiB limit is an address window

OMLX measured an approximately 4 GiB device-address window for resident program
weights on each physical ANE instance. Loading a bank maps its complete static
weight blob into that instance's window; loads beyond the window fail with
`0x20004`. The two M3 Ultra instances provide two independent windows, not one
shared 8 GiB allocation.

This is not the machine's unified-memory capacity and it is not the ANE's small
on-chip working memory. Apple has not published the implementation cause. A
32-bit address or byte-offset field in the compiler, program format, firmware,
DMA commands, or per-instance IOMMU aperture is the leading explanation, but
must remain labelled as inference.

H3's large linear weights are approximately 38.5 GB at 16 bits. A balanced
static split would therefore require about 19.3 GB per ANE instance and cannot
fit. Smaller resident banks could be unloaded and reloaded, but that introduces
program-create latency and turns the address problem into a scheduling problem.

### Dynamic weights bypass static residency, not DRAM

The maderix dynamic pipeline packs activations and weights into an IOSurface
input, slices them inside a fixed MIL program, and changes weights without
recompilation. A local proof using that path successfully executed a
`768 x 768 x 256` dynamic matmul. It measured approximately 0.88 TFLOP/s when
weight rewriting was excluded and 0.52 TFLOP/s when CPU weight I/O was included.
That establishes feasibility and identifies staging as a first-class cost; it
does not establish production-shape performance.

Static and dynamic paths both fetch weights from unified DRAM. The bad dynamic
implementation adds a source read and IOSurface write before the ANE read:

```text
static or zero-copy dynamic:  ANE reads W
staged dynamic:               read W + write surface + ANE reads W
```

The address-window workaround is therefore compatible with spare memory
bandwidth only if the storage layout avoids per-dispatch repacking.

### Long-sequence H3 linears should have high arithmetic intensity

For `X[S,K] * W[K,N]`, FP16/BF16 work is `2*S*K*N` FLOPs and one weight pass is
`2*K*N` bytes, giving approximately `S` FLOPs per weight byte. At an ANE tile
of `S=2048`, 4 TFLOP/s corresponds to only about 2 GB/s of ideal weight traffic.
Hardware tiling and cache misses increase this, but the starting intensity is
high. Apple independently reports that transformer layers tend to become
bandwidth-bound at short sequence lengths because weights are reused over too
few inputs; H3 is deliberately testing the opposite regime.

The M3 Ultra advertises more than 800 GB/s of aggregate unified-memory
bandwidth. That number is not a per-engine guarantee, and GPU and ANE share the
memory controllers and system caches. "GPU-bound" must therefore be verified
with counters and concurrent timing; high GPU utilization alone does not prove
compute-bound execution.

### Precision is a release constraint

The reference DiT computes in BF16. The explored ANE paths use FP16 I/O and may
accumulate differently. BF16 checkpoint values within FP16's exponent range can
be represented exactly in FP16 — measurement has since shown representation is
**not** where the loss comes from. Two mechanisms are, in this order:

1. **Denormal flush.** Products below fp16's smallest normal, 6.10e-5, are
   dropped inside the multiply-accumulate. This dominated the QKV spike by
   152x. **Measured against real captured activations it costs this model
   nothing**: underflow runs 0.02%-6.7% and relative error stays at 1e-4.
2. **Per-product fp16 rounding**, the floor, at 7e-5 to 5e-4 on real tensors.

> **Measured 2026-08-26 — see `docs/ANE_PRECISION_RESULTS.md`.** Against real
> block taps and real checkpoint weights, the FP16 ANE path scores 7e-5 to
> 5e-4 relative RMS per projection while the **bf16 GPU path it would replace
> scores 1.66e-3**. Per projection the engine is 3 to 20 times *more* accurate
> than contract 8's ceiling, so precision is no longer the objection to this
> work.
>
> The objection that replaces it is **saturation**. At block 49, fc2 drives
> interior partials to 34,649 against the 2^15 threshold at which a dot product
> returns zero, destroying 0.02% of its outputs and taking that projection's
> error from 1.0e-4 to 0.101 — silently. A power-of-two operand scale of 1/16
> restores 15x headroom at no precision cost, exactly, because a linear is
> homogeneous and the scale is exact in fp16.

The accumulator itself is wide and exact, so long contractions are not a
hazard and fc2's K=14,336 is the least sensitive projection, not the most.
A running partial that reaches 2^15 returns **zero** rather than inf or NaN, so
`max|interior partial|` must be bounded per projection *and per block* before a
shape is enabled — this is now a measured failure, not a hypothetical, and the
bound must carry margin because the published figures are maxima over samples
rather than over full tensors. No speed result is
eligible for integration until it passes production-shape operator comparisons,
block-boundary taps, trajectory parity, and the existing perceptual release
checks.

### The interface is private

The direct route uses private AppleNeuralEngine classes such as in-memory model
descriptors, requests, and IOSurface objects. It can break on an OS update and
is unsuitable for an App Store distribution promise. It must be runtime-loaded,
hardware- and OS-gated, disabled by default, and fall back to the existing MLX
path without changing its results.

## Proposed architecture

```text
                         existing MLX graph
                                |
                   split eligible linear by N
                    /                       \
         Metal: X * W_gpu^T          ANE dispatcher
                    |                 /            \
                 Y_gpu          ANE die 0      ANE die 1
                                  W_ane0          W_ane1
                                     \            /
                                      Y_ane slices
                    \                       /
                     zero-copy column join
                                |
                     existing MLX graph
```

The differential pass in `docs/ANE_DIFFERENTIAL_RESULTS.md` resolved the first
weight-binding decision: use two ordinary tensor inputs, one activation and one
weight IOSurface. Runtime weight mutation works without recompilation and
matches the CPU oracle. Packed activation-plus-weight input is now a fallback,
not the baseline, and `_ANERequest.weightsBuffer` is rejected because it was
accepted but had no numerical effect.

Per-die energy telemetry (`Tools/ANE/counters.h`) later showed that
`kANEFAneInstanceHint` does **not** pin a program to a physical die: a job
submitted alone with hint 2 executes on die 0 while die 1 stays powered down.
Two dies are engaged by submitting two jobs concurrently and letting the kernel
load-balancer place the second. Any design below that assumes a program runs
where it was asked to must verify placement rather than request it — this
matters most for per-die resident weight banks, where a misplaced job is a
correctness failure and not a scheduling one.

The coexecution pass in `docs/ANE_OVERLAP_RESULTS.md` also resolved the hardware
side-channel question on h15g/25F84. Metal produced an activation directly into
the ANE IOSurface with zero output error. A compute-bound 4096-cubed GPU GEMM,
ANE instance 1, and ANE instance 2 completed together in 23.8–24.0 ms versus
about 57.1 ms serialized. At the tested H3-shaped dynamic matmul, each ANE
contributed about 3.7 TFLOP/s without slowing the GPU. The implementation target
is therefore an evidence-based roughly 1.25x full-forward ceiling before
handoff and integration costs.

The first production-shape projection spike is recorded in
`docs/ANE_PROJECTION_SPIKE.md`. QKV `[2048,5376] x [5376,21504]` split into
15,360 GPU channels and 3,072 channels on each ANE ran 1.27–1.32x faster than
the full GPU projection after conversion and join costs. Its synthetic
independent-distribution precision result did not clear production: each FP16
ANE shard differed from the bf16 GPU oracle by about 3.17% relative RMS. The
next implementation is therefore one disabled QKV custom primitive plus real
checkpoint/captured-activation conformance, not broad routing.

### Repository boundary

Add a narrowly scoped C/Objective-C bridge target rather than exposing private
Objective-C types throughout Swift:

```text
Sources/H3ANEBridge/
  include/H3ANEBridge.h       stable C ABI used by Swift
  H3ANEBridge.m               runtime lookup, compile, request, IOSurface

Sources/H3Modules/
  ANE/ANEAvailability.swift   capability and OS checks
  ANE/ANEProgramCache.swift   fixed-shape program ownership
  ANE/ANESurfacePool.swift    reusable input/output surfaces
  ANE/ANELinearBackend.swift  split policy and async dispatch
```

Names are proposed, not committed API. The bridge must resolve private symbols
at runtime so an unavailable framework or changed selector becomes a clean
`unavailable` result rather than a load-time crash.

### Program set

Compile programs once per sequence tile and projection geometry, not per block
or denoising step. The initial program set covers the four production shapes:

```text
[S,  5376] x [N,  5376]  QKV slices,       total N=21504
[S,  7168] x [N,  7168]  attention output, total N=5376
[S,  5376] x [N,  5376]  MLP fc1 slices,   total N=28672
[S, 14336] x [N, 14336]  MLP fc2 slices,   total N=5376
```

Start with fixed `S=2048` tiles because row tiling is exact for a linear and the
existing ceiling measurement used that shape. Tail rows are zero-padded and
sliced after execution. Sweep `S=512,1024,2048,4096` only after the direct path
is correct; choose by end-to-end time, including packing and the tail.

### Output-channel split

Split the output dimension `N`, never `K`. Each engine receives the complete
activation tile and a disjoint set of weight rows, so no partial sums or
cross-engine reduction are required. The result is an exact concatenation in
the model's canonical output order. Splitting `K` would require an additional
sum with changed accumulation order and more synchronization.

The split fraction is selected per projection from measured rates, not fixed at
50 percent. Respect head and SwiGLU boundaries:

- QKV slices remain separately ordered as Q, K, and V and align to complete
  attention heads.
- MLP fc1 keeps corresponding gate/up channel partitions paired.
- Output widths obey the ANE compiler's channel alignment.
- GPU, die 0, and die 1 receive contiguous weight regions whenever possible.

### Memory ownership

Use IOSurface-backed allocations that can also be represented by Metal resources.
The intended steady state is:

1. Checkpoint weights are converted or loaded once into their final partitioned,
   ANE-consumable layout.
2. Metal retains only its output-channel partitions and ANE surfaces retain only
   theirs; the complete 38.5 GB linear set is not duplicated in both forms.
3. Weight surfaces remain host-resident but are only mapped into an ANE request
   for the active layer/tile.
4. MLX/Metal writes activation tiles into shared surfaces without a CPU readback.
5. ANE writes its output slices into shared surfaces.
6. Metal consumes or joins those slices without a CPU copy.

Capability discovery and program compilation happen before partitioned model
loading. If they fail, load the ordinary complete MLX weights. Once a render has
started, retain the selected partition ownership for its lifetime rather than
keeping a second complete copy solely for fallback.

If the private API requires activations and weights in one IOSurface, use a
reusable surface layout with fixed offsets. First test whether weights can stay
in prepacked per-layer surfaces while only the activation region changes. If
not, measure the compulsory copy and apply the stop gate rather than hiding it
inside dispatch time.

Memory lifetime must be explicit. The program cache, IOSurface pool, Metal
views, and in-flight requests retain their storage until both GPU and ANE command
completion. No pointer obtained from an MLX array is assumed stable across lazy
evaluation.

### Scheduling

For each eligible projection:

1. Force only the dependencies needed to make `X` available.
2. Submit the GPU output-channel slice.
3. Submit both ANE requests without waiting for the GPU.
4. Synchronize once at the first consumer of the complete output.
5. Expose the joined result back to MLX/Metal.

Do not place a CPU wait between the two submissions. Use one serial ownership
queue per ANE instance and enough surface slots for at least two in-flight
tiles. Record GPU time, each ANE time, overlap, staging, join, and total time
separately.

### Feature policy

The backend starts as an explicit diagnostic opt-in, for example
`H3_ANE=experimental`. It activates only when all of these are true:

- supported Apple Silicon and macOS build;
- both physical ANE instances can be acquired when dual mode is requested;
- every required program compiles and passes a warm-up dispatch;
- memory allocation and Metal interop succeed;
- the model uses a validated geometry and dtype;
- every enabled projection has a measured `max|interior partial|` bound with at
  least 4x margin below 2^15, at every block it is enabled for, with the
  per-block operand scale applied;
- no unsupported packaging or hardened-runtime constraint is present.

Any initialization failure falls back before sampling begins. A request failure
after sampling begins is fatal for that render; silently changing arithmetic
halfway through a diffusion trajectory would invalidate reproducibility.

## Work plan and gates

### Phase R — Reverse-engineer the installed stack

Complete RE-1 through RE-6 in `docs/ANE_REVERSE_ENGINEERING.md` before creating
the production bridge. Use public Core ML as the semantic and placement oracle,
then compare equivalent direct in-memory MIL programs one variable at a time.
Resolve the separate request `weightsBuffer`, multiple-input behavior, explicit
IOSurface mapping, private shared events, performance statistics, physical
instance hints, and compiler boundaries on this M3 Ultra and OS build.

Gate: produce a versioned machine/ABI fingerprint and a production-shape table
covering static and dynamic weights, copy cost, synchronization, dual-instance
overlap, numerical error, and failure boundaries. No private selector or option
key enters production code solely because OMLX or another machine accepted it.

### Phase 0 — Preserve and formalize the evidence

Deliverables:

- Move the current ANE ceiling probe into an intentional, documented diagnostic
  once its uncommitted ownership is resolved.
- Record machine identifier, OS build, model geometry, compute-unit placement,
  program compile time, median dispatch time, and spread.
- Add the four ANE projection results to the performance evidence format rather
  than relying on terminal output.
- Add a bandwidth baseline for each projection and a concurrent GPU memory-load
  control.

Gate: repeated measurements must have less than 5 percent spread and reproduce
ANE placement for all four production shapes. Failure means the public Core ML
measurement is too unstable to size the direct implementation.

### Phase 1 — Minimal direct-runtime bridge

Deliverables:

- Runtime-load the private framework and selectors.
- Compile and execute one constant-weight MIL matmul through the in-memory API.
- Add structured errors for missing framework, selector drift, compiler failure,
  program load failure, request failure, and timeout.
- Prove resource cleanup under repeated compile/load/unload cycles.
- Capture the framework and selector fingerprint with every benchmark.

Tests progress through `64`, `768`, and one H3 production geometry at
`S=2048`. Compare against MLX using FP32 relative-RMS, maximum absolute error,
cosine similarity, and non-finite counts.

Gate: the direct constant-weight path must agree with the Core ML ceiling within
15 percent at a production shape and produce a stable numerical error class.
Otherwise the bridge is driving a different compiler or execution path and the
existing projection cannot be used.

### Phase 2 — Dynamic weights at production shapes

Deliverables:

- Generate fixed MIL that slices activation and weight regions from IOSurface.
- Exercise all four production shapes with real checkpoint distributions.
- Measure separately: activation write, weight staging, request submission,
  ANE execution, output availability, and surface reuse.
- Verify weights can change between consecutive requests without recompile or
  stale-cache behavior.
- Sweep sequence tile size and output-channel alignment.

Gate: with weights already in their final surface, dynamic execution must retain
at least 80 percent of constant-weight ANE throughput, or its measured
full-forward projection must still exceed 1.20x. Per-dispatch CPU weight packing
is not accepted as the final design.

### Phase 3 — Prove zero-copy Metal/ANE interop

Deliverables:

- Create Metal views of ANE input and output IOSurfaces.
- Write the activation from a Metal kernel and consume ANE output from Metal.
- Prove visibility and ordering with explicit completion primitives.
- Compare shared-surface and copied paths using identical data and ABBA timing.
- Measure page faults and first-use effects separately from steady state.

Gate: steady-state CPU-transferred bytes for weights and activations are zero,
and boundary overhead is below 5 percent of the ANE slice time. If zero-copy is
impossible, recompute the complete speed projection with measured copies and
require at least 1.20x.

### Phase 4 — Prove actual GPU plus dual-ANE overlap

Deliverables:

- Run the real GPU slice alone, each ANE slice alone, both ANEs together, then
  all three engines concurrently.
- Sweep split fractions per projection and solve for the minimum completion time.
- Record aggregate DRAM traffic and GPU counters when available.
- Run a synthetic bandwidth saturator beside the experiment to identify where
  contention begins.
- Repeat warm and cold, with thermal and power state recorded.

Use these quantities:

```text
serial_cost       = T_gpu + max(T_ane0, T_ane1)
ideal_concurrent  = max(T_gpu, T_ane0, T_ane1)
observed          = wall time from first submit to all complete
overlap_efficiency = ideal_concurrent / observed
```

Gate: all engines must overlap, overlap efficiency must be at least 0.85, and
the measured projection across the real four-layer mix must exceed **1.15x**.

> **Corrected 2026-08-26.** This gate read 1.25x, which was chosen when the
> ceiling was believed to be 1.46x. The corrected ceiling *is* 1.25x, so the
> gate demanded a perfect implementation with zero handoff cost. 1.15x keeps a
> real margin below the ceiling while still being worth the private-API
> exposure; anything at or below 1.10x is not.

The DRAM-traffic half of this phase is answered, though not in the form the
phase asked for. Engine-attributed byte counters do not exist on M3 Ultra, but
`PMP*|DCS BW` gives a 32-bin total-bandwidth histogram per memory controller,
and across every measured arm **peak DRAM read never exceeds 192 GB/s against
roughly 800 GB/s available**, with the controllers in their lowest bin for at
least 85% of each arm. There is no bandwidth contention to find at these
shapes. See the telemetry section of `docs/ANE_OVERLAP_RESULTS.md`.

### Phase 5 — One complete DiT block behind a seam

Deliverables:

- Introduce a linear-backend seam at QKV, attention output, fc1, and fc2.
- Keep the existing MLX implementation as the unconditional fallback.
- Run one real block with captured production activations.
- Preserve QKV ordering, head grouping, SwiGLU gate/up ordering, residual timing,
  and lazy-evaluation boundaries.
- Add per-site timing and numerical taps.

Gate: all four sites must stay within the numerical class established in Phase
1, no error may amplify unexpectedly across the block, and the block must be at
least 1.20x faster. A fast isolated GEMM with a slow block is a failed result.

### Phase 6 — Full 50-block forward

Deliverables:

- Execute an uncached conditional and unconditional forward with the ANE path.
- Record time per projection and block, fallback count, surface high-water mark,
  memory peak, and first divergence from the MLX reference.
- Compare dense control and experimental arms under the benchmark contract in
  `docs/PERF_ROADMAP.md`.
- Test repeated forwards to expose leaks, compiler limits, stale mappings, and
  thermal behavior.

Gate: median full-forward speedup must be at least 1.20x with no fallbacks and
no memory growth over repeated forwards. The result must clear the existing
run-to-run timing noise floor by a wide margin.

### Phase 7 — Release parity and operational hardening

Deliverables:

- Run Tier 2 numerical fixtures and Tier 3 production trajectory parity.
- Run the existing visual, temporal, audio, transcript, and lip-sync oracles.
- Repeat across supported macOS point releases and at least one non-Ultra Apple
  Silicon machine to confirm clean fallback.
- Add crash-safe teardown, timeout handling, diagnostics, and a single-command
  capability report.
- Document private-API, code-signing, distribution, and license implications.

Gate: explicit human acceptance of the render comparison and operational risk.
The backend remains opt-in until repeated production renders establish stability.

## Benchmark matrix

Every performance result must include:

| Axis | Required values |
|---|---|
| Projection | QKV, attention output, fc1, fc2 |
| Sequence tile | 512, 1024, 2048, 4096; retain the winner |
| Weight mode | constant, dynamic prepacked, dynamic staged |
| Engine mode | GPU, ANE 0, ANE 1, dual ANE, GPU + dual ANE |
| Data | deterministic synthetic and real checkpoint/activation capture |
| State | first run, warm median, sustained loop |
| Output | dispatch, copy, join, wall time, TFLOP/s, numerical error |

Use interleaved ABBA sampling and medians, as the existing performance tests do.
No other GPU workload may run during a control sweep. Compile time and first
page-in time are reported but excluded from steady-state dispatch comparisons.

## Stop conditions

Stop or redesign before production integration if any of the following is true:

- A production shape cannot be compiled or placed through the direct runtime.
- Dynamic weights fall below the 1.20x full-forward projection gate.
- Weights must be copied or transformed by the CPU every dispatch and measured
  end-to-end results do not clear the gate.
- GPU and ANE submissions serialize or contend enough to miss 0.85 overlap
  efficiency.
- FP16/ANE error grows outside a stable class across one block or a trajectory.
- Either ANE instance is unavailable or cannot be selected reliably on M3 Ultra.
- Resident surfaces materially raise memory pressure or destabilize the current
  62 GiB checkpoint path.
- macOS selector drift cannot be detected safely before sampling starts.

## Definition of success

The research succeeds when it produces a reproducible answer, including a
negative one. The backend succeeds only when an M3 Ultra can run the existing
H3 render path with:

- GPU attention and GPU linear slices executing concurrently with two ANE
  linear slices;
- no per-dispatch CPU weight packing or activation readback;
- no silent fallback during a trajectory;
- at least 1.20x median full-forward speedup, with 1.35x or better the target;
- numerical and perceptual acceptance under the repository's conformance
  strategy;
- unchanged behavior when the experimental backend is disabled or unavailable.

## Research sources

- [OMLX experimental Qwen 3.5 ANE prefill notes](https://github.com/jundot/omlx/blob/main/docs/experimental/qwen35_ane_prefill.md) — dual-instance execution, weight banks, and the measured per-instance address window.
- [OMLX source](https://github.com/jundot/omlx) — Objective-C++ custom primitive and private ANE runtime integration under Apache-2.0.
- [maderix/ANE](https://github.com/maderix/ANE) — in-memory compilation, IOSurface I/O, dynamic-weight MIL, and GPU/ANE shared-surface experiments.
- [Apple: Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers) — ANE-friendly layout and the relationship between sequence length and bandwidth-bound execution.
- [Apple M3 Ultra announcement](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/) — dual Neural Engine configuration and advertised aggregate memory bandwidth.
- [Apple Neural Engine: Architecture, Programming, and Performance](https://arxiv.org/abs/2606.22283) — reverse-engineered compiler, driver, firmware, program, and performance model.
- [ANEForge](https://github.com/sbryngelson/ANEForge) — direct ANE tooling and program segmentation evidence; its approximately 2 GiB single-program constraint is separate from OMLX's approximately 4 GiB resident-weight window.
