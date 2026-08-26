# Metal and ANE Overlap Results

Date: 2026-08-26  
Host: M3 Ultra, macOS build 25F84, ANE architecture `h15g`  
Tool: `Tools/ANE/overlap.mm`

> This page records concurrency, per-die energy, DRAM bandwidth. For where the whole ANE question stands today, and which of its conclusions have since been corrected, see `docs/ANE_STATUS.md`.

## Result

Metal and both physical ANE instances execute concurrently on this host. A
Metal-produced activation can be consumed directly by ANE through the same
IOSurface without a CPU payload copy. Two ANE programs can read the same activation and weight surfaces
concurrently, and a compute-bound GPU GEMM can run beside both without a
measurable increase in total wall time. **Concurrency, not the instance hint,
is what engages the second die** — see the telemetry section.

This validates the central side-channel hypothesis. It does not yet validate a
complete DiT backend: model integration, dependency scheduling, output joining,
weight residency, and render-level FP16 conformance remain.

## Workloads

The GPU control is an MPS FP16 GEMM:

```text
[4096,4096] × [4096,4096]
137.44 GFLOP per evaluation
15.6–16.4 TFLOP/s measured host-visible rate
```

Each ANE instance runs the selected two-input dynamic MIL matmul:

```text
activation [1,5376,1,2048]
weight     [1,5376,1,4096]
output     [1,4096,1,2048]
90.19 GFLOP per evaluation
3.65–3.69 TFLOP/s per ANE instance
```

The dimensions represent a production H3 contraction K and sequence tile with
a 4,096-channel output shard. Both ANE instances deliberately share the same
activation and weight IOSurfaces to test concurrent read mapping; they write to
separate output surfaces.

## Zero-copy producer/consumer test

Metal creates an `R16Float` texture view over the activation IOSurface and fills
all values with one. No CPU code writes or repacks that activation. ANE then
multiplies it by a dynamically supplied rectangular identity matrix from a
second IOSurface.

```text
Metal texture -> shared IOSurface -> ANE ordinary tensor input
maximum output error: 0
result: PASS
```

Waiting for the Metal command buffer before calling ANE is sufficient for this
producer/consumer control. The next integration step should replace the host
wait with the least-serializing event mechanism that remains correct.

## Coexecution measurements

Each reported run contains 15 warm samples. Three independent repetitions gave:

| Measurement | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| GPU isolated | 8.396 ms | 8.598 ms | 8.794 ms |
| ANE instance 1 isolated | 24.703 ms | 24.502 ms | 24.445 ms |
| ANE instance 2 isolated | 23.931 ms | 24.032 ms | 24.019 ms |
| GPU + ANE 1 wall | 23.940 ms | 23.897 ms | 23.892 ms |
| ANE 1 + ANE 2 wall | 23.937 ms | 23.818 ms | 23.883 ms |
| GPU + ANE 1 + ANE 2 wall | 23.963 ms | 23.835 ms | 23.961 ms |

For one GPU plus one ANE, serialized work would take 33.1 ms; concurrent wall
time is 23.9 ms, a 1.38–1.39x speedup for that workload pair. The GPU hardware
duration during overlap was 7.19–7.25 ms, showing no slowdown relative to the
isolated host-visible measurement.

For both ANEs, serialized work would take about 48.5 ms; concurrent wall time is
23.8–23.9 ms, a 2.03x speedup. For all three processors, serialized work would
take about 57.1 ms; concurrent wall time is 23.8–24.0 ms, a 2.38–2.40x speedup.

The three-way speedup is a concurrency diagnostic, not the projected DiT
speedup: these three synthetic jobs contain different quantities of arithmetic.

## Implication for H3

At this tested dynamic-weight shape, two ANEs contribute approximately 7.3
TFLOP/s beside the project's roughly 16 TFLOP/s GPU ceiling. If eligible DiT
linears split efficiently, their compute-rate ceiling rises from about 16 to
23.3 TFLOP/s, or 1.46x for the linear portion.

Using the current forward decomposition—355 TFLOP attention remaining on the
GPU and 606 TFLOP eligible linears—the optimistic dependency-respecting bound
is:

```text
attention: 355 / 16.0          = 22.2 s
linears:   606 / (16.0 + 7.3) = 26.0 s
forward:                         48.2 s
speedup over 60.0 s:             1.25x
```

That is now an evidence-based hardware ceiling for this dynamic FP16 route, not
an implementation result. Transfer, conversion, joining, tails, and changed
GPU GEMM efficiency can reduce it. Better ANE lowering or static resident
weight banks could raise it.

## Engine telemetry — the instance hint does not pin

*Added 2026-08-26 after instrumenting every arm with IOReport (`Tools/ANE/counters.h`).*

The timings above cannot say **which** die ran a job: `ane2_isolated_ms` reads
the same whichever engine executed it. Per-die energy can, because a die that
did nothing reports exactly 0 mJ. Two consecutive runs, agreeing to under 1%:

| arm | die 0 | die 1 | clock-up d1 |
|---|---:|---:|---:|
| GPU isolated | 0.0 mJ | 0.0 mJ | — |
| ANE 1 isolated | 1204 / 1203 mJ | 0 / 0 mJ | — |
| GPU + ANE 1 | 1207 / 1206 mJ | 0 / 0 mJ | — |
| **ANE 2 isolated** | **1209 / 1198 mJ** | **0 / 0 mJ** | **0%** |
| ANE 1 + ANE 2 | 1201 / 1203 mJ | 1214 / 1216 mJ | 33% |
| GPU + ANE 1 + ANE 2 | 1202 / 1201 mJ | 1184 / 1183 mJ | 100% |

Read the fourth row. A job submitted with `kANEFAneInstanceHint = 2` burned
about 1,200 mJ on **die 0** and nothing at all on die 1, which was also fully
powered down at 0% clock-up residency.

**`kANEFAneInstanceHint` is not pinning anything.** What produces dual-die
execution is submitting two jobs concurrently, at which point the kernel
load-balancer steers the second one to the idle die. With a single job in
flight it goes to die 0 regardless of the hint. That is the documented
behaviour of the balancer — it steers whole independent submissions to the
least-busy engine die — and this measurement is the first direct confirmation
of it on this host.

The dual-die speedup reported above is real and reproduces. The *mechanism* is
not the one this document originally claimed. The distinction matters for the
design: if weights become resident per-die inside a roughly 4 GiB window, a job
steered to the die that does not hold its weights is a correctness problem
rather than a scheduling one. Any bank-residency scheme must therefore verify
placement rather than request it.

### DRAM bandwidth — the spare-bandwidth hypothesis, measured

**Engine-attributed DRAM bytes are not available on this host.** The published
byte channels (`AMC Stats|Perf Counters|ANE0 RD`) are M1 and M5 names: there is
no `AMC Stats` group on M3 Ultra, and the per-agent `PMP*|AGENT BW` group lists
only IO agents.

What is available is better suited to the actual question. `PMP*|DCS BW` is a
32-bin bandwidth histogram per memory controller — 64 GB/s per bin, to 2048
GB/s — reporting **total** DRAM bandwidth per die. The hypothesis was never
really "how many bytes does the engine pull", it was "is there room", and total
demand against the ceiling answers that directly.

| arm | peak read d0 / d1 | time above 64 GB/s d0 / d1 |
|---|---:|---:|
| GPU isolated | 128 / 128 GB/s | 0.1% / 0.1% |
| ANE 1 isolated | 128 / 64 | 7.0% / 0.0% |
| GPU + ANE 1 | 128 / 128 | 14.5% / 0.1% |
| ANE 2 isolated | 128 / 64 | 5.8% / 0.0% |
| ANE 1 + ANE 2 | 128 / 128 | 4.4% / 0.6% |
| GPU + ANE 1 + ANE 2 | 128 / **192** | 11.2% / 8.7% |

**Peak DRAM read never exceeds 192 GB/s in any arm**, against roughly 800 GB/s
of aggregate unified-memory bandwidth, and the controllers sit in their lowest
bin for at least 85% of every arm. The spare-bandwidth hypothesis holds with a
very large margin, and it is now measured rather than inferred from the GPU
failing to slow down.

Two caveats on reading the table. Bin 0 is labelled 64 GB/s and covers
everything below it, so a residency-weighted mean can never report under 64 and
is an upper bound rather than a measurement — only the peak bin and the time
above the floor are used here. And this is total controller traffic, so it
cannot attribute bytes to the engine; the arms are differenced instead.

Clock-up residency is DVFS state, not work: the engine stays in `ACT` for an
idle timeout after its last dispatch, so 0% is conclusive that a die did
nothing and anything above 0% is not evidence that it did.

## Reproduction

```sh
xcrun clang++ -O2 -std=c++17 -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface \
  -framework Metal -framework MetalPerformanceShaders -framework CoreFoundation \
  -ldl -I Tools/ANE Tools/ANE/overlap.mm -o /tmp/h3-ane-overlap
/tmp/h3-ane-overlap
```

## Decision

Proceed to an integration spike with these constraints:

- separate activation and weight tensor inputs;
- 64-byte physical row strides;
- one program per ANE instance, **without relying on the hint to place it**
  (see the telemetry section: the hint does not pin);
- shared activation and weight IOSurfaces where ownership permits;
- disjoint output-channel shards and separate output surfaces;
- a correctness gate around every new shape and OS profile;
- no `weightsBuffer` dependency;
- measure the synchronization join and full block before expanding coverage.

The QKV projection spike has now been performed. See
`docs/ANE_PROJECTION_SPIKE.md`: the balanced hybrid is 1.27–1.32x faster at the
kernel seam, but its FP16 shards require real-model conformance before routing.
