# H3 QKV Projection Spike

Date: 2026-08-26  
Host: M3 Ultra, macOS build 25F84, ANE architecture `h15g`  
Tool: `Tools/ANE/projection-spike.mm`

> This page records one production QKV projection. For where the whole ANE question stands today, and which of its conclusions have since been corrected, see `docs/ANE_STATUS.md`.

## Outcome

A production-shape H3 QKV projection is faster when split across the GPU and
both ANE dies. The measured kernel-seam speedup is 1.27–1.32x after including
activation conversion, two host synchronization joins, and output conversion
and concatenation.

The performance gate passes. The numerical result originally recorded here —
3.17% relative RMS on the ANE shards — was **an artifact of the test fixture,
not the engine**: a third of its products fell below fp16's smallest normal and
were flushed to zero. Against an FP64 oracle with operands scaled off the
denormal boundary the engine reaches 2.07e-4, matching its arithmetic floor.
See "Resolved 2026-08-26" below. Actual checkpoint weights and captured block
activations must still be evaluated, and the quantity to measure is the
fraction of products below 6.10e-5, not the cancellation ratio.

## Projection and split

The spike uses the real QKV geometry and layouts from `AttentionLayer`:

```text
input       [S,K]       bf16    [2048,5376]
weight      [N,K]       bf16    [21504,5376] checkpoint layout
operation   input × weight.T
output      [S,N]       bf16    [2048,21504]
```

The checkpoint weight is transformed once into the resident `[K,N]` layout.
The measured balanced output-channel split is:

```text
GPU          N=15360    bf16 MPSGraph matmul
ANE hint 1   N= 3072    fp16 two-input MIL matmul
ANE hint 2   N= 3072    fp16 two-input MIL matmul
```

The two ANE shards are labelled by their instance *hint*, not by a physical
die: the hint does not choose the engine, and both shards land on two dies only
because they are submitted concurrently. See `docs/ANE_OVERLAP_RESULTS.md`.

The first trial assigned 3,584 channels to each ANE and 14,336 to the GPU. It
was ANE-heavy and reached only 1.12–1.14x. Moving to 3,072 channels per ANE
balances the concurrent phase and reaches 1.27–1.32x.

## Integration path exercised

The spike includes the boundary operations a real projection requires:

1. A canonical bf16 `[S,K]` Metal buffer is transposed and converted into the
   shared FP16 ANE activation IOSurface by a Metal kernel.
2. The GPU consumes the bf16 activation and its contiguous output shard.
3. Both pinned ANE programs consume the same activation IOSurface and disjoint
   resident weight IOSurfaces.
4. The three processors execute concurrently.
5. A Metal join kernel transposes the ANE outputs, converts FP16 to bf16, and
   places all three shards into canonical `[S,N]` order.

The GPU portion of the joined output is byte-identical to the full GPU oracle,
covering 31,457,280 elements. Both ANE output surfaces are fully populated and
joined in the expected channel ranges.

## Measurements

Across three independent seven-sample ABBA runs:

| Measurement | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Full GPU QKV | 26.179 ms | 26.058 ms | 25.681 ms |
| Hybrid end to end | 19.877 ms | 19.823 ms | 19.814 ms |
| Speedup | 1.317x | 1.315x | 1.296x |

A separately instrumented run measured:

```text
bf16 -> fp16 activation conversion     0.345 ms
concurrent GPU + ANE1 + ANE2          19.061 ms
  ANE instance 1                      18.930 ms
  ANE instance 2                      18.868 ms
fp16 -> bf16 canonical output join     0.547 ms
hybrid end to end                     20.325 ms
full GPU                              25.832 ms
speedup                                1.271x
```

Resident storage for this one projection is:

```text
canonical bf16 weight                 220.5 MiB
two FP16 ANE weight shards             63.0 MiB additional
activation                             21.0 MiB
joined output                          84.0 MiB
```

## Numerical comparison

The fixture uses independent deterministic bf16 activation and weight values,
not the earlier periodic values that canceled almost completely.

| Region | Maximum absolute error | Relative RMS |
|---|---:|---:|
| GPU shard | 0 | 0 |
| ANE shard 1 | 0.00241089 | 0.0316846 |
| ANE shard 2 | 0.00244141 | 0.0317181 |
| Whole joined QKV | 0.00244141 | 0.0169467 |

This measures the intended production difference: bf16 operands and output on
the GPU versus bf16 operands rounded to FP16 at the ANE boundary. The direct
FP16 differential harness already established that the ANE kernel matches an
FP16 CPU oracle; this result measures the model precision change, not a binding
or layout error.

### Resolved 2026-08-26: the 3.17% is denormal flush, not the engine's arithmetic

`projection-spike.mm` now scores both processors against an **FP64 oracle**
instead of against each other, over eight columns per shard at every row.

| processor | max abs vs FP64 | rel RMS vs FP64 | cancellation |
|---|---:|---:|---:|
| GPU bf16 | 0.000121759 | **0.00165764** | 54.6x |
| ANE fp16 shard 1 | 0.00200897 | **0.0315775** | 54.7x |
| ANE fp16 shard 2 | 0.00193116 | **0.0318546** | 55.2x |

The first thing this settles is that the 3.17% is **not** an artifact of
comparing two approximations: the GPU contributes only 0.17%, and the ANE
shards score the same against an exact reference as they did against the GPU.

The second thing is that the engine's own error model — `2^-12 * C / sqrt(K)`
from `Tools/ANE/numerics.m` — predicts 0.018% here and is wrong by 173x. That
discrepancy is the finding.

**Absolute error, not relative.** The differential fixture's products have RMS
magnitude 2.1e-2 and its absolute error matches per-product fp16 rounding
exactly (3.16e-4 measured, 3.76e-4 predicted). This fixture's products are
100x smaller at 2.0e-4, yet its absolute error is only 1.5x different
(4.65e-4). An error that barely moves when the operands change by two orders of
magnitude is a floor, not a rounding.

**The floor is denormal flush.** `numerics.m` measured that this engine flushes
denormals to zero inside the multiply-accumulate. fp16's smallest normal is
6.10e-5, and for this fixture's `|x| ~ U(0, 0.03)` against `|w| ~ U(0, 0.02)`,
`P(|x*w| < 6.10e-5) = c(1 - ln c) = 0.334`. **A third of the products are
silently dropped.**

`H3_SPIKE_SCALE` scales both operands together, so products scale by the
square. Per-product rounding is relative and must be scale-invariant; a
denormal floor must weaken as operands grow. The sweep decides it:

| fixture scale | GPU rel RMS | ANE rel RMS |
|---:|---:|---:|
| 1x | 0.00165764 | **0.0315775** |
| 4x | 0.00165764 | 0.00147276 |
| 16x | 0.00165764 | 0.000224582 |
| 64x | 0.00165764 | **0.000207435** |

The GPU is invariant to seven significant figures. The ANE falls **152x** and
plateaus at 2.07e-4 against the model's predicted 1.82e-4 — agreement to 14%,
which is the arithmetic floor the model always described. Everything above that
plateau was underflow.

**Consequences.**

The engine is roughly 150x better at this projection than this document
previously recorded. The precision gate was being failed by the fixture.

The question the production gate must ask is **not** cancellation. It is *what
fraction of real DiT products fall below 6.10e-5*, per projection.

> **Answered 2026-08-26 — `docs/ANE_PRECISION_RESULTS.md`.** Measured against
> real captured block taps: underflow runs 0.02%-6.7% and costs nothing, and
> the engine beats the bf16 GPU path by 3-20x per projection. Precision is not
> the blocker. Saturation is: block 49's fc2 breaches 2^15 on real data and
> silently zeroes 0.02% of its outputs.

And there is a cheap fix if the answer is unfavourable. A linear projection is
homogeneous, so scaling activations by a power of two on the way in and
dividing the output by the same factor on the way out is **exact in fp16** and
moves the product distribution off the denormal boundary. This is the
loss-scaling trick from FP16 training, and here it costs one multiply at each
boundary. It should be measured before any conclusion that FP16 breaches
contract 8.

## What the spike does not hide

The spike begins with a realized Metal activation buffer and ends with a joined
Metal output buffer. Current public MLX Swift APIs can expose an MLX allocation
with `asMTLBuffer(noCopy: true)`, but that call evaluates and drains the MLX
graph. They do not provide a public zero-copy constructor that adopts the joined
buffer back into MLX. A naive `Projection` implementation would therefore pay
the crossing cost already measured in `MPSIntegrationTests`, which is much
larger than this projection's approximately 6 ms hardware win.

Consequently, this should not be wired into `AttentionLayer` as four blocking
calls. The viable integration choices are:

- an MLX custom primitive whose evaluation owns the Metal/ANE split and returns
  an MLX-managed output;
- a patched MLX buffer-adoption and command-stream interface; or
- a larger fused block/step boundary that amortizes one drain and one adoption.

The custom primitive is the smallest next experiment. It can preserve the
existing projection signature while keeping all conversion, dispatch, and join
work below the lazy-graph boundary.

## Reproduction

```sh
xcrun clang++ -O2 -std=c++17 -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -framework Metal \
  -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
  -ldl Tools/ANE/projection-spike.mm -o /tmp/h3-ane-projection-spike
/tmp/h3-ane-projection-spike
```

`H3_SPIKE_SCALE` multiplies both operands, so products scale by the square; it
is the knob that separates a relative rounding error from an absolute
underflow floor. `H3_CHECKPOINT` and `H3_BLOCK` replace the synthetic fixture
with real weights.

```sh
for g in 1 4 16 64; do H3_SPIKE_SCALE=$g /tmp/h3-ane-projection-spike; done
```

## Decision

Proceed with a single custom-primitive prototype for QKV only, gated to
h15g/25F84 and disabled by default. Before routing production:

1. replay captured block-0, block-24, and block-49 QKV inputs and real weights;
2. measure Q, K, and V separately so head-aligned shard boundaries are kept;
3. run the existing block oracle and render-level conformance suite;
4. measure the fraction of real products below fp16's smallest normal (6.10e-5)
   per projection, and apply a power-of-two operand scale if it is material —
   the scale is exact in fp16 and costs one multiply at each boundary;
5. reject the FP16 route only if the checkpoint distribution still breaches
   contract 8 after that scaling;
6. confirm the primitive removes the MLX drain/adoption cost rather than merely
   moving it.

