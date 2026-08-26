# ANE Differential Test Results

Date: 2026-08-26  
Host: M3 Ultra, macOS build 25F84, ANE architecture `h15g`, two instances  
Tool: `Tools/ANE/differential.m`

## Result

The direct AppleNeuralEngine runtime accepts activation and weight tensors as
two independent FP16 IOSurface inputs. The weight surface can be changed between
evaluations without recompiling or reloading the program. Its result agrees
with both a compiled-constant convolution and a CPU FP32-accumulation oracle.

This changes the proposed implementation: use a separate weight input. Do not
adopt oMLX/maderix's packed activation-plus-weight tensor unless a later kernel
requires it. Separate inputs avoid a per-dispatch pack, avoid slice-offset
quirks, and let the same activation surface be paired with different resident
weight surfaces.

`_ANERequest.weightsBuffer` is not a usable dynamic-weight path in the tested
form. The request and evaluation both succeed, but a replacement weight buffer
does not affect the output. The installed runtime's adapter-weight entitlement
is still the likely explanation; acceptance of the field is not evidence that
the buffer is bound.

## Differential matrix

All cases use deterministic FP16 inputs and weights. The CPU oracle multiplies
the FP16 values with FP32 accumulation, then rounds the expected result to FP16.
The second evaluation changes only the weight payload from W1 to W2.

| Path | Compile/load | W1 correct | W2 correct | Runtime mutation | Decision |
|---|---:|---:|---:|---:|---|
| Static FP16 convolution | yes | yes | n/a | n/a | semantic control |
| Static plus `weightsBuffer=W2` | yes | yes | no | no | reject |
| Two ordinary tensor inputs | yes | yes | yes | yes | primary path |
| One packed tensor input | yes | yes | yes | yes | fallback only |

For K=N=S=64, the two-input path produced maximum absolute errors of
`1.53e-4` for W1 and `1.22e-4` for W2. Relative RMS error was `8.31e-4` and
`6.90e-4`; cosine similarity exceeded `0.9999996`.

The same mutation and agreement passed at:

| K | N | S | W1 relative RMS | W2 relative RMS | Separate-input median |
|---:|---:|---:|---:|---:|---:|
| 192 | 320 | 64 | 0.00331 | 0.00156 | 0.119 ms |
| 256 | 256 | 128 | 0.00153 | 0.00127 | 0.138 ms |
| 512 | 512 | 64 | 0.00425 | 0.00198 | 0.148 ms |
| 1024 | 1024 | 128 | 0.00338 | 0.00256 | 0.190 ms |
| 5376 | 256 | 64 | 0.01349 | 0.00626 | 0.323 ms |
| 7168 | 256 | 64 | 0.01898 | 0.00733 | 0.317 ms |
| 14336 | 128 | 32 | 0.01437 | 0.00990 | 0.368 ms |
| 5376 | 3072 | 64 | 0.01359 | 0.00626 | 1.174 ms |
| 5376 | 3584 | 64 | 0.01358 | 0.00626 | 1.338 ms |
| 5376 | 4096 | 64 | 0.01360 | 0.00626 | 1.455 ms |
| 5376 | 6144 | 64 | 0.01359 | 0.00667 | 2.144 ms |

These timings are dispatch microbenchmarks, not an end-to-end projection. They
exclude GPU/ANE handoff, output joining, and any weight movement required by
the final ownership design.

> **The numerical column above is an artifact of this fixture, corrected
> 2026-08-26.** The original reading — "error grows with the contraction K" —
> is not what the table measures. `FillInputs` builds correlated sinusoids,
> `sin(17k+13s)` against `cos(11k+19n)`, whose dot product cancels almost
> completely: at K=5376 it sums terms of RMS magnitude 91.5 to produce a result
> of 0.023, a cancellation of 3,900:1. Relative error against a deliberately
> near-zero denominator is a property of the fixture, not of the engine.
>
> Measured across a K sweep at fixed N=256, S=64, the cancellation ratio runs
> 285x, 470x, 3237x, 3903x, 6538x — and `rel_rms` tracks *that* curve, not K.
> It actually **falls** from 0.0161 at K=2688 to 0.0135 at K=5376 while K
> doubles. What grows cleanly is absolute error, as `4.31e-6 * sqrt(K)`, to
> within 2% across four doublings.
>
> The engine's own behaviour, established by `Tools/ANE/numerics.m`, is
> `rel_error ~= 2^-12 * cancellation / sqrt(K)`, which reproduces the measured
> 1.35% at K=5376 to within 4%. **Larger K is better, not worse**, at equal
> conditioning: fc2 at K=14,336 is the *least* error-sensitive of the four
> projections, not the most. It remains the slowest (1.81 TFLOP/s), which is a
> separate objection.
>
> The production precision gate therefore needs one number nobody has measured:
> the cancellation ratio of real DiT activations, per projection. That is a CPU
> computation over captured block inputs and needs no ANE at all.

The static constant-weight path and both dynamic paths return **bit-identical**
error at K=5376 (`rrms=0.0134911`, max `0.00128174`). The arithmetic is a fixed
hardware property, not a consequence of the lowering, so there is no better
route to look for.

## Physical tensor layout

An IOSurface-backed rank-four tensor `[1,C,1,W]` uses a 64-byte-aligned row for
each channel. For FP16, the physical row stride is:

```text
rowBytes = alignUp(W * 2, 64)
surfaceBytes = C * rowBytes
```

A tightly packed allocation appears correct whenever W is a multiple of 32,
which made the first square probes pass. At odd widths the runtime frequently
returned success with severely wrong output. After padding every channel row,
the separate-input path passed at S=1, 17, 31, 33, 48, 63, and 65 as well as
the aligned controls.

This is a hard bridge invariant. A successful private API return is never a
sufficient test; every new shape and layout needs a numerical oracle.

The packed path is more fragile. Some odd slice offsets produce roughly 1.5–2%
relative RMS error even with correct outer row padding, while the separate-input
path stays near 0.1%. That reinforces the decision to avoid slicing weights out
of a combined spatial dimension.

## Request fields

`procedureIndex` values 0 through 15 were accepted for a one-procedure program.
Index 15 produced a byte-identical result to index 0. Treat the value as ignored
or mapped to the same procedure in this case, not as evidence of sixteen valid
entry points.

`weightsBuffer` behaved similarly: construction and evaluation succeeded, but
substituting W2 left the W1 output byte-identical. It must remain `nil` in the
proposed bridge unless a separately declared adapter-weight MIL program and the
required entitlement prove otherwise.

## Synchronization and two-die behavior

Reading the output IOSurface immediately after `evaluateWithQoS` returned the
complete correct output in every case. For this synchronous request form,
evaluation is therefore a host-visible completion point. That does not yet
establish efficient Metal interoperability; shared-event and GPU-overlap tests
remain required.

The execution options used by oMLX were reproduced:

```text
kANEFProcedureVariantHint = 1
kANEFAneInstanceHint = 1 or 2
```

Both instance hints compiled, loaded, mutated weights, and matched the oracle.
With K=N=1024, S=128 and 2,000 evaluations per phase, a single process measured
0.153 ms. After the first cold concurrent round, two simultaneous processes
using hints 1 and 2 measured 0.146–0.149 ms each in rounds two and three. This
supports independent physical-die execution under concurrency.

> **It does not show that the hint chose the die, corrected 2026-08-26.**
> Per-die energy telemetry shows a job submitted alone with
> `kANEFAneInstanceHint = 2` running on die 0 while die 1 stays powered down at
> 0 mJ. The second die is engaged by concurrency — the kernel load-balancer
> placing an independent submission on the idle engine — not by the hint. See
> the telemetry section of `docs/ANE_OVERLAP_RESULTS.md`.

## Reproduction

Build and run the default differential:

```sh
xcrun clang -O2 -Wall -Wextra -Werror -fobjc-arc \
  -framework Foundation -framework IOSurface -ldl \
  Tools/ANE/differential.m -o /tmp/h3-ane-differential
/tmp/h3-ane-differential
```

Override dimensions and pin an instance at compile time:

```sh
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
  -DDIFF_K=5376 -DDIFF_N=4096 -DDIFF_S=64 \
  -DDIFF_ITERATIONS=100 -DDIFF_DYNAMIC_ONLY -DDIFF_ANE_INSTANCE=1 \
  Tools/ANE/differential.m -o /tmp/h3-ane-differential-i1
```

## Next differential

The feasibility question is now narrower:

1. Wrap a Metal buffer and ANE tensor around the same IOSurface and compare
   zero-copy visibility against explicit CPU staging.
2. Run a long GPU GEMM concurrently with instance-pinned ANE requests and
   compare isolated versus overlapped wall time and throughput.
3. Determine whether one activation surface can be mapped into both ANE
   instances without a duplicate copy.
4. Measure production tiles and output-channel shards with resident weights,
   including the 64-byte row stride and full transfer/join costs.
5. Run block-level and render-level bf16-GPU versus fp16-ANE conformance before
   integrating an execution backend.

