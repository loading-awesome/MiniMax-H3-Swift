# ANE precision against real production tensors

Date: 2026-08-26
Host: M3 Ultra, macOS build 25F84, ANE architecture `h15g`
Tools: `Tools/ANE/underflow.py`, `Tools/ANE/numerics.m`
Data: `parity/goldens/oracle_block00_prod` and `oracle_prod_matrix/b{00,24,49}_c{000,013,019}`
      against `MiniMax-H3-FL2VA_bf16.safetensors`

## Result

**The precision objection to the FP16 ANE path is wrong, and the real blocker is
somewhere else.** Measured against real captured block activations and real
checkpoint weights, the engine's arithmetic costs 7e-5 to 5e-4 relative RMS per
projection — **3 to 20 times more accurate than the bf16 GPU path it would
replace**, which scores 1.66e-3 against the same kind of exact reference.

What does fail is saturation. At block 49 the `fc2` projection drives interior
partials to **34,649** against the 2^15 = 32,768 threshold at which this engine
returns **zero** instead of a result. On real data, 0.02% of that projection's
outputs are destroyed, which is enough to take its relative error from 1.0e-4 to
**0.101**. It is silent: no inf, no NaN, nothing downstream can detect it.

A power-of-two scale on the operand fixes it exactly, at no precision cost.

## Why the earlier numbers were wrong

`docs/ANE_PROJECTION_SPIKE.md` recorded 3.17% and treated it as the engine's
error. It was the fixture's. That fixture used activations at RMS ~0.017 and
weights at RMS ~0.012; the real model runs activations from 0.1 to 293 and
weights near 0.13, so a third of the fixture's products fell below fp16's
smallest normal and were flushed to zero while almost none of the model's do at
the same rate. The engine was being blamed for underflow that this model does
not have.

## Method

Two things had to be real, and both now are.

**Real activations.** The block oracles carry F32 taps of the exact tensor each
projection consumes: `ref.mod_scale_shift.1` into qkv, `ref.attn.sdpa` into the
attention output projection, `ref.mod_scale_shift.2` into fc1, and
`ref.mlp.swiglu` into fc2. No distribution is assumed anywhere.

**Real weights**, read from the bf16 checkpoint.

The engine's arithmetic is **simulated**, not assumed: per-product rounding to
fp16, denormals flushed, an exact wide accumulator, and the 2^15 output-port
saturation — the datapath `Tools/ANE/numerics.m` measured on this silicon.
`--validate` checks that simulation against hardware before any of its
predictions are used:

| fixture scale | measured on silicon | simulated | ratio | underflow |
|---:|---:|---:|---:|---:|
| 1x | 0.0315775 | 0.0878939 | 2.78x | 32.5% |
| 4x | 0.00147276 | 0.00186801 | 1.27x | 3.9% |
| 16x | 0.000224582 | 0.000201899 | 0.90x | 0.3% |
| 64x | 0.000207435 | 0.00019215 | 0.93x | 0.0% |

Agreement is 0.90–0.93x in the low-underflow regime the real model occupies.
The 2.78x at 32% underflow is the simulation being **pessimistic**, which is the
safe direction, and the real model never gets near that rate.

## Measured

Sixty-four rows by sixty-four output channels per projection.

### Block 0, first timestep

| projection | K | act RMS | \|w\| RMS | underflow | max\|partial\| | ANE relRMS |
|---|---:|---:|---:|---:|---:|---:|
| qkv | 5376 | 0.299 | 0.135 | 1.15% | 36 | 1.3e-4 |
| attn out | 7168 | 0.837 | 0.082 | 0.72% | 98 | 1.1e-4 |
| mlp fc1 | 5376 | 0.108 | 0.132 | 6.24% | 12 | 5.0e-4 |
| mlp fc2 | 14336 | 8.09 | 0.145 | 6.65% | 4321 | 1.5e-4 |

### Block 24

| projection | act RMS | underflow | max\|partial\| | ANE relRMS |
|---|---:|---:|---:|---:|
| qkv | 0.557 | 0.71% | 35 | 1.8e-4 |
| attn out | 3.68 | 0.12% | 343 | 1.7e-4 |
| mlp fc1 | 0.241 | 1.24% | 24 | 2.1e-4 |
| mlp fc2 | 5.67 | 3.20% | 894 | 1.8e-4 |

### Block 49 — the failure

| projection | act RMS | underflow | max\|partial\| | ANE relRMS | |
|---|---:|---:|---:|---:|---|
| qkv | 2.98 | 0.31% | 485 | 7.2e-5 | |
| attn out | 31.3 | 0.02% | 3925 | 2.0e-4 | |
| mlp fc1 | 1.06 | 0.48% | 72 | 1.2e-4 | |
| **mlp fc2** | **293** | 4.92% | **34649** | **0.101** | **0.02% of outputs ZEROED** |

Activation scale grows sharply with depth — fc2's input goes from RMS 8 at
block 0 to 293 at block 49 — and the partial-sum envelope grows with it.

## Two conclusions

**Underflow is a non-issue for this model.** It runs from 0.02% to 6.7%, and
even at 6.65% the relative error is 1.5e-4. The denormal flush that dominated
the synthetic fixture 152x costs this model nothing measurable.

**Saturation is the real gate, and it is not a precision gradient.** It is a
cliff that returns zero. A single destroyed element in 4,096 moves a
projection's relative error by three orders of magnitude, and nothing in the
output signals it. The quantity that must be bounded before any shape is
enabled is `max|interior partial|`, per projection and per block — not the
output magnitude, which stays small precisely because the partials cancel.

## The fix, measured

A linear projection is homogeneous, so scaling the activation by a power of two
and dividing the output by the same factor is exact in fp16 and changes nothing
about the arithmetic. It moves the partial-sum envelope off the threshold.
Block 49 fc2:

| operand scale | max\|partial\| | headroom | underflow | ANE relRMS |
|---:|---:|---:|---:|---:|
| 1 | 34,649 | **breached** | 4.92% | **0.101** |
| 1/4 | 8,662 | 3.8x | 6.14% | 1.02e-4 |
| 1/16 | 2,166 | 15.1x | 7.86% | 1.02e-4 |

The window is wide: partials fall by the scale factor while products remain
four orders of magnitude above the denormal boundary, so underflow rises only
from 4.9% to 7.9% and the relative error does not move at all. **1/16 buys 15x
headroom for free.**

## Caveats

`max|partial|` here is a **lower bound**. It is a maximum over a 64x64 sample of
a projection that is 15,406 rows by up to 28,672 columns, so the true envelope
over a full tensor is higher and the real margin is tighter than these numbers
show. Block 49 fc2 already breaches on the sample; the others need a full-tensor
bound before any of them is called safe.

These are single-block, single-projection errors. Propagation through 50 blocks
and 20 steps is not measured here, and a per-projection improvement over bf16
does not automatically survive that.

The simulation is validated at four points on one fixture family. It should be
re-validated on silicon at a real production shape before it gates a release.

## What this changes

Contract 8 pins the DiT at bf16 as its verified ceiling. Per projection, the
FP16 ANE path is **more** accurate than that ceiling, not less, so the objection
that has been standing against this work since the beginning does not survive
contact with real tensors.

The gate that replaces it is a saturation bound with margin, plus a per-block
operand scale where the bound is tight. Both are cheap, both are exact, and
neither was on the plan's list before this measurement.

## Reproduction

```sh
python3 Tools/ANE/underflow.py --validate

python3 Tools/ANE/underflow.py \
  --golden  parity/goldens/oracle_prod_matrix/b49_c019/oracle.safetensors \
  --checkpoint /path/MiniMax-H3-FL2VA_bf16.safetensors \
  --block 49 --scale 0.0625
```
