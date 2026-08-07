# MiniMax H3 acceleration: what worked, what failed, and what users should try

This is a public summary of the performance work carried out while building the
Swift/MLX implementation of MiniMax H3 for Apple Silicon. It is written for
people who are comfortable building Comfy workflows, changing node settings,
comparing renders, and reading benchmark tables, but who do not necessarily
write model kernels.

The short version is simple:

- Cross-step feature caching was the only large, practical speedup.
- A conservative cache limit improved that result further without a visible
  quality failure in the tested stress set.
- Audio-aware cache decisions mattered, particularly for lip synchronization.
- Sparse attention, partial cache refresh, quantized matrix multiplication,
  graph compilation, tensor fusion, and weight rearrangement either saved too
  little or failed quality/coherence checks.
- A kernel being several times faster in isolation does **not** mean the video
  will render several times faster.

The measurements below are not universal promises. They were collected on an
Apple M3 Ultra using MLX, primarily at 864×480, 124 frames, and 20 sampling
steps. Different GPUs, model builds, node implementations, resolutions, and
frame counts can move the exact numbers. The relative lessons are more useful
than treating any one timing as a guarantee.

## A small glossary

**DiT** — the diffusion transformer. This is the large transformer that does
most of the denoising work. In this implementation, more than 95% of sampling
time is spent inside the DiT.

**Block** — one transformer layer. H3 has 50 of them in the main stack.

**Dense attention** — normal attention, where every allowed token can interact
with every other allowed token.

**Sparse attention** — attention that deliberately visits fewer tokens to save
work. It is an approximation unless the model was designed or trained for that
topology.

**Cross-step cache / first-block cache** — run the first transformer block as a
probe. When the current diffusion step looks sufficiently similar to the last
one, reuse a previously calculated residual instead of running the other 49
blocks.

**Residual** — the change produced by the transformer stack. Reusing the change
is safer than reusing a complete old output, because the change is applied to
the current latent rather than pinning the render to an earlier state.

**Full step** — a sampling step that runs all 50 transformer blocks.

**Reused step** — a sampling step where the cache probe runs, but most of the
stack is skipped.

## The result table

| Approach | Measured result | Quality result | Decision |
|---|---:|---|---|
| Cross-step cache at threshold 0.10 | **1.79×** over dense in repeated controls; 1.93× in the original threshold sweep | About 16% less measured high-frequency detail in the controlled sweep | Use as the balanced/default mode |
| Consecutive cache cap, 3 → 5 | **8.4–9.5%** faster at 20 steps; **24.5%** in the tested 40-step case | Passed viewing across dialogue, fast motion, foliage, and another seed | Use cap 5 |
| Per-stream/audio-aware cache probe | Similar waveform quality; materially stronger lip-sync result | Protected audio/video synchronization | Keep enabled |
| Fused modulation | 0.94% on affected steps; 1.81% wall-clock result in its run set | Changes numerical ordering slightly | Too small |
| AdaLN schedule batching | 0.8–1.3% render-level estimate | 360–729 MiB extra state; shape-dependent numerical drift | Too small and too much state |
| Fused Q/K RMSNorm + RoPE | Kernel about **4.2×** faster locally; only about **1.2%** per render | Small numerical difference | Keep as research fixture only |
| Dense-block graph compilation | 0.54–1.15% | Not bit-identical | Too small |
| Weight-layout/GEMM rearrangement | Exact 3.0–3.1% full-step gain | Exact math, but extra loader/lifecycle complexity | Below the project gate |
| int8/int4 matrix multiplication | About 1.2% overall | Quantized arithmetic adds its own validation burden | No useful compute win here |
| Sol-Attn sparse attention | Small incremental benefit once caching was enabled | Visible temporal pulsing/flicker | Rejected |
| Axial/factorized attention topology | Attractive theoretical reduction | Large error against dense H3 attention | Rejected before a production kernel |
| Prefix-2 partial cache refresh | 7.2% sampling, **4.3% finished-file speed** | Requested dialogue fell from 5/5 to 0/5 keywords | Rejected |

## What worked

### 1. Cross-step caching was the main event

The strongest result came from avoiding transformer work, not making individual
transformer operations a little cheaper.

The cache always runs block 0. It compares that block's residual with the
previous step. If the relative change is below a threshold, it reuses the last
full-stack residual and skips blocks 1–49. The first and last sampling steps
always run in full.

Repeated controls measured:

| Arm | Median seconds per step | Full-step cost | Reused steps | Speedup |
|---|---:|---:|---:|---:|
| Dense | 59.39 | 59.21 s | 0% | 1.00× |
| Cached | 33.23 | 58.84 s | 45% | **1.79×** |

The full-step costs agree within 0.6%. That is an important sanity check: the
cache did not secretly make a full transformer pass cheaper. It made fewer full
passes necessary.

The original threshold sweep showed the tradeoff clearly:

| Cache threshold | Steps skipped | Speedup | High-frequency detail vs dense |
|---|---:|---:|---:|
| 0.05 | 0/20 | None | 0% loss |
| **0.10** | 10/20 | **1.93×** | **−16%** |
| 0.15 | 13/20 | 2.60× | −28% |
| 0.25 | 14/20 | 2.93× | −44% |

Threshold 0.10 was the useful knee. Raising the threshold beyond it continued
to save time, but quality fell much faster than the speed improved.

For Comfy users, the equivalent control may be called `cache threshold`,
`TeaCache threshold`, `relative L1 threshold`, `first-block cache`, or something
node-specific. These implementations are not automatically identical. Treat
0.10 as the measured value for this implementation, not a magic number that can
be pasted into every unrelated caching node.

### 2. Limiting consecutive reuse made the cache safer

The cached residual gets older every time it is reused. A consecutive limit
forces a full refresh before that old residual is carried too far through the
trajectory.

Moving the limit from 3 to 5 was worthwhile:

- 8.4–9.5% faster than cap 3 in the tested 20-step renders.
- 24.5% faster in the tested 40-step render, where more small steps qualified
  for caching.
- Passed viewing on fast motion, a close talking head, foliage/fine detail,
  40 steps, and a second seed.

Increasing a cap does not simply remove a fixed number of refreshes. A forced
refresh resets the counter and changes where later refreshes happen. This is
why apparently obvious arithmetic can give the wrong answer.

Cap 5 is the shipped balance. An unlimited cap is not recommended.

#### The consequence nobody expects: raise your step count, do not lower it

At cap 5 the cache converges on **about ten full evaluations regardless of how
long the schedule is**. Same prompt, same seed, same shape, only the step count
changed:

| steps | full steps | reused | sampling | speedup |
|---:|---:|---:|---:|---:|
| 20 | **10** | 10 | 616.3 s | 1.96x |
| 40 | **10** | 30 | 641.8 s | 3.76x |

**Doubling the step count cost 4%.** The delta curve is set by the flow
schedule, so a finer schedule produces smaller per-step changes, more steps
fall under the threshold, and the cache absorbs them. What survives is the
number of genuinely distinct forwards the trajectory needs — and that is
roughly fixed.

Both directions follow from this, and the second one is the useful one:

* **Lowering the step count buys nothing.** At 10 steps the per-step deltas
  roughly double, most clear the 0.10 threshold, the cache disengages, and you
  run ~10 full steps — landing within a few percent of where 20-step-cached
  already sits, with a coarser trajectory. *(Measured at 20 and 40; the
  10-step case is inferred from the delta curve, not yet rendered.)*
* **Raising it is nearly free.** 40 steps gives twice the sampler updates at
  fine sigma spacing for 4% more wall clock. If you want quality, this is the
  lever.

This also bounds what the cache can ever be. It skips work that is redundant
*within* a trajectory; it cannot go below the number of distinct forwards the
schedule requires. Getting under that needs distillation — a step-distilled
LoRA at 4 steps really does run 4 forwards. On this model that is reported to
cost the soundtrack, which is the same failure this port measured directly in
§3 of *What did not work*.

### 3. Audio needed its own vote

H3 generates video and audio in the same transformer pass. At the tested shape,
video accounts for roughly 95% of the packed rows and audio only about 2.6%.
One average over the complete tensor is therefore almost a video-only decision.

Measured audio residual changes were sometimes substantially larger than the
whole-sequence average. The cache was changed to measure the whole sequence and
the audio rows separately, then reuse only when both were below threshold.

This did not materially change speech waveform correlation or transcription in
the initial A/B. It changed synchronization:

| Cache probe | Reused steps | Lip-sync correlation | Margin over timing-destroyed control |
|---|---:|---:|---:|
| Whole sequence only | 12/20 | 0.664 | 0.453 |
| Whole + audio veto | 10/20 | **0.783** | **0.718** |

The practical lesson is that “the audio sounds okay” is not enough. Check
whether the mouth and words still agree through the entire clip.

### 4. Better measurement worked

The benchmark system did not accelerate a render, but it prevented false
optimizations from shipping. Every render records its prompt digest, seed,
shape, step count, cache configuration, machine, library version, memory peak,
phase timings, and per-step cache decisions.

That exposed several mistakes:

- Timing lazy MLX graphs before evaluation moved work into later phases.
- Comparing runs from different machine states made small gains look real.
- A supposedly aggressive early-sigma cache schedule was contradicted by the
  data: residual changes formed a U-shaped curve, with large changes early and
  late and the quietest region near the middle.
- A downsampled “detail” metric had low-pass filtered away the detail it claimed
  to measure and initially reversed a cache conclusion.

For node users, saving workflow JSON, seeds, model hashes, dimensions, step
counts, and timings alongside outputs is the equivalent discipline.

## What did not work

### 1. Sol-Attn and inference-time sparse attention

Sparse attention looked promising on paper. Attention represented about 37% of
the DiT work at the verified sequence length, and a correctly shortened K/V
operation could be much faster than dense attention.

The problem appeared in complete videos. Sol-Attn introduced temporal pulsing
and flicker that per-call tensor-error metrics did not predict. With caching
already skipping a large fraction of full transformer work, the remaining
incremental speedup was too small to justify that artifact risk.

This is the central warning for attention nodes: a tolerable error on one
attention call does not establish temporal coherence after 50 blocks and 20
diffusion steps.

### 2. Spatial/temporal or axial factorization was not equivalent for H3

Splitting full 3D attention into spatial and temporal passes can be exact for a
model designed around factorized attention. It is **not** mathematically
equivalent to replacing an already-trained dense H3 attention layer after the
fact.

An axial-union oracle was built before writing an optimized kernel. The
topology had useful properties—streaming execution and query chunking could be
made exact relative to its own definition—but its output error against H3's
dense attention was already too large. The Metal production backend was never
built because the mathematical gate had failed.

The distinction matters: factorized attention can be a good architecture for a
future model. It was not a safe inference patch for this one.

### 3. Partial cache refresh gained speed but broke dialogue

This was the most informative failed cache experiment.

The “prefix-2” arm recomputed blocks 0 and 1 on every reused step, then applied
the cached residual for blocks 2–49. The normal consecutive cap was relaxed,
while the threshold and audio veto remained enabled.

Against a current cap-5 control:

| Arm | Sampling | Finished file | Reused steps |
|---|---:|---:|---:|
| Cap-5 control | 626.22 s | 707.49 s | 10/20 |
| Prefix-2 | 584.33 s | 678.27 s | 11/20 |

That was 7.2% faster in sampling and 4.3% faster to a finished file. It failed
quality decisively:

- Control: all 5 requested content keywords recognized.
- Prefix-2: 0/5 keywords recognized.
- The speech checker detected repetitive Welsh-like output rather than the
  requested sentence.
- A face and landmarks were still detected in all 124 frames, so this was not
  a simple total-render collapse.

Refreshing two blocks did not make the old 48-block tail coherent. The
audio-aware admission test could decide whether to reuse, but it could not
repair an already stale shared audio/video residual.

Prefix-3 was not worth rendering. It had a lower speed ceiling and did not
address the failed mechanism.

### 4. Fusions were locally fast and globally small

Several operations looked expensive in source code because they produced many
intermediate tensors. Measuring the full transformer showed that large matrix
multiplications and attention dominated instead.

Examples:

- Fusing modulation measured about 0.94% on the steps it affected.
- Fusing Q/K RMSNorm and RoPE made that local chain about 4.2× faster, but the
  complete render improved by only about 1.2%.
- AdaLN schedule batching reduced repeated small projections dramatically in
  isolation, yet projected to only 0.8–1.3% at render level and required
  hundreds of MiB of persistent tables.

This is Amdahl's law in practical form: optimizing 1% of the runtime infinitely
still saves at most 1%.

### 5. Compiling larger graphs did not recover meaningful time

Compiling a complete dense transformer block was tested at production shape,
including real runtime timestep embeddings. Capturing a timestep embedding as
a constant initially appeared to save about 2%, but that configuration cannot
render a diffusion schedule because the embedding changes every step.

With the timestep embedding passed honestly as a runtime input, compilation
saved roughly 0.5% for one block and at most about 1.15% in the tested larger
region. It also changed numerical results. That was below both the timing noise
and the quality-validation cost.

### 6. Weight layout helped, but not enough

Rearranging weights into the matrix layout preferred by the measured MLX GEMM
path produced a real, exact 3.0–3.1% gain on a full step. This was one of the
cleanest technical wins, but shipping it would complicate checkpoint loading,
memory ownership, and model lifecycle for a result below the project's 5%
production gate.

This may be a different decision in a system that already stores weights in the
preferred layout. Here it was not worth converting and maintaining 66 GB of
resident weights for each run.

### 7. int8 and int4 weights did not make inference meaningfully faster

Quantized files can save disk space without accelerating this workload. At a
sequence length around 15,700, each weight is reused across many rows. The
large matrix multiplications are compute-bound rather than dominated by reading
the weights once.

Measured int8/int4 matrix multiplication was slower on two important shapes and
only modestly faster on one. Weighted across the transformer, the expected gain
was about 1.2%.

This explains a common Comfy misunderstanding: a smaller checkpoint does not
necessarily mean faster generation. It may only mean less disk space, less
download time, or lower load-time memory if the runtime keeps it quantized.

### 8. Dynamic sigma schedules were not justified by the observed curve

The intuitive proposal was to cache or sparsify more aggressively at high noise
and become conservative near the end. The measured residual changes did not
follow that monotonic story. They were high near the beginning, lowest around
the middle, and rose again late.

A schedule based only on “early equals safe” would have been fitted backwards
for this model. The useful middle region was already being exploited by the
ordinary threshold and consecutive cap.

## Metrics that can mislead you

### Smoother can mean blurrier

Aggressive caching lowered frame-to-frame variation and sometimes improved a
simple lip-sync correlation. It also removed high-frequency detail. A softer
video is naturally more stable from frame to frame.

Always pair temporal smoothness with a sharpness/detail check and actual
viewing. Do not rank configurations on one smoothness score.

### Two outputs from the same seed may still depict different details

Configuration changes alter the diffusion trajectory. Once two outputs have
shifted composition, comparing their raw Laplacian variance asks which scene
happened to contain more texture, not which method preserved quality better.

In these tests, structural dissimilarity above roughly 0.01 was treated as a
warning that direct detail ratios were no longer interpretable.

### The same configuration may not be pixel-identical every time

Repeated cached controls had bit-identical audio but one video differed
slightly, despite matching seed and settings. GPU kernel selection or reduction
ordering can introduce a small numerical noise floor.

Small one-run differences should not be promoted as findings. Use repeats or
multiple seeds, especially for claimed gains below a few percent.

### Listen to generated speech, not just its waveform

Envelope and spectral correlation can say the audio signal is structurally
similar while the words or synchronization are wrong. Dialogue tests should
include:

- Transcription against the requested line.
- Repetition/hallucination detection.
- Lip synchronization and drift across the clip.
- Human listening with the video visible.

## What remains genuinely open

The rejected experiments do not prove that H3 can never run faster. They narrow
the credible work considerably.

**A better exact GEMM kernel** remains theoretically possible. The current MLX
path already reached its isolated throughput at the model's real shapes, so
there is no measured unused margin yet. Beating it would require serious
hardware-specific kernel work, not another weight transpose.

**A model trained for variable or factorized attention** could make spatial,
temporal, local, or timestep-dependent topologies safe. The failed result here
was replacing dense attention after training. It was not a verdict against
architectures designed around those patterns from the beginning.

**A stream-aware cache design** could be different from the rejected prefix
experiment. Prefix-2 still reused one old tail across the joined audio/video
state. A future cache that stores, validates, and refreshes stream-specific
state may address that mechanism, but it is a model research project rather
than a small inference wrapper.

**Memory work remains useful even when it is not a speedup.** Keeping weights
quantized in memory and decoding video in chunks could make H3 usable on smaller
machines. Those changes solve capacity, not the 95%-DiT sampling bottleneck, and
should be evaluated under a different success criterion.

## Practical recommendations for Comfy users

### If you want balanced speed and quality

1. Start with the node author's conservative cache preset.
2. For an H3 cache with semantics matching this implementation, threshold 0.10
   and a maximum of 5 consecutive reuses are the measured balance.
3. Enable a separate audio/per-stream cache check if the node provides one.
4. Keep the first and final diffusion steps full.
5. Test a close talking face before trusting the setting for audio/video work.

### If you want maximum fidelity

- Disable cross-step caching.
- Use dense attention.
- Avoid inference-time topology substitutions unless the model was trained for
  them.
- Treat a quantized checkpoint as a storage/memory choice unless the runtime
  demonstrates faster quantized compute on your exact hardware.

### If you are testing an optimization node

Hold these constant:

- Model and checkpoint hash.
- Prompt and negative prompt.
- Seed.
- Width, height, and frame count.
- Step count, sampler, scheduler, and CFG.
- Decode and output settings.
- Other cache or attention nodes.

Then test at least three kinds of content:

1. **Dialogue:** close face, visible mouth, known sentence.
2. **Fast motion:** tracking camera, moving subject, detailed background.
3. **Fine texture:** hair, fabric, leaves, text, or patterned surfaces.

Add another seed and, if you normally use it, a longer step schedule. Do not run
another GPU-heavy job during the timing sweep.

### A useful acceptance rule

For approximations that can change the diffusion trajectory, a 1–3% speed gain
is rarely worth a new quality failure mode. This project used a 5% technical
screen for bounded experiments and a stricter practical standard for anything
that required a full quality campaign. Once caching was already enabled, most
kernel ideas fell below that bar.

## Bottom line

H3 on this Apple Silicon stack was already spending nearly all of its time in
well-utilized dense matrix multiplication and attention. There was little
general overhead waiting to be removed.

The successful strategy was therefore to run the expensive transformer fewer
times, with conservative limits and an audio-aware admission test. Attempts to
make the remaining work approximate—sparse attention or long-lived partial
residual reuse—created temporal or dialogue failures. Attempts to make exact
work cheaper were mostly real but too small to matter at finished-video level.

For users, the recommendation is not “turn on every acceleration node.” Use one
well-bounded cache, validate it on the content that is easiest to break, and
keep a dense workflow available when coherence and fine detail matter more than
render time.

## Detailed evidence

The engineering measurements and machine-readable records remain available in:

- `docs/PERF_ROADMAP.md`
- `docs/ACCELERATION.md`
- `docs/SOL_ATTN.md`
- `docs/bench/`
