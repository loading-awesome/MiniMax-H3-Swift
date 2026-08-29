# The Apple Neural Engine route

Both ANE dies run alongside Metal during sampling. On an M3 Ultra at
864x480x124, 20 steps, this is **1.22x end to end** and 1.28x on the steps that
run the full stack.

Opt in with `H3_ANE=experimental` for the linear projections and
`H3_ANE_ATTENTION=1` plus `--attention ane` for attention. `H3_ANE_FC2=1` adds
`fc2`, which is separate because its per-block scales are calibrated against one
prompt at one shape.

## What runs where

Per block, in order: `qkv`, attention, `attn out`, `fc1`, `fc2`. Each linear
projection is cut by output column between the GPU and the two dies; attention
is split by complete head, 48 on Metal and four on each die. No score plane
crosses a device boundary — only Q/K/V and finished head outputs do.

    arm                             per step   full step
    GPU only                         28.34 s     55.28 s
    + attention on both dies         27.20       51.97     1.042x
    + qkv, attn out, fc1             24.48       46.28     1.158x
    + fc2                            24.10       44.99     1.176x
    + split-k 4                      23.17       43.26     1.223x

## Attention is a win at one shape and a loss at another

The 1.042x above is 864x480x124 at 20 steps. At 448x832x124 on the four-step
distilled checkpoint it reverses. Sampling seconds, and the projections each
run's calibration accepted:

    Metal attention   166.9  all four routed
                      173.7  all four routed
                      174.5  three routed
                      188.1  two routed
    ANE attention     184.6  all four routed
                      185.5  all four routed
                      186.6  all four routed

Compared like for like -- runs where all four projections routed -- ANE
attention costs about 15 s, 170.3 against 185.6. Leave `H3_ANE_ATTENTION`
unset at this shape.

**One run of each does not establish this**, and a first pass here claimed
18.6 s from a single pair before the repeats existed. The Metal runs span 21 s
and overlap the ANE runs; only the ANE side is tight, at 2 s across three runs.

The backend is not doing anything wrong. It calibrates once and refuses:

    calibrate S=13938 D=128 H=56: engine(8) 673 ms, gpu(48) 282 ms,
    routed 673 vs unrouted 330 -> DECLINE

Every later call hits the refusal cache, and the two renders are bit-identical
-- PSNR inf, not merely close -- so the attention arithmetic is provably
untouched. What a backend that computes nothing is costing is still unknown.

## The routing calibration is a coin flip worth about 18 s

`routingBeatsMetal` races the engine against Metal once per shape and caches
the verdict, and at this shape it does not decide the same way twice. Across
four otherwise identical Metal-attention renders it accepted four projections,
then four, then three, then two -- and the sampling phase tracked it: 166.9 and
173.7 s with four, 188.1 s with two.

So a render's speed depends on how a millisecond-scale race fell out during its
first block. Two things make that race unrepresentative of production. It runs
`best(2)` of each side, so by the routed timing the activation is already
materialised, and the wait that dominates in production -- `start` calls
`eval` on an activation whose input is the attention output -- has already been
paid. And it races one projection alone, where the dies are idle, rather than
against the other projections competing for them.

The phase report shows the same effect from the other side: with ANE attention
`attn out` carries a 360 ms materialise, and with Metal attention that cost
moves to `fc1` at 417 ms. It is not a property of either projection. Whichever
routed projection runs first after attention absorbs the wait for it.

## The constants, and why they are what they are

Swept jointly, because they interact and sweeping them separately once produced
a wrong default. Full step in seconds:

    split \ share   0.30    0.375   0.45    0.52    0.60
      8            46.23   45.25   46.99      -       -
      4               -    43.36   43.07   44.25      -
      2               -       -    46.98   50.83   53.55

Split 4 at share 0.45 is the optimum, and both axes have interior minima.
Fewer pieces means less GPU-side fp32 partial summing per column, so the engine
can afford more columns — but it also lowers the engine's own rate. Between 8
and 4 the first effect wins; by 2 the engine has lost too much rate to recover.

`fc2` runs at a per-block operand scale (`ANEFC2Scales`). Its worst partial-sum
bound is 4,573,078 at block 45, nine times past the engine's 2^15 cliff at the
shipping 1/16 scale, and the failure mode is silent zeros. The bounds vary 790x
across blocks, so one global scale would either miss the peak or push the quiet
blocks into fp16 denormals. Regenerate the table with a bound render —
`H3_ANE_BOUND=out.json h3 render`, engine off — and audit it with
`H3_ANE_FC2_VERIFY=1`, which recomputes every routed `fc2` on the GPU and
reports both relative error and the fraction of outputs that came back zero
where the reference did not. That second number is the saturation signature.

## Shape cliffs, and the guard

Three separate shape cliffs have been measured, all with the same signature —
flat, then catastrophic, with no rule that predicts the edge:

- **Heads per attention graph.** Four runs at 1.90 TFLOP/s, five at 0.20.
- **Sequence length.** S=15,744 runs at 1.71 TFLOP/s and S=15,731 at 0.41.
  Thirteen elements. Not 64-alignment: S=11,008 and 11,072 are exact multiples
  of 64 and run no faster than unaligned 11,000.
- **Contraction pieces.** 4 and 8 are fine; 16 and 32 are slower than not
  routing at all, and the engine's own busy time triples, so it is the engine
  slowing down rather than reduction cost rising.

None of these produce an error. They compute the right answer slowly, which
would ship as a silent regression. So `ANEAttentionBackend` calibrates at
session creation: it times the dies against the GPU and takes the route only if
`max(engine, retained) < unrouted * 0.97`. That needs no constant for either
device's rate and catches cliffs that have not been found yet.

**Every number here is one prompt at one shape.** Given the sequence cliff,
none of it should be assumed to transfer to another resolution or duration.

## Measuring a new shape

`H3_ANE_PHASES=1` reports where each routed projection's wall time went —
upload split into its GPU materialise and its CPU memcpy, the wait on the
engine, and the join. `H3_ANE_UTILISATION=1` adds the engine's own busy time.
Run both on any new shape before trusting the figures here.

Measured on the arm above, per call:

    qkv       upload 26.5 ms gpu +  3.7 memcpy   wait  96.7 ms   join 15.2 ms
    attn out  upload 21.6        +  3.8          wait  41.5      join  3.8
    fc1       upload 12.0        +  3.0          wait 126.3      join 21.3
    fc2       upload 11.6        +  8.6          wait  91.0      join  3.3

`wait` totals 180.0 s against the engine's own 187.8 s of busy time, so the
engine — not the GPU shard — is the critical path while the two run
concurrently. That is the mechanical reason the share sits at 0.45 rather than
higher: past it, every extra column is given to the device that already
finishes last.

## The ceiling

A block is 1105.6 ms on the GPU alone: attention 401.8, linear 703.8. The GPU
keeps 48 of 56 heads, so routed attention floors at 344.4 ms.

The engine's rate is the term that decides the rest, and it is a wall rather
than a setting: 7.7 to 7.95 TFLOP/s across both dies, swept by output column, by
sequence tile, and the rectangles between, never exceeded. Expressing a
projection as a 1x1 convolution — the engine's native form — is bit-identical
and 2.6x slower.

At that rate, perfect sharing of every routable projection gives **1.303x**.
1.5x would need a 36.4 s full step against a best-possible 42.1, so it is not
reachable by moving more work to the dies. Reaching it needs a faster engine
path, a cheaper GPU baseline, or genuinely parallel work rather than merely
relocatable work — and at `cfgScale 1` there is no second CFG branch to
pipeline.

The route is at 1.223x, which is 94% of that ceiling. The remainder is seam
cost: the activation upload and the partial-sum join are serial regions in which
the dies have nothing to do.

Four attacks on that seam have been measured and all four lost. The whole-MLP
island fixes the die budget where routing each projection separately lets the
share float, and is 10% worse. Split-k 1 with a native Metal pack and merge is
worse still, and exhausted memory in decode. Submitting each contraction piece
as it lands would hide only the memcpy, which is 9.6 s against the GPU
materialise's 36.5 — worth under 2%. And feeding the activation untransposed,
which would delete the materialise outright, was measured at 2.42 TFLOP/s a die
against this form's 3.79.

What remains is structural. Within a block the projections are a dependency
chain and each block feeds the next, so there is no independent work to hide the
seam behind, and the engine is already the critical path while it runs.

## Private ABI

`AppleNeuralEngine.framework`, via `_ANEInMemoryModelDescriptor`,
`_ANEInMemoryModel`, `_ANERequest` and `_ANEIOSurfaceObject`. Models are
compiled from MIL text in memory. Two notes that cost real time:

- **The compiler renames function arguments to alphabetically ordered input
  symbols.** For `q, k, v` the compiled order is `(k, q, v)` at indices
  `(0, 1, 2)`. Bind by the symbol order in the load reply
  (`kANEFModelInputSymbolsArrayKey`), never by the order the MIL declares.
  Binding textual order produces a working graph if a second error cancels it,
  which is how the attention graph shipped for a while.
- **`kANEFAneInstanceHint` does not select a die.** Two dies run only when two
  evaluations are in flight at once, which is why work goes through
  `h3_ane_attention_run_pair` and `h3_ane_run_pair`.

The MIL `softmax` op normalises per tile at long sequence lengths and is wrong
at production scale — 0.974 relative RMS at S=15,744. Writing the reduction out
as max, subtract, exp, sum, divide keeps it global and gives 8.99e-4, which is
better than the bf16 GPU path's own 1.66e-3.

## Other Macs

The route gates on macOS 27 or newer and on nothing else. It does not check
`hw.model`: an allowlist of one machine made the feature silently unavailable
on every other Mac, and what actually has to hold is that the driver is fixed
and the private classes resolve.

Everything tuned to a particular chip is measured at startup instead of
assumed:

- **Die count.** `kANEFAneInstanceHint` does not select a die, so two dies run
  only when two evaluations are in flight. The probe submits one evaluation,
  then two concurrently: about the same cost means two dies, about twice means
  one. On the calibrated machine it reports `one evaluation 0.22 ms, two
  concurrent 0.26`.
- **The engine's share of the columns.** 0.45 was measured against two dies at
  7.9 TFLOP/s combined. With one die the balance `r / (1 + r)` for a halved
  engine gives **0.263**, and the constant is rescaled accordingly.
- **Whether routing is worth it at all**, per projection shape. The first call
  at each `(s, k, n)` runs both ways and takes the faster. This matters in both
  directions: it is what protects an unfamiliar machine from a route that loses,
  and on the calibrated machine it keeps the small `s=8` conditioning
  projections on the GPU, where the engine had been losing 4.5 ms against
  Metal's 1.9.
- **Attention**, by the same contract, in `ANEAttentionBackend.worthTaking`.

**This makes other Macs safe and self-tuning, not fast.** A single-die part has
roughly half the engine, so the 1.303x ceiling here is nearer 1.10 to 1.15
there, and on some shapes the calibration will correctly decline. `headsPerDie`
is still a constant measured on this ANE; the attention guard will refuse a
shape where it is wrong rather than find the right value.

Under macOS 26 or earlier the bridge logs why it is off and everything falls
back to Metal.

## Machine safety

macOS 26A5421a or later. On earlier builds a request arriving during the
driver's power transition was admitted against a DART whose mappings were being
torn down, which panicked the machine minutes later. 27.0 fails closed instead,
cancelling the request.

That cancellation is transient and safe to retry: the request is refused before
admission, so nothing ran and no surface was touched. The bridge retries a
bounded handful of times 3 ms apart, which is inside the 8-15 ms transition it
collides with. At a 2 s submission cadence that takes 108 failures in 120 pairs
to zero. `Tools/ANE/pair-stress.m` is the regression.

Nothing here forces admission — every refusal is honoured — so the retry does
not reopen the panic path.
