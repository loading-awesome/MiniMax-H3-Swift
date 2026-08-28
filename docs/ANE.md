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
cost: the dies are idle about a third of the window they could work in, waiting
on the activation upload and the partial-sum join. Both attacks on it were
measured and lost — the whole-MLP island (fixed die budget, 10% worse) and
split-k 1 with a native Metal pack and merge (worse still, and it exhausted
memory in decode). Within a block the projections are a dependency chain and
each block feeds the next, so there is no independent work to hide the seam
behind.

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
