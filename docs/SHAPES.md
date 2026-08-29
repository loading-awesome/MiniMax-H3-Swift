# Vetting an output shape

`h3 recipes` marks a rung `ok` when the policy permits it. That is not the same
as measured, and until now only 864x480 was ever measured. This is what it takes
to promote a rung from permitted to production, and what the answer cannot cover.

Run it with `Tools/ShapeVet/vet.sh <outdir> <base.safetensors> <distill.safetensors>`.

## What a shape has to clear

| | instrument | why |
|---|---|---|
| renders at all | exit code, `h3` policy | frame lattice, memory ceiling |
| the ANE route engages | `H3_ANE_ATTENTION_TRACE=1` | a declined shape is correct and slower, and says nothing |
| cost | `per step`, **and step 1 separately** | see the compile cliff below |
| no strobing | `Tools/coherence_check.py`, `pulse` | the failure the 20-step floor exists for |
| detail retained | same tool, Laplacian variance | the blur trap |
| lip-sync, **both arms** | `h3-lipsync`, `h3-facecheck` | speech is where a small shape and a distill both fail |
| looks right | the frame strips | nothing here replaces this |

## Four things this exercise established the hard way

**Compare only within a shape, and only within a model.** `coherence_check`
refuses ratios once dssim passes ~0.01, because two different trajectories are
two different scenes. A base render and a distilled render are *always* a
different scene — measured dssim 0.32 on the same prompt and seed — so no ratio
in that table can answer "is the distill as good". Only the absolute columns
and your eyes can.

**The absolute columns are content-dependent too.** `pulse` reads 0.118 on a
calm beach, 0.233 on a speaking close-up and 0.586 on a gale-blown wood. A high
value is strobing *or* legitimately busy content, and the metric cannot tell
them apart. It catches gross failure, not quality.

**Run every quality check on both arms.** Lip-sync failed on the distill at
352x608 and it was tempting to blame the distill. The base fails there too:
margin -0.013 with the mouth measured in 124 of 124 frames. The rung is
unsuitable for speech whatever renders it. Checking one arm would have recorded
the wrong cause.

**Step 1 is its own measurement.** ANE programs compile per shape, and the cost
is a shape lottery like every other property of this hardware:

    S = 8,155   step 1  53.2 s
    S = 8,167   step 1 249.8 s
    S = 8,170   step 1  50.7 s

The bad value sits between two good ones, and 8,170 is *longer* than 8,167. `S`
includes the text tokens, so **editing the prompt changes the ANE job**: the
same recipe with a different prompt is not the same render. A mean over four
steps hides this; over twenty it hides it completely.

The defence is to cache compiled programs by shape, so a bad shape is paid once
ever rather than once per render. Padding `S` to a known-good value would also
work for the linear programs, but attention would need a mask and no rule
predicts which values are good.

## The decode has a cliff of its own, and it is not the DiT's

The video VAE decodes in fixed 256-pixel tiles with a 64-pixel minimum overlap,
so **the tile count steps rather than scales** and a frame just past a step pays
for tiles it does not use:

    480x864   3 x 5 = 15 tiles   63.7 s
    448x832   2 x 4 =  8 tiles   33.9 s
    448x768   2 x 4 =  8 tiles   33.9 s
    512x896   3 x 5 = 15 tiles   63.5 s

480x864 sits just past the cliff on *both* axes -- 480 needs three tiles where
448 needs two, 864 needs five where 832 needs four -- and is therefore strictly
dominated in both directions: drop to 448x832 for 1.88x the decode speed at 10%
fewer pixels, or rise to 512x896 for 11% more pixels at the same cost. In a real
render this was 75.6 s to 40.1 s of decode, 314.0 s to 260.5 s end to end, with
no code and no numerics changed.

Enlarging the tiles is not available. `createTokenIds` normalises coordinates by
each tile's own dimensions, so every tile decodes as a complete image spanning
[-1, 1] and a different tile size changes every token's rotary phase.

**Check this before picking a rung.** It is independent of the DiT's token
count, so a shape can be cheap to sample and expensive to decode.

## The ladder is chosen for tiles, not only for pixels

Each megapixel rung is the fewest-tile shape within 10% of its target, inside an
aspect band of 1.70 to 1.87, and strictly larger than the rung below it. Both
axes stay multiples of 32, so the VAE's /16 and the DiT's /2 patch both divide
without remainder -- that constraint is prior to everything else here.

    mp    was          tiles      now          tiles   decode
    0.4   864x480      15         832x448       8      1.88x
    0.6   1056x608     18         1024x576     15      1.20x
    0.9   1280x736     28         1216x704     24      1.17x
    1.2   1504x832     32         1408x800     28      1.14x
    1.5   1664x928     45         1600x928     40      1.12x
    1.8   1824x1024    50         1792x1024    45      1.11x
    2.0   1920x1088    60         1888x1024    50      1.20x

Four rungs were already tile-minimal and moved only to keep the ladder strictly
increasing. Two properties are worth stating because a naive re-derivation
breaks them: rungs must not collide (a first pass put 0.7 and 0.8 on the same
shape, and 1.8 and 2.0 on another), and tile counts must not go backwards as
megapixels rise.

**Decode is about 15% of a render, so these are 1.5-4% end to end.** The larger
effect is that tile-minimal shapes carry 4-10% fewer pixels, which cuts sampling
quadratically -- but that is partly just a smaller frame, not a free win.

## Duration is the expensive axis

`H3Video.trainedFrameRange` is `124...362` -- at 24 fps, 5.2 s to 15.1 s, with
frame counts snapping up onto a 17k+5 lattice. A 15 s request becomes 362
frames, the exact top of the range. There is no headroom above it: 20 s is
outside what the model was trained to sample, not merely longer.

15 s renders and delivers -- 361 frames at 448x832, valid video and audio,
17.6 min end to end, ~113 GB resident. But sampling scales super-linearly,
because attention is quadratic in sequence length:

    frames  124 -> 361   2.91x
    s/step  40.1 -> 235.1   5.86x
    decode  34 s -> 96 s    2.82x   (linear, as expected)
    text    16.1 -> 16.2 s  1.00x   (fixed cost, weights paging in)

So duration reads like a linear request and is not one. Decode scales linearly;
only sampling pays the quadratic.

## Results

*(filled in as rungs are vetted; see `Tools/ShapeVet`)*

| rung | tokens | ANE | distill gain | calm | motion | speech | texture |
|---|---:|---|---:|---|---|---|---|
| 352x608 | 8,147 | routes | 1.34-1.78x | ok | ok | **fails, both arms** | ok |
| 448x832 | 13,882 | routes | not run | — | — | — | — |

**No current rung is verified, and that is a regression the ladder chose.**
864x480 held the verification — parity gate and lip-sync on both arms — and the
ladder no longer contains it: rungs are now picked for the decoder's tile grid,
which moved 0.4 MP to 832x448 for 1.88x the decode speed. Verification does not
travel with a name, so `h3 recipes` calls that rung `reference` rather than
`verified`, and `H3RenderPolicy.verified*` still records 864x480 as what was
actually measured. It stays reachable with explicit `--width 864 --height 480`.

448x832 is listed because it is where the ANE and decode measurements in
`docs/ANE.md` were taken, and because four-step speech renders were judged good
by eye and ear there — which is not what the columns above mean. Running
`vet.sh` on it is the outstanding work.
