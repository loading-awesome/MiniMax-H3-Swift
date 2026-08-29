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

## Results

*(filled in as rungs are vetted; see `Tools/ShapeVet`)*

| rung | tokens | ANE | distill gain | calm | motion | speech | texture |
|---|---:|---|---:|---|---|---|---|
| 352x608 | 8,147 | routes | 1.34-1.78x | ok | ok | **fails, both arms** | ok |
