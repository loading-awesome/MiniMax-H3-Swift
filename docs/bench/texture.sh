#!/bin/zsh
# ARCHIVED. Kept as the record of what produced the .h3-bench.json files
# beside it. Does not run today: the cache flags it passes were removed
# once their sweeps finished and their answers became constants.
# A probe that is high-detail AND compositionally constrained.
#
# The previous two probes each failed in opposite directions and neither could
# settle the cap question:
#
#   beach     dssim 0.007 between arms — same scene, so the comparison is valid
#             — but absolute detail 0.0068, a near-featureless frame where a
#             relative change has a tiny denominator and is easy to exaggerate.
#   speaker   absolute detail 0.053, eight times the beach — but dssim 0.059
#             between arms and 0.109 against dense. Different scenes. Comparing
#             their Laplacian variance compares market stalls, not quality.
#
# The proof that the second case is broken: speaker-dense scored *lower*
# absolute detail (0.046) than speaker-cap3 (0.053). Dense is the faithful arm
# with no approximation in it; it cannot be worse. The metric was reading scene
# content.
#
# What is needed is a scene with many high spatial frequencies and few semantic
# degrees of freedom, so that a perturbed trajectory lands on the same picture
# rather than a different one. A packed bookshelf is close to ideal: the
# composition is pinned by the prompt, and the spine text is about the most
# brutal detail probe available — legible or mush, with little in between.
#
# The check before the result: if dssim between arms is not below ~0.015, this
# probe has failed the same way the speaker one did, and the answer is that
# reference-ratio metrics cannot settle this at all — it goes to reference-free
# absolute measures and blinded viewing instead.
set -u
BIN=/Volumes/scratch_disk/MiniMax-H3-Swift/.build/release/h3
OUT=/Volumes/big_daddy/scratch_disk/H3_Swift/bench6a

TEXTURE="a slow steady push in on a bookshelf packed tight with worn paperbacks, faded spine lettering, warm lamplight raking across the covers, dust in the air"

run () {
  arm=$1; shift
  echo "=== $arm — $(date +%H:%M:%S)"
  H3_BENCH_ARM="$arm" "$BIN" render \
    --prompt "$TEXTURE" --out "$OUT/${arm}.mp4" \
    --width 864 --height 480 --seconds 5 --steps 20 --seed 7 \
    "$@" > "$OUT/${arm}.log" 2>&1
  echo "    exit=$? at $(date +%H:%M:%S)"
}

run texture-cap3  --quality balanced --cache-max-skips 3
run texture-cap5  --quality balanced --cache-max-skips 5
run texture-dense --quality faithful

echo "=== done $(date +%H:%M:%S)"
