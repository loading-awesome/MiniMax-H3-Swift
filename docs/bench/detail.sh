#!/bin/zsh
# ARCHIVED. Kept as the record of what produced the .h3-bench.json files
# beside it. Does not run today: the cache flags it passes were removed
# once their sweeps finished and their answers became constants.
# Re-testing the cache conclusions on content that actually has fine detail.
#
# Every number before this came from "a red kite over a beach at sunset" —
# smooth sky, sand and water. Convenient control, bad detail probe. It
# conditions the speed figures too: reuse count depends on how far the residual
# moves per step, so a calm scene qualifies more steps than a busy one, and 45%
# reuse / 1.79x are an easy-content result.
#
# ## Order
#
# **The speaker probe runs first, and its cap-3 arm runs before its cap-5 arm.**
#
# H3 generates picture and soundtrack in one pass, so a silent forest is not a
# representative test of it whatever its Laplacian variance says. The speaker
# probe carries a face, lip-sync, dialogue *and* real texture — patterned coat,
# market stalls — so it exercises every gate at once. Foliage is the specialist
# detail probe and goes second.
#
# cap-3 before cap-5 because cap-3 is the shipping configuration: its delta
# trace is the one that legitimately projects the others. The replay model
# reproduced every rendered arm of the previous sweep step for step, so **one
# cap-3 render on new content projects the reuse count and speed of every cap
# on that content** without rendering any of them. Only quality needs the
# second render. Check after the first, not after the fifth.
#
# NOTHING ELSE MAY TOUCH THE GPU WHILE THIS RUNS.
set -u
BIN=/Volumes/scratch_disk/MiniMax-H3-Swift/.build/release/h3
OUT=/Volumes/big_daddy/scratch_disk/H3_Swift/bench6a

SPEAKER="a woman in a knitted patterned coat talking straight to camera on a busy market street, stalls of fruit and printed fabric behind her, she says: the question is whether the fine detail survives"
FOLIAGE="a hiker pushing through dense ferns in an old forest, sunlight breaking through the canopy, bark and moss in close focus, leaves shaking as she passes"

run () {
  arm=$1; prompt=$2; shift 2
  echo "=== $arm — $(date +%H:%M:%S)"
  H3_BENCH_ARM="$arm" "$BIN" render \
    --prompt "$prompt" --out "$OUT/${arm}.mp4" \
    --width 864 --height 480 --seconds 5 --steps 20 --seed 7 \
    "$@" > "$OUT/${arm}.log" 2>&1
  echo "    exit=$? at $(date +%H:%M:%S)"
}

# Representative probe first, and self-contained: dense gives it a detail
# ceiling of its own so the cache's cost can be stated on this content rather
# than borrowed from the beach.
run speaker-cap3  "$SPEAKER" --quality balanced --cache-max-skips 3
run speaker-cap5  "$SPEAKER" --quality balanced --cache-max-skips 5
run speaker-dense "$SPEAKER" --quality faithful

# Detail specialist second.
run foliage-cap3  "$FOLIAGE" --quality balanced --cache-max-skips 3
run foliage-cap5  "$FOLIAGE" --quality balanced --cache-max-skips 5

echo "=== done $(date +%H:%M:%S)"
