#!/bin/zsh
# Re-testing the cache conclusions on content that actually has fine detail.
#
# Every number so far came from "a red kite over a beach at sunset" — smooth
# sky, sand and water, almost no high-frequency content. That made it a
# convenient control and a bad probe for a detail metric, and it conditions the
# speed numbers too: reuse count depends on how far the residual moves per
# step, so a calm scene qualifies more steps than a busy one. 45% reuse and
# 1.79x are an easy-content result reported as a general one.
#
# Two probes, chosen for what the beach lacked:
#
#   foliage  Maximum high-frequency content — ferns, bark, dappled light — with
#            motion through it. This is where a detail metric has something to
#            measure and where softening should be most visible.
#   speaker  A face, lip-sync, spoken dialogue, and patterned fabric. Covers the
#            audio and face gates the foliage probe cannot, on a scene that
#            still carries real texture.
#
# Arms: dense (detail ceiling) and cap 3 vs cap 5 on foliage; cap 3 vs cap 5 on
# speaker. Cap 5 because it is the only arm that cleared the speed gate; cap 4
# bought nothing and cap 6 tracked cap 5. Seed and shape match the controls.
set -u
BIN=/Volumes/scratch_disk/MiniMax-H3-Swift/.build/release/h3
OUT=/Volumes/big_daddy/scratch_disk/H3_Swift/bench6a

FOLIAGE="a hiker pushing through dense ferns in an old forest, sunlight breaking through the canopy, bark and moss in close focus, leaves shaking as she passes"
SPEAKER="a woman in a knitted patterned coat talking straight to camera on a busy market street, stalls of fruit and printed fabric behind her, she says: the question is whether the fine detail survives"

run () {
  arm=$1; prompt=$2; shift 2
  echo "=== $arm — $(date +%H:%M:%S)"
  H3_BENCH_ARM="$arm" "$BIN" render \
    --prompt "$prompt" --out "$OUT/${arm}.mp4" \
    --width 864 --height 480 --seconds 5 --steps 20 --seed 7 \
    "$@" > "$OUT/${arm}.log" 2>&1
  echo "    exit=$? at $(date +%H:%M:%S)"
}

run foliage-cap3 "$FOLIAGE" --quality balanced --cache-max-skips 3
run foliage-cap5 "$FOLIAGE" --quality balanced --cache-max-skips 5
run foliage-dense "$FOLIAGE" --quality faithful
run speaker-cap3 "$SPEAKER" --quality balanced --cache-max-skips 3
run speaker-cap5 "$SPEAKER" --quality balanced --cache-max-skips 5

echo "=== done $(date +%H:%M:%S)"
