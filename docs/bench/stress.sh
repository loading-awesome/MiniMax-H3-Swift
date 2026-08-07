#!/bin/zsh
# ARCHIVED. Kept as the record of what produced the .h3-bench.json files
# beside it. Does not run today: the cache flags it passes were removed
# once their sweeps finished and their answers became constants.
# Stress test for cap 5, which looked good on the speaker probe by eye.
#
# Four axes, chosen for where a cross-step *residual* cache should break rather
# than for variety. The cache re-applies one step's total residual to a later
# step, so it fails when the residual stops being a good prediction of the next
# one — which is a statement about how fast the trajectory is moving, not about
# how detailed the frame is.
#
#   moto     Fast motion, tracking camera, background whipping past. Every
#            pixel changes every frame, so a reused residual has the least
#            chance of still being right. The most direct test there is.
#
#   talk     Close-up dialogue. **The documented failure mode for this model**:
#            every published cache for H3 degrades audio, and the tool author's
#            summary is "all cache methods currently warp audio". The speaker
#            probe had a voice but at market distance; this puts the mouth in
#            frame where lip-sync desync is visible rather than inferred.
#
#   moto40   The same fast motion at 40 steps instead of 20. **The structural
#            risk nobody has tested.** A finer schedule means smaller deltas per
#            step, so more steps fall under the threshold and the cap becomes
#            the binding constraint far more often. At cap 3 that is bounded; at
#            cap 5 the render can coast in longer chains, and the shipping
#            default is 20 only because that is what people usually ask for.
#
#   seed 42  The same fast motion on a different seed. Different scene, same
#            configuration — the only way to tell a cap effect from a draw.
#
# Ordered highest-risk-first so an early check is informative: if fast motion
# at seed 7 breaks, nothing after it needs running.
set -u
BIN=/Volumes/scratch_disk/MiniMax-H3-Swift/.build/release/h3
OUT=/Volumes/big_daddy/scratch_disk/H3_Swift/bench6a

MOTO="a motorcycle racer leaning hard through a wet corner, spray flying off the tyres, camera tracking fast alongside, trees and barriers whipping past behind"
TALK="close-up of a man with a short grey beard speaking directly to camera in a quiet room, shallow depth of field, he says: if the cache is going to break anything at all it will break this sentence"

run () {
  arm=$1; prompt=$2; seed=$3; steps=$4; cap=$5
  echo "=== $arm — $(date +%H:%M:%S)"
  H3_BENCH_ARM="$arm" "$BIN" render \
    --prompt "$prompt" --out "$OUT/${arm}.mp4" \
    --width 864 --height 480 --seconds 5 --steps "$steps" --seed "$seed" \
    --quality balanced --cache-max-skips "$cap" > "$OUT/${arm}.log" 2>&1
  echo "    exit=$? at $(date +%H:%M:%S)"
}

run moto-cap3-s7    "$MOTO" 7  20 3
run moto-cap5-s7    "$MOTO" 7  20 5
run talk-cap3-s7    "$TALK" 7  20 3
run talk-cap5-s7    "$TALK" 7  20 5
run moto40-cap3-s7  "$MOTO" 7  40 3
run moto40-cap5-s7  "$MOTO" 7  40 5
run moto-cap3-s42   "$MOTO" 42 20 3
run moto-cap5-s42   "$MOTO" 42 20 5

echo "=== done $(date +%H:%M:%S)"
