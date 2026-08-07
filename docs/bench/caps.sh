#!/bin/zsh
# ARCHIVED. Kept as the record of what produced the .h3-bench.json files
# beside it. Does not run today: the cache flags it passes were removed
# once their sweeps finished and their answers became constants.
# Phase 6C: the consecutive-cap sweep.
#
# Everything is held at the control's settings — threshold 0.10, per-stream
# probe, fusion off, seed 7, same prompt and shape — so the cap is the only
# thing that varies.
#
# cap-3-recheck runs first and is not a variant: it re-renders the shipping
# configuration with the *current* binary. The controls were rendered before
# the constraint-recording and rename work, and "those changes cannot have
# altered a decision" is a claim, not a measurement. If this does not
# reproduce control-cached, nothing after it can be trusted.
#
# Projected from the captured deltas (StepCachePolicyReplayTests):
#   cap 4  -> 9 reuses,  no speed change, refreshes move 7/11/15 -> 8/13
#   cap 5  -> 10 reuses, ~9.6% of sampling
#   cap 6  -> 10 reuses, same, refresh moves to step 10
# A rendered arm that disagrees with its projection means the trajectory
# shifted enough to change the deltas, which is itself the finding.
#
# NOTHING ELSE MAY TOUCH THE GPU WHILE THIS RUNS.
set -u
BIN=/Volumes/scratch_disk/MiniMax-H3-Swift/.build/release/h3
OUT=/Volumes/big_daddy/scratch_disk/H3_Swift/bench6a
PROMPT="a red kite over a beach at sunset, the wind picking up"

run () {
  arm=$1; cap=$2
  name="$OUT/${arm}"
  echo "=== $arm (cap $cap) — $(date +%H:%M:%S)"
  H3_BENCH_ARM="$arm" "$BIN" render \
    --prompt "$PROMPT" --out "${name}.mp4" \
    --width 864 --height 480 --seconds 5 --steps 20 --seed 7 \
    --quality balanced --cache-max-skips "$cap" > "${name}.log" 2>&1
  echo "    exit=$? at $(date +%H:%M:%S)"
}

run cap-3-recheck 3
run cap-4 4
run cap-5 5
run cap-6 6

echo "=== done $(date +%H:%M:%S)"
