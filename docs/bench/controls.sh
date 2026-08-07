#!/bin/zsh
# Phase 6A controls plus the Phase 6B arm.
#
# Same prompt, same seed, same shape throughout; the configuration is the only
# thing that varies. Repeats use the same seed deliberately — the question is
# how much the machine varies, not how much the model does.
#
# Fusion is switched inside one binary rather than across two builds, so the
# comparison isolates the kernel instead of everything that changed between
# commits.
#
# NOTHING ELSE MAY TOUCH THE GPU WHILE THIS RUNS. Test runs taken beside the
# first attempt moved its step time from 25 s to 29 s, which is larger than the
# effect being measured.
set -u
BIN=/Volumes/scratch_disk/MiniMax-H3-Swift/.build/release/h3
OUT=/Volumes/big_daddy/scratch_disk/H3_Swift/bench6a
PROMPT="a red kite over a beach at sunset, the wind picking up"

run () {
  arm=$1; rep=$2; fused=$3; shift 3
  name="$OUT/${arm}-${rep}"
  echo "=== $arm rep $rep — $(date +%H:%M:%S)"
  H3_BENCH_ARM="$arm" H3_FUSED_MODULATION="$fused" "$BIN" render \
    --prompt "$PROMPT" --out "${name}.mp4" \
    --width 864 --height 480 --seconds 5 --steps 20 --seed 7 \
    "$@" > "${name}.log" 2>&1
  echo "    exit=$? at $(date +%H:%M:%S)"
}

# Cached first: it is the shipping default, so it is the arm every proposal has
# to beat. Fused next, so the 6B answer lands before the slower dense runs.
for r in 1 2 3; do run control-cached $r 0 --quality balanced; done
for r in 1 2 3; do run fused-cached   $r 1 --quality balanced; done
for r in 1 2;   do run control-dense  $r 0 --quality faithful; done

echo "=== done $(date +%H:%M:%S)"
