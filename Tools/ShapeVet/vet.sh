#!/bin/zsh
# Vet an output shape for production: does it render, does the ANE route take
# it, what does it cost, and does the four-step distill hold up there.
#
# A shape is "ok" in `h3 recipes` when the policy permits it. That is not the
# same as measured, and only 864x480 has ever been measured. This promotes a
# rung from permitted to measured.
#
# For every (shape, prompt) it renders two arms -- base at 20 steps and the
# distill at 4 -- and compares them *at that shape*. Comparing across shapes is
# not meaningful: `coherence_check` warns that once dssim passes about 0.01 the
# detail ratio beside it means nothing, and two different resolutions are two
# different scenes.
#
#   Tools/ShapeVet/vet.sh <outdir> <base.safetensors> <distill.safetensors>
set -u
OUT=${1:?outdir}; BASE_CKPT=${2:?base checkpoint filename}; DIST_CKPT=${3:?distill checkpoint filename}
REPO=/Volumes/scratch_disk/MiniMax-H3-Swift
CFG=$HOME/.config/minimax-h3/config.json
cd $REPO; mkdir -p $OUT/frames

# Prompts chosen to probe different failure modes. The beach sunset is the
# project's control and, in its own words, "a convenient control and a poor
# probe" -- calm, low frequency, forgiving. It cannot vet a shape alone.
typeset -A PROMPTS
PROMPTS[calm]="a red kite over a beach at sunset"
PROMPTS[motion]="a motorcycle racing along a wet city street at night, camera tracking alongside, rain spray and neon reflections"
PROMPTS[speech]="a close-up of a woman with freckles talking directly to camera in a sunlit kitchen, explaining a recipe"
PROMPTS[texture]="dense autumn woodland in a gale, thousands of leaves scattering, dappled light through branches"

point_at() {  # swap which checkpoint fl2va.bf16 names
  python3 - "$1" <<'PY'
import json,sys,os
p=os.path.expanduser('~/.config/minimax-h3/config.json')
d=json.load(open(p)); d['checkpoints']['fl2va']['bf16']=sys.argv[1]
json.dump(d,open(p,'w'),indent=1,sort_keys=True)
PY
}
restore() { point_at "$BASE_CKPT"; }
trap restore EXIT INT TERM

render() {  # $1=ckpt $2=tag $3=w $4=h $5=prompt  -> echoes seconds
  point_at "$1"
  H3_ANE=experimental H3_ANE_FC2=1 H3_ANE_ATTENTION=1 H3_ANE_ATTENTION_TRACE=1 \
    $REPO/.build/release/h3 render --prompt "$5" --width $3 --height $4 \
    --seconds 5 --steps 20 --seed 7 --overwrite --attention ane \
    --out $OUT/$2.mp4 > $OUT/$2.log 2>&1
  echo $?
}

echo "shape,prompt,tokens,arm,exit,total_s,step_s,ane_route"
for spec in "352 608" "416 736" "480 864"; do
  W=${spec%% *}; H=${spec##* }
  for key in calm motion speech texture; do
    P=$PROMPTS[$key]
    for arm in base dist; do
      [[ $arm == base ]] && CK=$BASE_CKPT || CK=$DIST_CKPT
      TAG="${W}x${H}-${key}-${arm}"
      rc=$(render "$CK" "$TAG" $W $H "$P")
      tot=$(tr '\r' '\n' < $OUT/$TAG.log | grep -o "done in [0-9hms ]*" | tail -1)
      st=$(tr '\r' '\n' < $OUT/$TAG.log | grep "per step" | awk '{print $3}')
      route=$(grep -c "calibrate.*-> route" $OUT/$TAG.log)
      tok=$(python3 -c "
import json,glob
try: print(json.load(open('$OUT/$TAG.h3-bench.json'))['identity'].get('packedTokens','?'))
except Exception: print('?')")
      echo "${W}x${H},$key,$tok,$arm,$rc,${tot#done in },$st,$route"
      ffmpeg -hide_banner -loglevel error -i $OUT/$TAG.mp4 \
        -vf "select='not(mod(n\,24))',scale=200:-1,tile=5x1" -frames:v 1 \
        $OUT/frames/$TAG.png -y 2>/dev/null
    done
    # Same shape, same seed, same prompt: the only valid comparison.
    python3 $REPO/Tools/coherence_check.py \
      $OUT/${W}x${H}-${key}-base.mp4 $OUT/${W}x${H}-${key}-dist.mp4 \
      > $OUT/coherence-${W}x${H}-${key}.txt 2>&1
    if [[ $key == speech ]]; then
      $REPO/.build/release/h3-lipsync $OUT/${W}x${H}-speech-dist.mp4 \
        > $OUT/lipsync-${W}x${H}.txt 2>&1
    fi
  done
done
echo "VET DONE"
