#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diff our sampling trajectory against the reference pipeline's, step by step.

The reference dumps, per DMD step, both latents going in and both raw
velocities coming out, plus each stream's sigma. Ours dumps the same. With the
initial noise and prompt embeds forced to theirs, any divergence is ours.

Read the columns in this order:

  velocity    Whether the *model* agrees. Our port and theirs run the same
              weights, so a mismatch here is a forward-pass bug -- attention,
              norms, modulation -- not a sampler bug. This must be ~1.0 before
              any other column means anything.

  latent_in   Whether the *sampler* agrees. Step 0 is the initial noise, so it
              is 1.0 by construction; step 1 is the first thing our step rule
              produced. The first step where this drops is the step whose
              update rule is wrong, and the stream that drops tells you which
              of the two it is.

A stream can diverge alone: audio has its own sigma ladder and its own
velocity scaling, which is exactly how four wrong audio samplers passed video
unharmed.

    python3 compare_trajectory.py --theirs <dir> --ours <dir>
"""
import argparse, glob, os, sys
import numpy as np


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    n = min(a.size, b.size)
    if a.size != b.size:
        print(f"    ! shape mismatch {a.size} vs {b.size}, comparing first {n}")
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else float("nan")


def rel_rms(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    denom = np.sqrt((a * a).mean()) or 1e-30
    return float(np.sqrt(((a - b) ** 2).mean()) / denom)


def load(d, name):
    p = os.path.join(d, name)
    return np.load(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--ours", required=True)
    args = ap.parse_args()

    steps = sorted({int(os.path.basename(f).split(".")[0][4:])
                    for f in glob.glob(os.path.join(args.theirs, "step*.npy"))})
    if not steps:
        print(f"no step tensors in {args.theirs}", file=sys.stderr)
        return 2

    print(f"{'step':>4}  {'stream':<6} {'latent_in cos':>14} {'velocity cos':>13} "
          f"{'vel rel_rms':>12}")
    first_bad = None
    for i in steps:
        for stream in ("video", "audio"):
            tl = load(args.theirs, f"step{i}.{stream}_latent_in.npy")
            ol = load(args.ours, f"step{i}.{stream}_latent_in.npy")
            tv = load(args.theirs, f"step{i}.{stream}_velocity.npy")
            ov = load(args.ours, f"step{i}.{stream}_velocity.npy")
            if tl is None or ol is None:
                continue
            lc = cos(tl, ol)
            vc = cos(tv, ov) if (tv is not None and ov is not None) else float("nan")
            vr = rel_rms(tv, ov) if (tv is not None and ov is not None) else float("nan")
            flag = ""
            if lc < 0.999 and first_bad is None:
                first_bad = (i, stream); flag = "  <-- first divergence"
            print(f"{i:>4}  {stream:<6} {lc:>14.6f} {vc:>13.6f} {vr:>12.4e}{flag}")

    for name in ("final.video_latent", "final.audio_latent"):
        t, o = load(args.theirs, name + ".npy"), load(args.ours, name + ".npy")
        if t is not None and o is not None:
            print(f"{name:<24} cos {cos(t,o):.6f}   rel_rms {rel_rms(t,o):.4e}")

    print()
    if first_bad is None:
        print("No divergence above 1e-3. If the audio still differs by ear, the "
              "difference is below this resolution or downstream of the loop.")
    else:
        i, stream = first_bad
        print(f"First divergence: step {i}, {stream} stream.")
        print("If that step's velocity cos is ~1.0, the model agrees and the "
              "update rule that produced this latent is wrong.")
        print("If the velocity also differs, the forward pass diverges and the "
              "sampler is not the problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
