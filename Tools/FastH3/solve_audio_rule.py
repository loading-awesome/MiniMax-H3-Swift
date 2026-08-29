#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Recover the reference sampler's update rule from its own captured tensors.

Four readings of the DMD audio step were argued from the reference source and
all four sounded wrong. This does not read source. Each captured rung gives

    x_i  (latent in)   v_i  (raw velocity out)   x_{i+1}  (next latent in)

and every candidate rule is affine in exactly those two tensors, so fit

    x_{i+1} ~= a * x_i + b * v_i

by least squares over ~10^7 elements. Two unknowns, ten million equations: if
the true rule is affine in (x, v) the fit is exact and (a, b) *is* the rule.
A poor fit is itself the answer -- it means the step is stochastic, and the
residual's size says how much fresh noise went in.

Then name it: each candidate predicts its own (a, b) from the sigmas, so the
recovered pair identifies the rule rather than merely scoring a guess. The
whole point is that this needs only THEIR data. It does not care whether our
model, noise, or text embeddings match theirs.

    python3 solve_audio_rule.py --dir <their capture dir>
"""
import argparse, glob, os, re, sys
import numpy as np


def read_meta(d, i):
    """sigma/sigma_next for both streams, as the stage recorded them."""
    p = os.path.join(d, f"step{i}.meta.txt")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        m = re.match(r"\s*(\w+)\s*=\s*([-0-9.eE+]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def fit(x, v, xn):
    """Least squares for a, b in xn = a*x + b*v, plus the residual it leaves."""
    x, v, xn = (t.astype(np.float64).ravel() for t in (x, v, xn))
    G = np.array([[x @ x, x @ v], [x @ v, v @ v]])
    rhs = np.array([x @ xn, v @ xn])
    a, b = np.linalg.solve(G, rhs)
    resid = xn - (a * x + b * v)
    rms = float(np.sqrt((resid ** 2).mean()))
    scale = float(np.sqrt((xn ** 2).mean())) or 1e-30
    return a, b, rms / scale


def candidates(sig):
    """(name, a, b) for each rule, predicted from the sigmas alone.

    Every deterministic candidate has a == 1: they all move x by some multiple
    of v. They differ only in b -- which sigma gap they step across, and which
    sigma they used to form x0. So b alone names the rule.
    """
    sv, svn = sig.get("video_sigma"), sig.get("video_sigma_next")
    sa, san = sig.get("audio_sigma"), sig.get("audio_sigma_next")
    out = []
    if None not in (sa, san):
        out.append(("euler/DDIM at audio sigma", 1.0, -(sa - san)))
    if None not in (sv, svn):
        out.append(("euler at video sigma", 1.0, -(sv - svn)))
    if None not in (sa, san, sv) and sa:
        # x0 formed with the VIDEO sigma, then re-noised on the audio ladder
        out.append(("x0 at video sigma, audio ratio", 1.0, -(1 - san / sa) * sv))
    if None not in (sv, svn, sa) and sv:
        out.append(("x0 at audio sigma, video ratio", 1.0, -(1 - svn / sv) * sa))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--streams", default="audio,video")
    args = ap.parse_args()

    steps = sorted({int(os.path.basename(f).split(".")[0][4:])
                    for f in glob.glob(os.path.join(args.dir, "step*_latent_in.npy"))})
    if len(steps) < 2:
        print(f"need >= 2 rungs in {args.dir}, found {len(steps)}", file=sys.stderr)
        return 2

    for stream in args.streams.split(","):
        print(f"\n=== {stream} stream " + "=" * 46)
        for i in steps[:-1]:
            x = np.load(os.path.join(args.dir, f"step{i}.{stream}_latent_in.npy"))
            v = np.load(os.path.join(args.dir, f"step{i}.{stream}_velocity.npy"))
            nxt = os.path.join(args.dir, f"step{i+1}.{stream}_latent_in.npy")
            if not os.path.exists(nxt):
                continue
            a, b, rel = fit(x, v, np.load(nxt))
            sig = read_meta(args.dir, i)
            print(f"\nstep {i} -> {i+1}   "
                  f"sigma_v {sig.get('video_sigma', float('nan')):.4f}"
                  f"->{sig.get('video_sigma_next', float('nan')):.4f}   "
                  f"sigma_a {sig.get('audio_sigma', float('nan')):.4f}"
                  f"->{sig.get('audio_sigma_next', float('nan')):.4f}")
            print(f"  recovered:  a = {a:+.6f}   b = {b:+.6f}   "
                  f"residual {rel:.3e} of signal")
            if rel > 1e-3:
                print("  ** residual too large for an affine rule: this step "
                      "injects fresh noise, or uses state beyond (x, v).")
            best = None
            for name, ca, cb in candidates(sig):
                err = abs(a - ca) + abs(b - cb)
                mark = ""
                if best is None or err < best[0]:
                    best = (err, name); mark = ""
                print(f"     {name:<34} predicts a={ca:+.6f} b={cb:+.6f}"
                      f"   |db| {abs(b - cb):.2e}")
            if best:
                print(f"  -> closest: {best[1]}")
    print("\nThe rule that matches at EVERY rung is the rule. One that matches "
          "the first and drifts later is a coincidence of that rung's sigmas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
