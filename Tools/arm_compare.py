#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
"""Compare configurations by distribution, when they cannot be compared by clip.

**Why this exists.** Every change to the cache policy changes the sampling
trajectory, so every arm renders a *different video* — not a degraded version
of the same one. That breaks the obvious comparison: the detail ratio between
two clips is only meaningful while their compositions still match, and past a
structural dissimilarity of about 0.01 they do not.

That is not a theoretical worry. Measured on one probe, the **dense** arm — no
cache, no approximation, the best render obtainable — scored *lower* detail than
a cached arm, at dssim 0.108. The metric was reading which market stall was in
shot.

So this tool never compares clip to clip. It takes several seeds per arm,
measures each clip **on its own**, and asks whether the arms' distributions
differ by more than the seeds within an arm already do. Scene variation is
noise that averages out; a systematic softening is not.

    arm_compare.py cap3=a1.mp4,a2.mp4,a3.mp4 cap5=b1.mp4,b2.mp4,b3.mp4

**Three seeds is not many.** With small samples the honest statement is the
range, not a p-value, so that is what gets printed: an arm's spread across
seeds, beside the gap between arms. A gap smaller than the spread is not a
finding, and is labelled as such rather than left for the reader to notice.
"""

import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from coherence_check import measure  # noqa: E402

# Higher is better for all of these except pulse, where lower is steadier.
KEYS = [("detail", "higher is sharper"),
        ("accel", "lower is steadier"),
        ("motion", "context only"),
        ("pulse", "lower is steadier")]


def main():
    arms = {}
    order = []
    for spec in sys.argv[1:]:
        if "=" not in spec:
            print(f"usage: {sys.argv[0]} name=clip.mp4,clip.mp4 ...")
            return 1
        name, clips = spec.split("=", 1)
        arms[name] = clips.split(",")
        order.append(name)
    if len(order) < 2:
        print("need at least two arms")
        return 1

    stats = {}
    for name in order:
        rows = [measure(c) for c in arms[name]]
        stats[name] = {k: np.array([r[k] for r in rows]) for k, _ in KEYS}

    base = order[0]
    for key, sense in KEYS:
        print(f"\n{key}  ({sense})")
        print(f"  {'arm':<12}{'n':>3}{'median':>12}{'spread':>10}{'vs ' + base:>10}   verdict")
        b = stats[base][key]
        bmed = float(np.median(b))
        bspread = float(b.max() - b.min()) / bmed if bmed else 0.0
        for name in order:
            v = stats[name][key]
            med = float(np.median(v))
            spread = (float(v.max() - v.min()) / med) if med else 0.0
            gap = (med / bmed - 1.0) if bmed else 0.0
            if name == base:
                verdict = "reference"
            else:
                # The comparison that matters: is the gap between arms larger
                # than the variation the seeds already produce inside them? The
                # widest within-arm spread is the floor, not the narrowest —
                # using the reference's alone would flatter whichever arm
                # happened to draw consistent seeds.
                floor = max(spread, bspread)
                verdict = ("below the seed spread — not a finding"
                           if abs(gap) <= floor else
                           f"clears the {floor * 100:.1f}% seed spread")
            print(f"  {name:<12}{len(v):>3}{med:>12.6f}{spread * 100:>9.1f}%"
                  f"{gap * 100:>9.1f}%   {verdict}")

    print("\n  Absolute per-clip measures only. No ratio between two clips is used,")
    print("  because arms render different scenes and such a ratio compares content.")
    print("  With a handful of seeds the seed spread is the noise floor; a gap")
    print("  inside it means the sample is too small to say, not that the arms agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
