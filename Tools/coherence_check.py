#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
"""Temporal coherence and detail, for comparing cache settings.

**Every measure here can be gamed by blur, and one of them was.** During the
Sol-Attn sweep, ranking on temporal smoothness alone would have crowned the
blurriest clip in the set — it scored best on frame-to-frame stability precisely
because it had lost the detail that was moving. So detail is reported beside
every temporal number and neither is a ranking key on its own.

What each column is:

  accel     Median frame-to-frame *acceleration* of the luminance field: the
            second difference, |(f[t+1]-f[t]) - (f[t]-f[t-1])|. First
            differences measure motion, which a moving subject is supposed to
            have. Acceleration measures motion that changes abruptly, which is
            what pulsing and warping look like. Reported relative to the
            reference clip.

  detail    Median Laplacian variance per frame — high-frequency content.
            **The blur guard.** A clip that improves on `accel` while dropping
            here has not become steadier, it has become softer.

  pulse     Fraction of frame-pair energy in the 1.5-10 Hz band of the
            per-frame mean luminance. The band matters: an earlier version of
            this measurement used >6 Hz and scored a visibly pulsing clip
            *better* than dense, because the pulse was slower than the band it
            was looking in.

  dssim     Structural dissimilarity from the reference, averaged over frames.
            Context only. Two cache settings produce genuinely different
            renders — a different trajectory, not a degraded one — so a large
            value here is expected and means little by itself.

None of this replaces watching the clip. It is here to say which clips are
worth watching first, and to catch the failure that a viewer would catch late.
"""

import subprocess, sys, json
import numpy as np


def frames(path, width=216, height=120):
    """Greyscale frames, downsampled. Downsampling is deliberate: the artifacts
    being hunted are regional, and full resolution mostly adds sensor-level
    noise to every measure."""
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"scale={width}:{height}", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (width * height)
    return np.frombuffer(raw, np.uint8)[: n * width * height] \
             .reshape(n, height, width).astype(np.float32) / 255.0


def laplacian_variance(f):
    lap = (-4 * f[1:-1, 1:-1] + f[:-2, 1:-1] + f[2:, 1:-1]
           + f[1:-1, :-2] + f[1:-1, 2:])
    return float(lap.var())


def measure(path, fps=24.0):
    f = frames(path)
    d1 = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    d2 = np.abs(np.diff(f, n=2, axis=0)).mean(axis=(1, 2))
    mean_luma = f.mean(axis=(1, 2))

    # Pulse: energy in 1.5-10 Hz of the mean-luminance signal.
    sig = mean_luma - mean_luma.mean()
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freq = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    band = (freq >= 1.5) & (freq <= 10.0)
    pulse = float(spec[band].sum() / max(spec.sum(), 1e-12))

    return dict(frames=len(f),
                motion=float(np.median(d1)),
                accel=float(np.median(d2)),
                detail=float(np.median([laplacian_variance(x) for x in f])),
                pulse=pulse)


def dssim(a_path, b_path):
    a, b = frames(a_path), frames(b_path)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    # Global SSIM per frame, constants for data in [0, 1].
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for x, y in zip(a, b):
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mx) * (y - my)).mean()
        s = ((2 * mx * my + c1) * (2 * cov + c2)) / \
            ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
        out.append((1 - s) / 2)
    return float(np.mean(out))


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <reference.mp4> <arm.mp4> [arm.mp4 ...]")
        return 1
    ref = sys.argv[1]
    base = measure(ref)
    print(f"reference: {ref}")
    print(f"  {base['frames']} frames, motion {base['motion']:.5f}, "
          f"accel {base['accel']:.5f}, detail {base['detail']:.6f}, "
          f"pulse {base['pulse']:.4f}\n")
    print(f"{'arm':<20}{'accel':>9}{'detail':>9}{'motion':>9}{'pulse':>9}{'dssim':>8}")
    print(f"{'':<20}{'(rel)':>9}{'(rel)':>9}{'(rel)':>9}{'abs':>9}{'':>8}")
    for path in sys.argv[2:]:
        m = measure(path)
        name = path.split("/")[-1].replace(".mp4", "")
        print(f"{name:<20}"
              f"{m['accel'] / base['accel']:>9.3f}"
              f"{m['detail'] / base['detail']:>9.3f}"
              f"{m['motion'] / base['motion']:>9.3f}"
              f"{m['pulse']:>9.4f}"
              f"{dssim(ref, path):>8.4f}")
    print("\n  accel above ~1.1 with detail at or below 1.0 is the warping signature.")
    print("  detail below ~0.95 is blur, whatever the temporal numbers say.")
    print("  neither substitutes for watching the clip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
