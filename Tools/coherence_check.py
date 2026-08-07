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

  detail    Median Laplacian variance per frame **at native resolution** —
            high-frequency content. The blur guard: a clip that improves on
            `accel` while dropping here has not become steadier, it has become
            softer. Measured natively because measuring it on a downsampled
            copy low-passes away the thing being measured, and did — see
            `frames`.

  dssim     Also, and first: **if this exceeds about 0.01 the detail ratio
            beside it means nothing.** Two cache settings produce different
            trajectories and therefore different scenes, and comparing the
            Laplacian variance of two different scenes compares their content.
            The tell that this is real: on one probe the *dense* arm — no
            cache, no approximation, and by construction the best available —
            scored lower detail than a cached one, at both resolutions, with
            dssim 0.108.

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


def frames(path, width=None, height=None):
    """Greyscale frames. Native resolution unless a size is given.

    **The two families of measure need different resolutions, and conflating
    them produced a wrong answer.** This function downsampled to 216x120 for
    everything, on the reasoning that the artifacts being hunted are regional
    and full resolution adds noise. That is correct for the temporal measures
    and exactly backwards for the detail one: an area-average to a quarter of
    the linear resolution is a low-pass filter, so the "detail" figure was
    reading mid-frequency structure with the fine detail already removed.

    It reversed a result. On the beach probe, cap 5 measured 0.884 against
    cap 3 downsampled — an 11.6% detail loss, which is what a configuration was
    rejected on — and 1.037 at native resolution. Opposite sign.

    So detail is measured natively and the temporal measures keep their
    downsampling, which they need: at native resolution frame-to-frame
    differences are dominated by per-pixel generation noise rather than by the
    regional motion the artifacts live in.
    """
    if width and height:
        vf = ["-vf", f"scale={width}:{height}"]
    else:
        vf = []
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
            capture_output=True, text=True, check=True).stdout.strip()
        width, height = (int(v) for v in probe.split("x"))
    cmd = ["ffmpeg", "-v", "error", "-i", path, *vf, "-pix_fmt", "gray",
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
    # Detail natively; motion, acceleration and pulse on the downsampled copy.
    native = frames(path)
    detail = float(np.median([laplacian_variance(x) for x in native]))

    f = frames(path, 216, 120)
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
                detail=detail,
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
        d = dssim(ref, path)
        # The ratios are printed as unusable rather than printed and caveated.
        # A number with a footnote gets quoted without the footnote.
        flag = "" if d < 0.01 else "   <- different scene, ratios void"
        print(f"{name:<20}"
              f"{m['accel'] / base['accel']:>9.3f}"
              f"{m['detail'] / base['detail']:>9.3f}"
              f"{m['motion'] / base['motion']:>9.3f}"
              f"{m['pulse']:>9.4f}"
              f"{d:>8.4f}{flag}")
    print("\n  Read dssim first. Above ~0.01 the arms rendered different scenes and")
    print("  every ratio on that row is comparing content, not quality.")
    print("  Below it: accel above ~1.1 with detail at or below 1.0 is warping;")
    print("  detail below ~0.95 is blur, whatever the temporal numbers say.")
    print("  Neither substitutes for watching the clip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
