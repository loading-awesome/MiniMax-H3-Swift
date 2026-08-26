#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
"""Does a per-projection precision change survive a whole trajectory?

`docs/ANE_PRECISION_RESULTS.md` measured the ANE's arithmetic against real
captured block taps and found it 3 to 20 times *more* accurate than the bf16
GPU path, per projection. That result is single-shot, and a render is not: 50
blocks times 20 steps is a thousand block evaluations, and a diffusion
trajectory can compound a small perturbation as easily as it can wash one out.
Nothing measured so far can distinguish those two outcomes.

This compares two complete renders that differ only in the DiT's compute dtype,
seed and sampler held identical. fp16 with a wide accumulator is the closest
proxy for the engine's datapath that runs on the GPU, so this answers the
propagation question with no private API and no bridge — and if it fails, the
whole ANE route is dead regardless of how the integration work turns out.

What to look for, in order:

  * **Divergence, not error.** Diffusion is chaotic in the sense that a small
    early perturbation can select a different sample entirely. A PSNR near 20
    dB with different content is a different render; a PSNR of 45 dB with the
    same content is the same render carrying rounding noise. Read the frames,
    not only the number.
  * **Growth across frames.** Video frames are decoded from one latent, so
    they do not index the trajectory — but a drift that grew during sampling
    tends to show as structure in the per-frame error rather than as a flat
    floor.
  * **Audio.** The model generates picture and sound from the same pass, and
    speech is the least forgiving thing in the output. Waveform correlation
    catches a drift that a video metric can average away.

    python3 Tools/ANE/dtype_drift.py a.mp4 b.mp4 [--audio a.wav b.wav]
"""

import argparse
import subprocess
import sys
import wave

import numpy as np


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def frames(path, width, height, limit=None):
    """Decodes to raw RGB. Small clips only; this holds the whole thing."""
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (width * height * 3)
    a = np.frombuffer(raw, np.uint8, count=n * width * height * 3)
    a = a.reshape(n, height, width, 3)
    return a[:limit] if limit else a


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sw = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(raw, dtype).astype(np.float64)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a / float(np.iinfo(dtype).max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--audio", nargs=2, metavar=("REF", "CAND"))
    args = ap.parse_args()

    w, h = probe(args.reference)
    w2, h2 = probe(args.candidate)
    if (w, h) != (w2, h2):
        print("geometry differs: %dx%d vs %dx%d" % (w, h, w2, h2))
        return 1

    a = frames(args.reference, w, h).astype(np.float64)
    b = frames(args.candidate, w, h).astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    print("compared %d frames at %dx%d\n" % (n, w, h))

    diff = b - a
    mse = float(np.mean(diff * diff))
    psnr = 10 * np.log10(255.0 * 255.0 / mse) if mse > 0 else float("inf")
    rel = float(np.sqrt(np.sum(diff * diff) / max(np.sum(a * a), 1e-30)))
    print("overall   PSNR %7.2f dB   rel RMS %.4g   max |delta| %3.0f/255"
          % (psnr, rel, float(np.max(np.abs(diff)))))
    identical = int(np.count_nonzero(np.all(diff == 0, axis=(1, 2, 3))))
    print("identical frames: %d of %d" % (identical, n))

    # Per-frame, to separate a flat rounding floor from a drift with structure.
    per = []
    for i in range(n):
        d = diff[i]
        m = float(np.mean(d * d))
        per.append(10 * np.log10(255.0 * 255.0 / m) if m > 0 else float("inf"))
    per = np.array(per)
    finite = per[np.isfinite(per)]
    if finite.size:
        print("per-frame PSNR: min %.2f  median %.2f  max %.2f  (first %.2f, last %.2f)"
              % (finite.min(), float(np.median(finite)), finite.max(),
                 per[0] if np.isfinite(per[0]) else -1,
                 per[-1] if np.isfinite(per[-1]) else -1))
        third = max(1, n // 3)
        print("thirds: %.2f / %.2f / %.2f dB"
              % (float(np.mean(per[:third])), float(np.mean(per[third:2 * third])),
                 float(np.mean(per[2 * third:]))))

    if args.audio:
        try:
            x, y = read_wav(args.audio[0]), read_wav(args.audio[1])
        except Exception as exc:                       # noqa: BLE001
            print("\naudio: unreadable (%s)" % exc)
            x = y = None
        if x is not None:
            m = min(len(x), len(y))
            x, y = x[:m], y[:m]
            corr = float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))
            arel = float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30))
            print("\naudio     %d samples   correlation %.6f   rel RMS %.4g"
                  % (m, corr, arel))

    print("""
Reading it. Above ~40 dB with no structure across thirds is a rounding floor:
the same render, carrying arithmetic noise. Below ~25 dB, or a clear trend
across thirds, is trajectory divergence — the perturbation selected a
different sample, and per-projection accuracy did not survive the loop.
Audio correlation is the sharper test; speech tolerates far less drift than
picture does.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
