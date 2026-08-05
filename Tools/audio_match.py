#!/usr/bin/env python3
"""Did the render's audio come from the reference clip, or from the prompt?

`speech_check.py` asks whether the words are right, and for a2v that is not
enough on its own: the prompt names the same line the reference speaks, so a
correct transcript is consistent with the reference having been packed,
validated, printed and then ignored. Whisper cannot tell those apart.

This can. It compares the rendered waveform to the reference waveform in two
ways that do not care what the words are:

  * **envelope correlation** — short-time RMS on a 20 ms grid, then Pearson
    correlation at the best lag within +/- 250 ms. This is where speech *is*,
    not what it says: syllable onsets, pauses, the shape of the phrase.
  * **log-mel-ish spectral correlation** — per-band energy over time on a coarse
    band grid, which adds "and it sounds like the same voice saying it".

Both are reported against a **control** clip, because two recordings of one
sentence are already correlated and a bare number cannot be read. `fully_copy`
should score far above the control; a prompt-driven render that ignored the
reference should score near it.

    audio_match.py render.wav --reference drive.wav --control other_render.wav
"""
import argparse
import subprocess
import sys
import tempfile

import numpy as np

SR = 16000
HOP = int(0.020 * SR)          # 20 ms
MAX_LAG = int(0.250 / 0.020)   # +/- 250 ms, in frames


def load(path):
    """Mono float32 at 16 kHz, via ffmpeg so any container works."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=True) as tf:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1",
             "-ar", str(SR), "-f", "f32le", tf.name], check=True)
        x = np.fromfile(tf.name, dtype=np.float32)
    if x.size == 0:
        raise SystemExit(f"{path}: no samples")
    return x


def envelope(x):
    n = len(x) // HOP
    e = np.abs(x[:n * HOP].reshape(n, HOP)).mean(1)
    return np.log(e + 1e-6)


def bands(x, nbands=16):
    """[frames, nbands] log energy. A coarse spectrogram, no mel table needed."""
    n = len(x) // HOP
    frames = x[:n * HOP].reshape(n, HOP) * np.hanning(HOP)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    # geometric band edges: speech energy is not uniform across linear bins
    edges = np.geomspace(2, spec.shape[1], nbands + 1).astype(int)
    out = np.stack([spec[:, edges[i]:max(edges[i + 1], edges[i] + 1)].mean(1)
                    for i in range(nbands)], axis=1)
    return np.log(out + 1e-6)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def best_lag_corr(a, b):
    """Max correlation over lags, and the lag that achieved it (in frames)."""
    best, best_l = -1.0, 0
    for l in range(-MAX_LAG, MAX_LAG + 1):
        if l >= 0:
            x, y = a[l:], b[:len(b) - l] if l else b
        else:
            x, y = a[:len(a) + l], b[-l:]
        n = min(len(x), len(y))
        if n < 10:
            continue
        c = corr(x[:n], y[:n])
        if c > best:
            best, best_l = c, l
    return best, best_l


def score(ref, other):
    ev, lag = best_lag_corr(envelope(ref), envelope(other))
    ra, rb = bands(ref), bands(other)
    n = min(len(ra), len(rb))
    sp = float(np.mean([corr(ra[:n, k], rb[:n, k]) for k in range(ra.shape[1])]))
    return ev, lag * 0.020, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="the rendered wav")
    ap.add_argument("--reference", required=True, help="the clip passed to --reference-audio")
    ap.add_argument("--control", help="an unrelated render, to calibrate the number")
    a = ap.parse_args()

    rendered = load(a.audio)
    reference = load(a.reference)
    print(f"audio-match {a.audio}")
    print(f"  rendered   {len(rendered)/SR:.2f}s")
    print(f"  reference  {a.reference} ({len(reference)/SR:.2f}s)")

    ev, lag, sp = score(reference, rendered)
    print(f"\n  vs reference   envelope {ev:+.3f} (lag {lag:+.2f}s)   spectral {sp:+.3f}")

    problems = []
    if a.control:
        cev, clag, csp = score(reference, load(a.control))
        print(f"  vs control     envelope {cev:+.3f} (lag {clag:+.2f}s)   spectral {csp:+.3f}")
        print(f"    ({a.control} — an unrelated render, never conditioned on the reference)")
        print(f"\n  envelope gap {ev - cev:+.3f}   spectral gap {sp - csp:+.3f}")
        # The bar is the control. An absolute threshold on speech correlation is
        # not defensible; "more like the reference than an unrelated render is"
        # is.
        if (ev - cev) <= 0.10 and (sp - csp) <= 0.05:
            problems.append(
                f"the render's audio is no more like the reference (env {ev:+.3f}, "
                f"spec {sp:+.3f}) than an unrelated render is (env {cev:+.3f}, "
                f"spec {csp:+.3f}) — --reference-audio may have been packed but "
                "not honoured")
    else:
        print("\n  no --control given: this number is not interpretable on its own")

    print("")
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
    else:
        print("  PASS: the render's audio tracks the reference clip")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
