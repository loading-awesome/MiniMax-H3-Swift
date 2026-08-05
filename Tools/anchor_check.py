#!/usr/bin/env python3
"""Did the keyframe anchor actually land on the frame it was aimed at?

The third oracle that needs no golden. `face-check` asks whether a face is
there and `speech_check` asks whether the words are; this asks whether the
picture handed to `--first-frame` / `--last-frame` is the picture the render
starts or ends on.

**A single similarity number proves nothing**, because two frames of the same
render are already similar to each other and any two photographs of a wall are
already similar to a downsampler. So every comparison is reported against a
*control*:

  * the anchor vs the frame it was aimed at, and
  * the anchor vs the frame at the other end of the same render.

An anchor that landed shows a large gap between those two. An anchor that was
parsed, validated, printed and then ignored shows roughly none — which is
exactly the failure mode a shape check cannot see, since a dropped conditioning
row leaves every downstream shape valid.

Similarity is normalised cross-correlation on a 32x18 luma thumbnail (structure,
insensitive to overall exposure) plus mean absolute error per channel on the
same thumbnail (colour). Both are reported; neither alone is the verdict.

    anchor_check.py render.mp4 --first door_closed.png --last door_open.png
"""
import argparse
import subprocess
import sys
import tempfile

import numpy as np

W, H = 32, 18


def thumb(path, frame=None, nb_frames=None):
    """[H, W, 3] float32 in [0,1]. `frame` picks a frame from a video."""
    with tempfile.NamedTemporaryFile(suffix=".rawvideo", delete=True) as tf:
        cmd = ["ffmpeg", "-y", "-v", "error"]
        if frame == "last":
            # -sseof is unreliable on short clips; select the final frame by index
            cmd += ["-i", path, "-vf",
                    f"select='eq(n\\,{nb_frames - 1})',scale={W}:{H}", "-vsync", "0"]
        elif frame == "first":
            cmd += ["-i", path, "-vf", f"select='eq(n\\,0)',scale={W}:{H}", "-vsync", "0"]
        else:
            cmd += ["-i", path, "-vf", f"scale={W}:{H}"]
        cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", tf.name]
        subprocess.run(cmd, check=True)
        raw = np.fromfile(tf.name, dtype=np.uint8)
    if raw.size != W * H * 3:
        raise SystemExit(f"{path}: expected {W*H*3} bytes, got {raw.size}")
    return raw.reshape(H, W, 3).astype(np.float32) / 255.0


def frame_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out.split(",")[0])


def luma(x):
    return x @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def ncc(a, b):
    """Normalised cross-correlation of two luma thumbnails, in [-1, 1]."""
    a, b = luma(a).ravel(), luma(b).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def mae(a, b):
    return float(np.abs(a - b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--first", help="the image passed to --first-frame")
    ap.add_argument("--last", help="the image passed to --last-frame")
    a = ap.parse_args()

    n = frame_count(a.video)
    f0 = thumb(a.video, "first")
    fN = thumb(a.video, "last", n)
    print(f"anchor-check {a.video}   {n} frames")
    print(f"  frame 0    mean rgb {f0.reshape(-1,3).mean(0).round(3)}")
    print(f"  frame {n-1:<4} mean rgb {fN.reshape(-1,3).mean(0).round(3)}")
    print(f"  the two ends of this render: ncc {ncc(f0, fN):+.3f}  mae {mae(f0, fN):.3f}")

    problems = []
    for label, path, aimed, other, other_label in (
            ("first", a.first, f0, fN, f"frame {n-1}"),
            ("last", a.last, fN, f0, "frame 0")):
        if not path:
            continue
        ref = thumb(path)
        hit_n, hit_m = ncc(ref, aimed), mae(ref, aimed)
        ctl_n, ctl_m = ncc(ref, other), mae(ref, other)
        tgt = "frame 0" if label == "first" else f"frame {n-1}"
        print(f"\n  --{label}-frame {path}")
        print(f"    anchor mean rgb        {ref.reshape(-1,3).mean(0).round(3)}")
        print(f"    vs {tgt:<10} (aimed)  ncc {hit_n:+.3f}  mae {hit_m:.3f}")
        print(f"    vs {other_label:<10} (control) ncc {ctl_n:+.3f}  mae {ctl_m:.3f}")
        gap = hit_n - ctl_n
        print(f"    structural gap         {gap:+.3f}"
              f"   colour gap {ctl_m - hit_m:+.3f}")
        # The bar is the control, not an absolute: an anchor that landed has to
        # look more like the end it was aimed at than like the other end.
        if gap <= 0.05 and (ctl_m - hit_m) <= 0.01:
            problems.append(
                f"--{label}-frame is no closer to {tgt} than to {other_label} "
                f"(ncc gap {gap:+.3f}, mae gap {ctl_m - hit_m:+.3f}) — the anchor "
                "may have been packed but not honoured")

    print("")
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
    else:
        print("  PASS: every anchor is closer to the frame it was aimed at")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
