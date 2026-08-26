#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
"""Can a projection's interior partial sums reach the engine's 2^15 cliff?

`docs/ANE_PRECISION_RESULTS.md` established that saturation is the one failure
mode of this hardware that cannot be detected downstream: a running partial that
reaches 2^15 makes the output port return **zero**, not an infinity or a NaN,
and a single destroyed element in 4,096 moves a projection's relative error by
three orders of magnitude while the output still looks like an activation. Every
projection routed to the engine therefore needs a bound on `max|interior
partial|` with margin, and `fc2` breached it at block 49.

That measurement sampled a 64x64 corner. This one is a proof over everything the
oracle captured, and it differs in two ways that matter.

**It bounds every accumulation order, not one of them.** A cumulative sum in
index order is a single ordering out of `K!`, and the engine's internal order is
not documented — it tiles the contraction across its MAC array in a pattern we
have not reverse-engineered. Any claim resting on index order is a claim about a
computation the hardware does not perform. What holds regardless of order is

    |any partial sum| <= sum_k |a_k * w_k|

so that L1 quantity is the bound to test, and it happens to be one GEMM against
the absolute values. Where it clears the threshold, saturation is impossible
under *any* order the hardware might use. Where it does not, nothing is proven
either way and the index-order figure is reported alongside to show the gap.

**It covers every captured row and every output channel.** All 1024 oracle rows
against all `N` columns, rather than 64 x 64. The oracle itself samples 1024 of
15,406 sequence positions, which this cannot fix — that limit is stated in the
output rather than hidden, because a bound over a sample is not a bound.

    python3 Tools/ANE/saturation_bound.py \\
        --oracles /path/oracle_prod_matrix \\
        --checkpoint /path/MiniMax-H3-FL2VA_bf16.safetensors
"""

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

MAC_SATURATION = 32768.0        # 2^15; a partial reaching this returns zero

# The four projections, as (name, oracle input tap, checkpoint weight).
# Inputs are the real modulated activations a block multiplies, not proxies.
TAPS = [
    ("qkv",      "ref.mod_scale_shift.1", "blocks.{b}.attn.qkv_proj.weight"),
    ("attn out", "ref.attn.sdpa",         "blocks.{b}.attn.out_proj.weight"),
    ("mlp fc1",  "ref.mod_scale_shift.2", "blocks.{b}.mlp.fc1.weight"),
    ("mlp fc2",  "ref.mlp.swiglu",        "blocks.{b}.mlp.fc2.weight"),
]

# Which projections the tree actually routes. `fc2` is the one this tool exists
# to rule on, so it is listed but not marked routed.
ROUTED = {"qkv", "attn out", "mlp fc1"}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_bf16(path, header, offset0, name):
    entry = header[name]
    assert entry["dtype"] == "BF16", entry["dtype"]
    begin, _ = entry["data_offsets"]
    rows, cols = entry["shape"]
    raw = np.memmap(path, dtype=np.uint16, mode="r",
                    offset=offset0 + begin, shape=(rows * cols,))
    wide = np.zeros(raw.shape, dtype=np.uint32)
    wide[:] = raw
    wide <<= 16
    return wide.view(np.float32).reshape(rows, cols)


def read_f32(path, header, offset0, name):
    entry = header[name]
    assert entry["dtype"] == "F32", entry["dtype"]
    begin, _ = entry["data_offsets"]
    shape = entry["shape"]
    r, c = shape[0], int(np.prod(shape[1:]))
    return np.memmap(path, dtype=np.float32, mode="r",
                     offset=offset0 + begin, shape=(r, c))


def order_free_bound(a, w, chunk=512):
    """max over (row, channel) of sum_k |a_k| |w_k| — one GEMM on magnitudes.

    Chunked over output channels so a [1024, 28672] intermediate never has to
    exist at once for `fc1`.
    """
    absa = np.abs(np.asarray(a, dtype=np.float32))
    worst = 0.0
    for start in range(0, w.shape[0], chunk):
        block = np.abs(np.asarray(w[start:start + chunk], dtype=np.float32))
        worst = max(worst, float(np.max(absa @ block.T)))
    return worst


def index_order_max(a, w, rows, channels, step=1024):
    """max|running sum| in index order, for the subset (rows x channels).

    Reported only to show how much slack the order-free bound carries. It is
    not the safety criterion: the engine does not promise this order.
    """
    worst = 0.0
    wsub = np.asarray(w[:channels], dtype=np.float64)
    for r in range(min(rows, a.shape[0])):
        row = np.asarray(a[r], dtype=np.float64)
        running = np.zeros(wsub.shape[0], dtype=np.float64)
        peak = np.zeros(wsub.shape[0], dtype=np.float64)
        for start in range(0, row.shape[0], step):
            products = row[start:start + step][None, :] * wsub[:, start:start + step]
            partial = running[:, None] + np.cumsum(products, axis=1)
            peak = np.maximum(peak, np.max(np.abs(partial), axis=1))
            running = partial[:, -1]
        worst = max(worst, float(np.max(peak)))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracles", required=True,
                    help="directory of oracle_* subdirectories, each with oracle.safetensors")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scale", type=float, default=0.0625,
                    help="operand scale the routed path applies (default 1/16)")
    ap.add_argument("--index-rows", type=int, default=4,
                    help="rows to also measure in index order, for slack (0 to skip)")
    ap.add_argument("--index-channels", type=int, default=256)
    args = ap.parse_args()

    oracles = sorted(glob.glob(os.path.join(args.oracles, "*", "oracle.safetensors")))
    if not oracles:
        ap.error("no oracle.safetensors under %s" % args.oracles)

    chead, coff = read_header(args.checkpoint)

    # The engine sees the scaled operands, so the threshold on the unscaled
    # quantity this tool measures is 2^15 divided by the scale.
    threshold = MAC_SATURATION / args.scale
    print("Order-free saturation bound — operand scale %g, so unscaled partials "
          "must stay under %.0f\n" % (args.scale, threshold))

    verdicts = {}
    needed = {}
    for path in oracles:
        ghead, goff = read_header(path)
        meta = ghead.get("__metadata__", {})
        block = int(meta.get("block", -1))
        tag = os.path.basename(os.path.dirname(path))

        print("%s  (block %d, seq_len %s, %s rows captured)"
              % (tag, block, meta.get("seq_len", "?"),
                 ghead["ref.mlp.swiglu"]["shape"][0] if "ref.mlp.swiglu" in ghead else "?"))
        print("  %-10s %8s %8s %14s %10s %9s %8s   %s"
              % ("projection", "rows", "cols", "bound", "headroom", "index", "scale", "verdict"))

        for name, tap, wkey in TAPS:
            key = wkey.format(b=block)
            if tap not in ghead or key not in chead:
                print("  %-10s missing" % name)
                continue
            a = read_f32(path, ghead, goff, tap)
            w = read_bf16(args.checkpoint, chead, coff, key)

            bound = order_free_bound(a, w)
            headroom = threshold / bound if bound > 0 else float("inf")
            safe = bound < threshold

            idx = ""
            if args.index_rows:
                idx = "%9.0f" % index_order_max(a, w, args.index_rows, args.index_channels)

            # The largest power-of-two operand scale that proves this safe with
            # a factor of two to spare. Powers of two only: any other factor is
            # inexact in fp16 and would change the arithmetic rather than just
            # relocating it.
            need = 1.0
            while bound * need >= MAC_SATURATION / 2 and need > 2.0 ** -20:
                need /= 2.0
            scale_txt = "1/%d" % round(1 / need) if need < 1 else "1"

            verdict = "PROVEN SAFE" if safe else "NOT PROVEN"
            if not safe:
                verdict += " (exceeds by %.1fx)" % (bound / threshold)
            key2 = (name, block)
            verdicts[key2] = min(verdicts.get(key2, float("inf")), headroom)
            needed[name] = min(needed.get(name, 1.0), need)
            print("  %-10s %8d %8d %14.0f %9.1fx %s %8s   %s"
                  % (name, a.shape[0], w.shape[0], bound, headroom, idx, scale_txt, verdict))
        print()

    print("Worst headroom per projection, across every oracle:\n")
    print("  %-10s %6s   %9s   %s" % ("projection", "block", "headroom", ""))
    byname = {}
    for (name, block), h in sorted(verdicts.items()):
        byname.setdefault(name, []).append((block, h))
    for name, rows in byname.items():
        worst_block, worst = min(rows, key=lambda r: r[1])
        note = "routed" if name in ROUTED else "NOT routed"
        need = needed.get(name, 1.0)
        scale_txt = "1/%d" % round(1 / need) if need < 1 else "1"
        flag = "" if worst > 1.0 else "  <-- CAN SATURATE at the current scale"
        print("  %-10s %6d   %8.1fx   needs scale %-6s %s%s"
              % (name, worst_block, worst, scale_txt, note, flag))

    print("""
Reading it. `headroom` is how many times the worst partial this projection can
possibly produce would have to grow before it could reach the cliff, under any
accumulation order. Above 1.0 is a proof for the rows the oracle captured;
`index` is the index-order figure for a small subset, and the gap between the
two is slack in the bound, not danger.

What this does not cover: the oracle holds 1024 of 15,406 sequence positions and
three of fifty blocks. A bound over a sample is evidence, not a proof, and the
honest scope of the claim is "over everything captured".
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
