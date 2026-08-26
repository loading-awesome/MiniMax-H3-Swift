#!/usr/bin/env python3
"""Does the ANE's denormal flush cost this model anything?

The QKV spike measured 3.17% relative error against an FP64 oracle and it was
almost entirely underflow: a third of that fixture's products fell below fp16's
smallest normal, 6.10e-5, and this engine flushes denormal products to zero
inside the multiply-accumulate (`Tools/ANE/numerics.m`). Scaling the operands so
the products clear the boundary dropped the error 152x, onto the arithmetic
floor. So the production question is not cancellation and not K: it is what
fraction of *this model's* products underflow.

Half of that is exactly measurable right now. The weights are real, in the
checkpoint, and this reads them. The other half is the activation scale, which
is not captured anywhere, so it stays an explicit variable and the answer is
reported as a function of it. That is the honest shape of the answer, and it
identifies the threshold that matters: the activation RMS below which underflow
starts costing accuracy.

The ANE's arithmetic is simulated rather than assumed. `simulate_ane` implements
what the probes measured — per-product rounding to fp16, denormals flushed, and
an exact wide accumulator — and `--validate` checks that simulation against the
hardware numbers already recorded, at four fixture scales, before any of its
predictions about real weights are believed.

    python3 Tools/ANE/underflow.py --validate
    python3 Tools/ANE/underflow.py --checkpoint /path/MiniMax-H3-FL2VA_bf16.safetensors
"""

import argparse
import json
import struct
import sys

import numpy as np

FP16_MIN_NORMAL = 6.103515625e-05      # 2^-14
FP16_MIN_DENORMAL = 5.960464477539063e-08
MAC_SATURATION = 32768.0               # 2^15; a partial reaching this returns zero

PROJECTIONS = [
    ("qkv", "blocks.{b}.attn.qkv_proj.weight", 5376),
    ("attn out", "blocks.{b}.attn.out_proj.weight", 7168),
    ("mlp fc1", "blocks.{b}.mlp.fc1.weight", 5376),
    ("mlp fc2", "blocks.{b}.mlp.fc2.weight", 14336),
]


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_bf16(path, header, offset0, name, max_rows=None):
    """Reads a BF16 tensor as float32. Rows are the output channels."""
    entry = header[name]
    assert entry["dtype"] == "BF16", entry["dtype"]
    begin, _ = entry["data_offsets"]
    shape = entry["shape"]
    # Norm gains are rank 1; projection weights are rank 2. Both are read here,
    # so treat a vector as a single row rather than assuming a matrix.
    rows, cols = (shape[0], 1) if len(shape) == 1 else (shape[0], shape[1])
    if max_rows is not None and max_rows < rows:
        rows = max_rows
    raw = np.memmap(path, dtype=np.uint16, mode="r",
                    offset=offset0 + begin, shape=(rows * cols,))
    wide = np.zeros(raw.shape, dtype=np.uint32)
    wide[:] = raw
    wide <<= 16
    return wide.view(np.float32).reshape(rows, cols)


def underflow_fraction(weight_rms, activation_rms, samples=1_000_000, rng=None):
    """Fraction of products below fp16's smallest normal.

    Both operands are modelled as zero-mean Gaussians, which is what the
    weights measure as. The product of two Gaussians is heavy at zero, so this
    fraction is never negligible — the question is whether it is 1% or 30%.
    """
    rng = rng or np.random.default_rng(0)
    x = rng.normal(0.0, activation_rms, samples)
    w = rng.normal(0.0, weight_rms, samples)
    p = np.abs(x * w)
    return float(np.mean(p < FP16_MIN_NORMAL))


def to_fp16_with_flush(values):
    """fp16 rounding with denormals flushed to zero, as the MAC does."""
    rounded = values.astype(np.float16).astype(np.float64)
    return np.where(np.abs(rounded) < FP16_MIN_NORMAL, 0.0, rounded)


def simulate_ane(activation, weight, saturate=True):
    """The measured datapath: fp16 product, flushed, into an exact accumulator,
    with the output port's 2^15 saturation.

    activation [S,K] and weight [K,N] in float64. Returns [S,N] float64 before
    the final store, so the caller can compare against an exact reference
    without the output rounding confounding it.

    Saturation is modelled because leaving it out flatters the result exactly
    where it matters: `numerics.m` measured that a running partial reaching
    2^15 makes the dot product return ZERO, not inf and not NaN, so an
    unmodelled saturation shows up as a small relative error instead of a
    destroyed output. Returns (result, saturated_fraction).
    """
    out = np.zeros((activation.shape[0], weight.shape[1]), dtype=np.float64)
    hit = 0
    for s in range(activation.shape[0]):
        products = to_fp16_with_flush(activation[s][:, None] * weight)
        if saturate:
            partial = np.cumsum(products, axis=0)
            blown = np.max(np.abs(partial), axis=0) >= MAC_SATURATION
            total = partial[-1]
            total[blown] = 0.0
            hit += int(np.count_nonzero(blown))
            out[s] = total
        else:
            out[s] = products.sum(axis=0)      # wide accumulator: exact
    return out, hit / float(out.size)


def rel_rms(actual, exact):
    err = actual - exact
    return float(np.sqrt(np.sum(err * err) / max(np.sum(exact * exact), 1e-300)))


def validate():
    """Check the simulator against hardware before trusting it on real data.

    The spike's fixture is uniform, deterministic, and was measured on silicon
    at four scales. If the simulation reproduces that curve, the model of the
    datapath is right and its predictions elsewhere mean something.
    """
    print("Validating the simulated datapath against measured hardware")
    print("fixture: spike synthetic, K=5376, uniform +-0.03 x +-0.02, scaled\n")
    measured = {1: 0.0315775, 4: 0.00147276, 16: 0.000224582, 64: 0.000207435}
    K, rows = 5376, 24
    rng = np.random.default_rng(7)
    print("%6s %14s %14s %10s %12s" % ("scale", "measured", "simulated", "ratio", "underflow"))
    for scale in (1, 4, 16, 64):
        x = rng.uniform(-0.03, 0.03, (rows, K)) * scale
        w = rng.uniform(-0.02, 0.02, (K, 8)) * scale
        # bf16 is what the real fixture stores; keep the simulation honest.
        x = x.astype(np.float32).astype(np.float64)
        w = w.astype(np.float32).astype(np.float64)
        exact = x @ w
        got, _ = simulate_ane(x, w)
        r = rel_rms(got, exact)
        frac = float(np.mean(np.abs(x[0][:, None] * w) < FP16_MIN_NORMAL))
        print("%5dx %14.6g %14.6g %9.2fx %11.1f%%"
              % (scale, measured[scale], r, r / measured[scale], 100 * frac))
    print("\nagreement within a small factor across two orders of magnitude in")
    print("error means the flush-plus-round model is the right one.\n")


def read_f32(path, header, offset0, name, rows=None):
    entry = header[name]
    assert entry["dtype"] == "F32", entry["dtype"]
    begin, _ = entry["data_offsets"]
    shape = entry["shape"]
    r, c = shape[0], int(np.prod(shape[1:]))
    if rows is not None and rows < r:
        r = rows
    return np.memmap(path, dtype=np.float32, mode="r",
                     offset=offset0 + begin, shape=(r, c))


# The four projections, as (name, golden input tap, checkpoint weight, K).
# The inputs are the real modulated activations a block actually multiplies:
# qkv and fc1 consume the AdaLN-modulated norm output, out_proj consumes the
# attention result, and fc2 consumes the SwiGLU activation.
GOLDEN_TAPS = [
    ("qkv",      "ref.mod_scale_shift.1", "blocks.{b}.attn.qkv_proj.weight", "ref.attn.qkv_proj"),
    ("attn out", "ref.attn.sdpa",         "blocks.{b}.attn.out_proj.weight", "ref.attn.out_proj"),
    ("mlp fc1",  "ref.mod_scale_shift.2", "blocks.{b}.mlp.fc1.weight",       "ref.mlp.fc1"),
    ("mlp fc2",  "ref.mlp.swiglu",        "blocks.{b}.mlp.fc2.weight",       "ref.mlp.fc2"),
]


def measure_golden(golden, checkpoint, block, rows, columns, scale=1.0):
    """The real answer: real activations against real weights, no distribution
    model anywhere. Reports what fraction of the products a block actually
    computes fall below fp16's smallest normal and are flushed, how close its
    interior partials come to the 2^15 threshold that silently returns zero,
    and what the engine's arithmetic would cost against an exact reference."""
    ghead, goff = read_header(golden)
    chead, coff = read_header(checkpoint)
    print("Real activations x real weights — block %d, %d rows, %d columns\n"
          % (block, rows, columns))
    print("%-10s %8s %11s %11s %12s %13s %11s"
          % ("projection", "K", "act RMS", "|w| RMS", "underflow", "max|partial|", "ANE relRMS"))
    for name, tap, wkey, _out in GOLDEN_TAPS:
        key = wkey.format(b=block)
        if tap not in ghead or key not in chead:
            print("%-10s  missing (%s)" % (name, tap if tap not in ghead else key))
            continue
        x = np.asarray(read_f32(golden, ghead, goff, tap, rows), dtype=np.float64)
        # A power-of-two operand scale is exact in fp16 and undone exactly on
        # the output, so it moves the partial-sum envelope away from 2^15
        # without changing the arithmetic. Products shrink by the same factor,
        # so the scale must stay well above the denormal boundary too.
        x = x * scale
        w_all = read_bf16(checkpoint, chead, coff, key, max_rows=columns)
        w = np.asarray(w_all, dtype=np.float64).T          # [K, columns]
        if w.shape[0] != x.shape[1]:
            print("%-10s  shape mismatch: x K=%d, w K=%d" % (name, x.shape[1], w.shape[0]))
            continue
        products = x[:, :, None] * w[None, :, :]           # [rows, K, columns]
        under = float(np.mean(np.abs(products) < FP16_MIN_NORMAL))
        partial = np.cumsum(products, axis=1)
        max_partial = float(np.max(np.abs(partial)))
        exact = x @ w
        got, blown = simulate_ane(x, w)
        # relative error is scale-invariant, so no unscaling is needed to score
        flag = "  ZEROED %.2f%%" % (100 * blown) if blown else ""
        print("%-10s %8d %11.4g %11.4g %11.2f%% %13.1f %11.3g%s"
              % (name, w.shape[0], float(np.sqrt(np.mean(x * x))),
                 float(np.sqrt(np.mean(w * w))), 100 * under, max_partial,
                 rel_rms(got, exact), flag))
        del products, partial

    print("""
underflow is the share of products below fp16's smallest normal (6.10e-5),
which this engine flushes to zero inside the multiply-accumulate.
max|partial| is against 32768, the value at which a running partial makes the
whole dot product return zero rather than its result.
ANE relRMS simulates the measured datapath against an exact reference; the
floor measured on silicon at production QKV shape is 2.07e-4, and the bf16 GPU
path this would replace scores 1.66e-3 against the same kind of reference.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", help="block oracle safetensors with ref.* taps")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--golden-rows", type=int, default=64)
    ap.add_argument("--golden-columns", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="power-of-two operand scale applied to the activation")
    ap.add_argument("--checkpoint")
    ap.add_argument("--blocks", default="0,24,49")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--rows", type=int, default=512,
                    help="output channels sampled per weight (statistics only)")
    args = ap.parse_args()

    if args.validate:
        validate()
        if not args.checkpoint:
            return

    if args.golden:
        if not args.checkpoint:
            ap.error("--golden also needs --checkpoint for the weights")
        measure_golden(args.golden, args.checkpoint, args.block,
                       args.golden_rows, args.golden_columns, args.scale)
        return

    if not args.checkpoint:
        ap.error("--checkpoint is required unless --validate is used alone")

    header, offset0 = read_header(args.checkpoint)
    blocks = [int(b) for b in args.blocks.split(",")]

    print("Real weight statistics, %s\n" % args.checkpoint.split("/")[-1])
    print("%-10s %6s %12s %12s %12s %12s"
          % ("projection", "block", "K", "weight RMS", "|w| median", "frac |w|<2^-14"))
    stats = {}
    for name, pattern, K in PROJECTIONS:
        for b in blocks:
            key = pattern.format(b=b)
            if key not in header:
                continue
            w = read_bf16(args.checkpoint, header, offset0, key, max_rows=args.rows)
            a = np.abs(w).astype(np.float64)
            rms = float(np.sqrt(np.mean(a * a)))
            stats.setdefault(name, []).append(rms)
            print("%-10s %6d %12d %12.5g %12.5g %11.2f%%"
                  % (name, b, K, rms, float(np.median(a)),
                     100 * float(np.mean(a < FP16_MIN_NORMAL))))
            del w

    print("\nUnderflow fraction vs activation RMS (the uncaptured variable)")
    print("products below fp16's smallest normal, 6.10e-5, are flushed to zero\n")
    act_scales = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    header_row = "%-10s %12s" % ("projection", "weight RMS")
    for a in act_scales:
        header_row += "%9s" % ("x~%g" % a)
    print(header_row)
    rng = np.random.default_rng(1)
    for name, _, _ in PROJECTIONS:
        if name not in stats:
            continue
        wr = float(np.mean(stats[name]))
        row = "%-10s %12.5g" % (name, wr)
        for a in act_scales:
            row += "%8.1f%%" % (100 * underflow_fraction(wr, a, rng=rng))
        print(row)

    print("""
Reading this table. The spike's fixture sat at 33% underflow and cost 3.17%
relative error; at 64x scale it reached 0.3% underflow and 0.0207% error, which
is the arithmetic floor. Underflow below roughly 1% should be invisible against
that floor.

The activation scale is the term nobody has captured. Post-RMSNorm activations
are unit-RMS before the norm gain and the AdaLN modulation, so the realistic
column is not the leftmost one — but that must be confirmed against a real
capture rather than argued. If it lands somewhere costly, a power-of-two
operand scale is exact in fp16 and moves the whole distribution off the
boundary for one multiply at each end.
""")


if __name__ == "__main__":
    sys.exit(main())
