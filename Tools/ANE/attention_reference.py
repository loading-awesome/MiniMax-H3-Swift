#!/usr/bin/env python3
"""fp32 reference for the fused ANE attention graph, at any sequence length.

`ReferenceError` in attention-spike.mm is a scalar O(H*S^2*D) loop capped at
S=512, which left precision unmeasured at the production sequence — the number
that decides whether the fused graph is usable at all. This computes the same
quantity chunked over query rows, in fp32, in about a second.

The graph applies no 1/sqrt(d) scale, and the spike's own comparison found
`key_axis` — softmax over the key axis, i.e. ordinary SDPA — to be the
orientation it implements. Both are reproduced here so this can be checked
against the scalar reference at S=512 before being trusted at S=15,744.

    python3 Tools/ANE/attention_reference.py /tmp/prefix- --heads 1 \
        --sequence 15744 --dimension 128
"""
import argparse, numpy as np

def load(path, heads, sequence, dimension):
    a = np.fromfile(path, dtype=np.float16)
    want = heads * sequence * dimension
    if a.size != want:
        raise SystemExit(f"{path}: {a.size} elements, expected {want}")
    return a.reshape(heads, sequence, dimension)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--sequence", type=int, required=True)
    p.add_argument("--dimension", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    a = p.parse_args()
    H, S, D = a.heads, a.sequence, a.dimension

    q = load(a.prefix + "q.bin", H, S, D)
    k = load(a.prefix + "k.bin", H, S, D)
    v = load(a.prefix + "v.bin", H, S, D)
    y = load(a.prefix + "y.bin", H, S, D).astype(np.float32)

    err2 = ref2 = 0.0
    worst_row_rel, worst_row = 0.0, -1
    for h in range(H):
        qh = q[h].astype(np.float32)
        kh = k[h].astype(np.float32)
        vh = v[h].astype(np.float32)
        for lo in range(0, S, a.chunk):
            hi = min(lo + a.chunk, S)
            s = qh[lo:hi] @ kh.T                       # [R, S], no 1/sqrt(d)
            s -= s.max(axis=1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(axis=1, keepdims=True)
            ref = s @ vh                                # [R, D]
            d = y[h, lo:hi] - ref
            err2 += float((d * d).sum())
            ref2 += float((ref * ref).sum())
            rel = np.sqrt((d * d).sum(axis=1) / np.maximum((ref * ref).sum(axis=1), 1e-30))
            i = int(rel.argmax())
            if rel[i] > worst_row_rel:
                worst_row_rel, worst_row = float(rel[i]), lo + i

    print(f"key_axis rel_rms = {np.sqrt(err2 / ref2):.6g}")

    # Rule out an orientation flip before concluding the graph is wrong: the
    # spike found `key_axis` at S=512, but a lowering that tiles the score plane
    # could normalise down the query axis instead. Two passes for the column
    # max and sum, then one for the output.
    qerr2 = qref2 = 0.0
    for h in range(H):
        qh, kh, vh = q[h].astype(np.float32), k[h].astype(np.float32), v[h].astype(np.float32)
        cmax = np.full(S, -np.inf, dtype=np.float32)
        for lo in range(0, S, a.chunk):
            hi = min(lo + a.chunk, S)
            np.maximum(cmax, (qh[lo:hi] @ kh.T).max(axis=0), out=cmax)
        csum = np.zeros(S, dtype=np.float32)
        for lo in range(0, S, a.chunk):
            hi = min(lo + a.chunk, S)
            csum += np.exp((qh[lo:hi] @ kh.T) - cmax).sum(axis=0)
        for lo in range(0, S, a.chunk):
            hi = min(lo + a.chunk, S)
            pq = np.exp((qh[lo:hi] @ kh.T) - cmax) / csum
            ref = pq @ vh
            d = y[h, lo:hi] - ref
            qerr2 += float((d * d).sum()); qref2 += float((ref * ref).sum())
    print(f"query_axis rel_rms = {np.sqrt(qerr2 / qref2):.6g}")
    print(f"worst row {worst_row}: rel {worst_row_rel:.6g}")
    print(f"(bf16 GPU path for comparison: 1.66e-3)")

main()
