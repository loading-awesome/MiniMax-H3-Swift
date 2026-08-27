#!/usr/bin/env python3
"""fp32 reference for the scores-only ANE graph.

Checks `matmul(k, transpose(q))` alone, with no softmax in the graph, so a wrong
answer here means the *matmul* fails at attention's N rather than the softmax
normalisation. The output is [H, keys, queries]:

    scores[h, s, t] = dot(k[h, s], q[h, t])
"""
import argparse, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--keys", type=int, required=True)
    p.add_argument("--queries", type=int, required=True)
    p.add_argument("--dimension", type=int, default=128)
    p.add_argument("--chunk", type=int, default=2048)
    a = p.parse_args()
    H, S, T, D = a.heads, a.keys, a.queries, a.dimension

    q = np.fromfile(a.prefix + "q.bin", dtype=np.float16).reshape(H, T, D)
    k = np.fromfile(a.prefix + "k.bin", dtype=np.float16).reshape(H, S, D)
    raw = np.fromfile(a.prefix + "scores.bin", dtype=np.float16)

    # This lowering is documented to transpose the score plane relative to what
    # the MIL types declare, so both readings are tried rather than assumed. A
    # rel_rms near sqrt(2) means uncorrelated — the signature of reading the
    # right numbers in the wrong order.
    def score(arr):
        e = r = 0.0
        for h in range(H):
            qh, kh = q[h].astype(np.float32), k[h].astype(np.float32)
            for lo in range(0, S, a.chunk):
                hi = min(lo + a.chunk, S)
                ref = kh[lo:hi] @ qh.T
                d = arr[h, lo:hi].astype(np.float32) - ref
                e += float((d * d).sum()); r += float((ref * ref).sum())
        return np.sqrt(e / r)

    declared = raw.reshape(H, S, T)
    transposed = raw.reshape(H, T, S).transpose(0, 2, 1)
    a_dec, a_tra = score(declared), score(transposed)
    print(f"as declared [H,{S},{T}] rel_rms = {a_dec:.6g}")
    print(f"as transposed [H,{T},{S}] rel_rms = {a_tra:.6g}")
    got = declared if a_dec <= a_tra else transposed
    print(f"-> taking the {'declared' if a_dec <= a_tra else 'transposed'} reading")

    err2 = ref2 = 0.0
    worst, worst_at = 0.0, (-1, -1)
    for h in range(H):
        qh = q[h].astype(np.float32)
        kh = k[h].astype(np.float32)
        for lo in range(0, S, a.chunk):
            hi = min(lo + a.chunk, S)
            ref = kh[lo:hi] @ qh.T                       # [chunk, T]
            d = got[h, lo:hi].astype(np.float32) - ref
            err2 += float((d * d).sum()); ref2 += float((ref * ref).sum())
            i = int(np.abs(d).argmax())
            if abs(d.flat[i]) > worst:
                worst = float(abs(d.flat[i]))
                worst_at = (lo + i // T, i % T)
    print(f"scores rel_rms = {np.sqrt(err2 / ref2):.6g}")
    print(f"worst abs err {worst:.6g} at key {worst_at[0]}, query {worst_at[1]}")
    print(f"(fp16 round-off alone should land near 1e-3)")

main()
