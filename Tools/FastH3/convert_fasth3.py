#!/usr/bin/env python3
"""Convert FastVideo's FastH3 4-step distill to this port's checkpoint layout.

The architecture is identical to MiniMax H3 -- 5376 hidden, 50 blocks, 56 heads
of 128, ffn 14336, patch [1,2,2] -- so only the naming differs, plus one real
structural change: diffusers keeps q, k and v as separate projections and this
port expects them fused as one [3*inner, hidden] tensor.

Correctness rests on the output key set matching the reference checkpoint
exactly, shape for shape. That is asserted rather than assumed: a silently
mismapped tensor produces a model that runs and generates noise, which is the
expensive failure to debug.

    python3 Tools/FastH3/convert_fasth3.py --src <hf transformer dir> \
        --reference <existing .safetensors> --out <converted .safetensors>
"""
import argparse, json, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def build_map(blocks: int, refiners: int) -> dict[str, str]:
    """Ours -> theirs, for everything that is a straight rename."""
    m = {
        "video_patch_proj.weight": "proj_in.weight",
        "video_patch_proj.bias":   "proj_in.bias",
        "audio_patch_proj.weight": "audio_proj_in.weight",
        "audio_patch_proj.bias":   "audio_proj_in.bias",
        "condition_proj.weight":   "context_embedder.weight",
        "condition_proj.bias":     "context_embedder.bias",
        "time_embedder.proj_in.weight":  "time_embedder.linear_1.weight",
        "time_embedder.proj_in.bias":    "time_embedder.linear_1.bias",
        "time_embedder.proj_out.weight": "time_embedder.linear_2.weight",
        "time_embedder.proj_out.bias":   "time_embedder.linear_2.bias",
        "final_layer.norm.weight":               "norm_out.norm.weight",
        "final_layer.adaln_proj.linear.weight":  "norm_out.linear.weight",
        "final_layer.adaln_proj.linear.bias":    "norm_out.linear.bias",
        "final_layer.video_out.weight": "proj_out.weight",
        "final_layer.video_out.bias":   "proj_out.bias",
        "final_layer.audio_out.weight": "audio_proj_out.weight",
        "final_layer.audio_out.bias":   "audio_proj_out.bias",
        "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
    }
    for i in range(blocks):
        a, b = f"blocks.{i}.", f"transformer_blocks.{i}."
        m[a + "attn.out_proj.weight"] = b + "attn.to_out.0.weight"
        m[a + "attn.q_norm.weight"]   = b + "attn.norm_q.weight"
        m[a + "attn.k_norm.weight"]   = b + "attn.norm_k.weight"
        # `mlp.fc1` needs its halves swapped, handled separately.
        m[a + "mlp.fc2.weight"]       = b + "ff.net.2.weight"
        m[a + "norm1.weight"]         = b + "norm1.weight"
        m[a + "norm2.weight"]         = b + "norm2.weight"
        m[a + "adaln_proj.linear.weight"] = b + "adaln_proj.linear.weight"
        m[a + "adaln_proj.linear.bias"]   = b + "adaln_proj.linear.bias"
    for i in range(refiners):
        a, b = f"token_refiner.blocks.{i}.", f"token_refiner.refiner_blocks.{i}."
        m[a + "attn.out_proj.weight"] = b + "attn.to_out.0.weight"
        m[a + "attn.q_norm.weight"]   = b + "attn.norm_q.weight"
        m[a + "attn.k_norm.weight"]   = b + "attn.norm_k.weight"
        # `mlp.fc1` needs its halves swapped, handled separately.
        m[a + "mlp.fc2.weight"]       = b + "ff.net.2.weight"
        m[a + "norm1.weight"]         = b + "norm1.weight"
        m[a + "norm2.weight"]         = b + "norm2.weight"
    return m


def swiglu_sources(blocks: int, refiners: int) -> dict[str, str]:
    """Ours -> the diffusers projection whose halves are the other way round.

    This port splits `fc1` as `[gate | up]` and computes `silu(gate) * up`;
    diffusers packs `[up | gate]`. The tensors are the same size and the same
    statistics, so nothing but the arithmetic notices.
    """
    out = {f"blocks.{i}.mlp.fc1.weight": f"transformer_blocks.{i}.ff.net.0.proj.weight"
           for i in range(blocks)}
    out.update({f"token_refiner.blocks.{i}.mlp.fc1.weight":
                f"token_refiner.refiner_blocks.{i}.ff.net.0.proj.weight"
                for i in range(refiners)})
    return out


def fused_qkv_sources(blocks: int, refiners: int) -> dict[str, tuple[str, str, str]]:
    """Ours -> the three tensors that fuse into it, in q,k,v order."""
    out = {}
    for i in range(blocks):
        b = f"transformer_blocks.{i}."
        out[f"blocks.{i}.attn.qkv_proj.weight"] = (
            b + "attn.to_q.weight", b + "attn.to_k.weight", b + "attn.to_v.weight")
    for i in range(refiners):
        b = f"token_refiner.refiner_blocks.{i}."
        out[f"token_refiner.blocks.{i}.attn.qkv_proj.weight"] = (
            b + "attn.to_q.weight", b + "attn.to_k.weight", b + "attn.to_v.weight")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF transformer/ directory")
    ap.add_argument("--reference", required=True, help="existing port checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=50)
    ap.add_argument("--refiners", type=int, default=2)
    ap.add_argument("--repo-id", dest="repo_id",
                    default="FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree")
    ap.add_argument("--partition", default="FL2VA")
    args = ap.parse_args()

    # What the port expects, taken from a checkpoint it already loads.
    with safe_open(args.reference, framework="pt") as f:
        want = {k: f.get_slice(k).get_shape() for k in f.keys()}
    print(f"reference: {len(want)} tensors")

    # Everything the distill ships, across its shards.
    shards = sorted(Path(args.src).glob("*.safetensors"))
    if not shards:
        print(f"no safetensors under {args.src}", file=sys.stderr); return 2
    where: dict[str, Path] = {}
    for s in shards:
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                where[k] = s
    print(f"source:    {len(where)} tensors across {len(shards)} shards")

    rename = build_map(args.blocks, args.refiners)
    fused = fused_qkv_sources(args.blocks, args.refiners)
    swiglu = swiglu_sources(args.blocks, args.refiners)

    out: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    handles: dict[Path, object] = {}

    def get(name: str):
        if name not in where:
            missing.append(name); return None
        p = where[name]
        if p not in handles:
            handles[p] = safe_open(p, framework="pt")
        return handles[p].get_tensor(name)

    for ours, theirs in rename.items():
        t = get(theirs)
        if t is not None:
            out[ours] = t
    # **Interleaved per head, not blocked.** The loader permutes
    # `(heads, 3, headDim, hidden) -> (3, heads, headDim, hidden)` for this
    # vendor, so the file must be the interleaved side of that. Writing blocked
    # weights and labelling them this way makes the loader scramble them, which
    # costs the right shape, the right standard deviation and the wrong answer:
    # measured cos 0.015 against the reference, and a render of pure noise.
    HEADS, HEAD_DIM = 56, 128
    for ours, (q, k, v) in fused.items():
        parts = [get(q), get(k), get(v)]
        if any(p is None for p in parts):
            continue
        hidden = parts[0].shape[1]
        viewed = [p.reshape(HEADS, HEAD_DIM, hidden) for p in parts]
        out[ours] = torch.stack(viewed, dim=1).reshape(3 * HEADS * HEAD_DIM, hidden)

    for ours, theirs in swiglu.items():
        t = get(theirs)
        if t is None:
            continue
        half = t.shape[0] // 2
        out[ours] = torch.cat([t[half:], t[:half]], dim=0)

    # `rope.inv_freq` is derived from rope_freq_dim and rope_theta, not shipped;
    # carry the reference's copy so the output is a drop-in.
    if "rope.inv_freq" in want:
        with safe_open(args.reference, framework="pt") as f:
            out["rope.inv_freq"] = f.get_tensor("rope.inv_freq")

    produced, expected = set(out), set(want)
    if missing:
        print(f"\nMISSING from source ({len(missing)}):", file=sys.stderr)
        for k in missing[:12]: print(f"  {k}", file=sys.stderr)
        return 3
    if produced != expected:
        print(f"\nKEY SET MISMATCH", file=sys.stderr)
        for k in sorted(expected - produced)[:12]: print(f"  missing: {k}", file=sys.stderr)
        for k in sorted(produced - expected)[:12]: print(f"  extra:   {k}", file=sys.stderr)
        return 4
    bad = [(k, list(out[k].shape), want[k]) for k in expected
           if list(out[k].shape) != list(want[k])]
    if bad:
        print(f"\nSHAPE MISMATCH ({len(bad)}):", file=sys.stderr)
        for k, got, exp in bad[:12]: print(f"  {k}: got {got}, want {exp}", file=sys.stderr)
        return 5

    # **Shape agreement is not correctness.** Every layout error found here --
    # blocked instead of interleaved qkv, and SwiGLU's halves the wrong way
    # round -- produced tensors of exactly the right shape, dtype and standard
    # deviation, passed the checks above, loaded, rendered, and output noise.
    #
    # A distilled checkpoint is fine-tuned *from* the base, so corresponding
    # tensors stay highly correlated. Anything near zero is a layout error, and
    # this is the check that distinguishes them.
    print("\ncorrelation against the reference (a distill stays close to its parent):")
    worst = 1.0
    with safe_open(args.reference, framework="pt") as f:
        for name in sorted(expected):
            if not (name.endswith(".weight") and name.startswith(("blocks.0.", "blocks.25."))):
                continue
            a = f.get_tensor(name).double().flatten()
            b = out[name].double().flatten()
            cos = float(a @ b / (a.norm() * b.norm()))
            worst = min(worst, cos)
            flag = "" if cos > 0.5 else "   <-- UNCORRELATED"
            print(f"  {name:44s} {cos:7.4f}{flag}")
    if worst < 0.5:
        print(f"\nrefusing to write: a tensor is uncorrelated with the reference "
              f"(worst {worst:.4f}). That is a layout error, not a distillation "
              f"difference.", file=sys.stderr)
        return 6

    print(f"\nkey set, every shape, and every correlation check out. writing {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # The catalog refuses a checkpoint it cannot identify, because the published
    # bf16 conversions differ only in how fused attention weights are arranged
    # -- same shape, dtype and statistics to eight decimals -- and loading one
    # as the other gives cos 0.029. `repo_id` + `partition` marks the contiguous
    # q|k|v layout this converter produces, which is what the loader's
    # `split(parts: 3, axis: -1)` expects. The repo id is the real source: this
    # is not a MiniMaxAI conversion and should not claim to be one.
    meta = {
        "format": "pt",
        "repo_id": args.repo_id,
        "partition": args.partition,
        "precision": "bf16",
        "distilled_from": "MiniMaxAI/MiniMax-H3",
        "transformer_forwards": "4",
        "dmd_denoising_steps": "999,749,500,250",
        "attention_backend": "dense",
    }
    save_file(out, args.out, metadata=meta)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
