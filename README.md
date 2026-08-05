# MiniMax-H3 for Apple Silicon

A native Swift/MLX implementation of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
the joint video-and-audio diffusion transformer — 50 blocks, hidden 5376,
56 heads, one packed sequence carrying text, audio and video together.

Ported against the CUDA reference with numerical parity **established rather
than hoped for**: 225 gating taps inside CUDA-measured equivalence classes at
production shape, plus 36 documented contracts in
[FRAGILE_CONTRACTS.md](FRAGILE_CONTRACTS.md), each carrying the evidence that
established it.

---

## Read this before anything else: it will not run on most Macs

The model is 124 GB of weights in bf16 and the activations at the verified
render shape are tens of gigabytes more. **Activations do not shrink when the
weights do** — they scale with packed sequence length and attention is quadratic
in it — so a smaller checkpoint alone will not rescue a small machine.

| unified memory | verdict |
|---|---|
| 8 / 16 / 24 GB | **cannot run.** Refused at startup, with the numbers. |
| 32 / 36 GB | not at the verified shape |
| 48 GB | approximate weights, reduced resolution or duration |
| 64 GB | approximate weights comfortably |
| 96 GB | tight for the full bf16 configuration |
| **128 GB+** | bf16, the configuration parity was measured on |

`h3 doctor` tells you which of these you are, what it would select, and what it
rejected, in about a second. It reads safetensors headers, not bodies.

```
$ h3 doctor
machine
  Apple M3 Ultra (Mac15,14), 275 GB unified memory, 28 cores
  132 GB available right now (free + inactive + speculative)

memory plan at 15750 packed tokens
  bf16         peak   98.7 GB   FITS, +17 GB headroom
  int8         peak   98.7 GB   FITS, +17 GB headroom   [dequantised at load — saves disk, not memory]
  pruned_bf16  peak   73.8 GB   refused (approximate weights)
```

## Checkpoints

Two DiT partitions, and **which one you need is decided by what you are
rendering, not by a flag**:

| partition | serves |
|---|---|
| **FL2VA** | text-to-video, first-frame, last-frame |
| **Ref2VA** | image / video / audio references |

They share an architecture and differ in weights, so loading the wrong one
renders successfully, without error, from weights that were never trained for
the job. The catalog refuses it — including when a configuration files an FL2VA
file under the `ref2va` key.

Four variants ship per partition. Two of the names are misleading and the
catalog reports what the file *is*, read from its own header:

| variant | on disk | resident | note |
|---|---|---|---|
| `bf16` | 66.3 GB | 66.3 GB | the configuration parity was measured on |
| `int8` | 34.0 GB | **66.3 GB** | per-channel int8, dequantised at load — saves disk, not memory |
| `pruned_bf16` | 41.4 GB | 41.4 GB | **not pruning**: AdaLN is a rank-64 curve approximation |
| `pruned_int8` | 22.1 GB | 41.4 GB | both of the above |

The `pruned` variants change numerics by construction and are refused unless
`policy.allow_approximate_weights` is set. Do not compare their output to the
gated configuration.

## Configuration

```bash
h3 config init          # writes ~/.config/minimax-h3/config.json
h3 config validate      # resolves every path and identifies what is there
h3 doctor               # the whole startup decision, printed
```

Precedence is defaults, then the config file, then environment, then CLI flags.

## Architecture

Dependencies point downward only. The two lowest layers do not link MLX, which
means the geometry, the 17k+5 frame lattice, checkpoint identification, the
flow schedule and the memory planner are all testable in seconds with no GPU
and no 66 GB download — and those are precisely the places where an error stays
silent.

| target | owns | MLX |
|---|---|---|
| `H3Foundation` | errors, geometry, config, safetensors | no |
| `H3Hardware` | machine detection, the memory planner | no |
| `H3Catalog` | checkpoint discovery and identification | no |
| `H3Recipes` | capability-aware recipe resolution | no |
| `H3Attention` | the attention backend seam | yes |
| `H3Modules` | DiT, VAEs, vision tower, text encoder | yes |
| `H3Pipeline` | conditioning, layout, sampler, decode, mux | yes |
| `H3Conformance` | the retained numerical checks | yes |
| `MiniMaxH3` | the public API | no |

MLX types do not appear in the public API. That is what lets the attention
backend, the quantisation scheme and the compute backend change without a
breaking release.

## Status

Ported and gated: the DiT stack, both VAEs, the vision tower, the tokenizer
(bit-identical), the packed layout for every conditioning kind, and the
conditioning presentations for keyframes and references. Rendered end to end:
text-to-video, first/last-frame anchors, and every reference kind.

Not implemented: in-context 2K upscaling (the upscaler is unreleased), and
resident quantised matmul — which is the item that decides whether 64 GB
machines are supportable.

## Licence

See [LICENSE](LICENSE). The model weights are licensed separately by MiniMax.
