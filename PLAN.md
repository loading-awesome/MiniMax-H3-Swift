# MiniMax-H3-Swift — productisation plan

Target repo: `git@github.com:loading-awesome/MiniMax-H3-Swift.git` (currently
empty, `main` with no commits). Source of truth for the port today:
`/Volumes/scratch_disk/H3_Swift` — 10,320 lines of Swift across 39 files, of
which roughly **3,900 lines are parity and discovery scaffolding** that should
not ship.

This plan is written to be argued with. Section 9 lists what I think is missing
from the brief.

---

## 0. One thing to fix before anything else

**The port has two DiT partitions and has been using the wrong one for
reference renders.**

`MiniMax-H3-FL2VA_bf16` and `MiniMax-H3-Ref2VA_bf16` are separate 66.28 GB
checkpoints — identical architecture, different weights. FL2VA serves T2VA and
first/last-frame; **Ref2VA serves image / video / audio references**. Every
ref2va render on 2026-08-05, including `ref_mixed` and `ref_single`, ran against
the FL2VA checkpoint because the runner hard-coded one path.

Those renders still prove the *plumbing* — payload assembly, presentation
ordering, packed layout, the sampler loop. They do not prove the outputs are
what Ref2VA would produce. And the identity transfer that looked so convincing
may have come through the Qwen vision tokens (`<Picture 1>` in the text
conditioning) rather than the DiT reference blocks at all, since FL2VA was never
trained to consume them.

Product consequence: **partition selection is a function of the requested mode,
not a CLI flag the caller can get wrong.** It belongs in the resolver, with a
hard error when a mode and a loaded partition disagree.

Cheap experiment worth running first: re-render `ref_single` against Ref2VA and
diff against the FL2VA version. If identity transfer is unchanged, the vision
tokens are doing the work and the reference blocks are close to inert — which is
a finding that changes what we tell users about reference strength.

---

## 1. Checkpoint catalogue

Everything below is on disk and was read out of the safetensors headers, not
from a model card.

### DiT — four variants per partition

| variant | size | format | parity status |
|---|---|---|---|
| `bf16` | **66.28 GB** | BF16 throughout | the gated reference; 225 taps |
| `int8_convrot` | **34.04 GB** | I8 weights + F32 `weight_scale [out,1]` + U8 `comfy_quant` descriptor | ungated |
| `pruned_bf16` | **41.40 GB** | BF16, **AdaLN factorised** | ungated, *and not an approximation-free variant* |
| `pruned_int8_convrot` | **22.14 GB** | both of the above | ungated |

Two things the headers make clear that the filenames do not:

* **`int8_convrot` is per-output-channel symmetric int8.** Dequantisation is
  `w.astype(.bfloat16) * scale`, which is trivial — but doing that at load
  returns you to 66 GB resident and saves nothing. Real savings need int8 kept
  resident with a quantised matmul at runtime. MLX has `quantized_matmul` with
  a *group-wise* layout, so this is a conversion problem, not a free win.
  **Estimate this properly before promising 64 GB Macs.**
* **`pruned` is not pruning.** `blocks.N.adaln_proj.linear.weight` is
  `[96768, 64]` instead of `[96768, 2688]`, with metadata
  `adaln_curve_grid=1001`, `adaln_curve_rank`, `adaln_curve_centered`. AdaLN has
  been replaced by a low-rank curve approximation — which accounts for exactly
  the 24.9 GB difference. It **changes numerics by construction** and must be
  labelled an approximation in the UI, never silently substituted.

### Text encoder

| variant | size | format |
|---|---|---|
| `Qwen3-VL-32B-Instruct-layer50_bf16` | **51.51 GB** | BF16 |
| `…layer50_quanto_bf16_int8` | **26.72 GB** | quanto: I8 `_data` + BF16 `_scale`, plus input/output scales |

### VAEs

| | size |
|---|---|
| video, fp16 | 5.21 GB |
| audio, **fp32** | 0.61 GB |

The audio VAE is the only component published at full precision. Contract 5
says do not downcast it, and that stands.

### Vendor layouts are not interchangeable

Contract 9: Comfy-Org stores `attn.qkv_proj.weight` blocked `Q|K|V`,
DeepBeepMeep stores it per-head interleaved. Same shape, same dtype, same mean
and std to 8 decimals, **cos 0.029** if you load one as the other. Detection is
by `__metadata__` and a checkpoint whose vendor cannot be identified must be
rejected, not guessed at. This is already implemented in
`H3Core.CheckpointInventory` and must survive the migration intact.

---

## 2. Memory model, and who can actually run this

Weights are only half the story. Measured on the current tree at 864x480x124,
20 steps: MLX-allocated peak **53.1 GB**, and that figure *excludes the
checkpoints entirely* because `loadArrays` memory-maps them. Resident footprint
during the same render was ~171 GB. Both numbers are honest; they measure
different things, and a productised memory planner has to model both.

Phase-peak, assuming each stage is loaded and released in turn (which the
current `generate` now does):

| phase | resident | notes |
|---|---|---|
| text encode | TE weights + ~2 GB | 51.5 or 26.7 |
| VAE encode | 5.8 GB + small | |
| **sampling** | **DiT weights + activations** | the binding constraint |
| decode | 5.8 GB + frame buffers | |

Activations at 864x480x124 do **not** shrink with quantisation — they scale with
packed sequence length, and attention is quadratic in it. That is the single
most important fact for low-memory targets: **a smaller checkpoint does not buy
a bigger render.**

Indicative sampling-phase peaks (weights + activations, to be measured, not
trusted from this table):

| config | weights | + activations @864x480x124 | plausible floor |
|---|---|---|---|
| bf16 | 66.3 | ~91 GB | 128 GB |
| int8 (if resident-quantised) | 34.0 | ~59 GB | 96 GB, tight on 64 |
| pruned int8 | 22.1 | ~47 GB | 64 GB, tight on 48 |

Against shipping Mac configurations:

| RAM | verdict |
|---|---|
| 8 / 16 / 24 GB | **cannot run.** Refuse at startup with the reason and the number. |
| 32 / 36 GB | not at the verified shape; possibly a much shorter/smaller render |
| 48 GB | pruned int8, reduced resolution or duration |
| 64 GB | pruned int8 comfortably; int8 tight |
| 96 GB | int8 comfortable; bf16 tight |
| 128 GB+ | bf16, the gated path |
| 192–512 GB | bf16 with headroom for larger shapes |

**Say this plainly in the README.** The model cannot run on the majority of Macs
sold, and a product that discovers this at minute 20 of a render is worse than
one that says so in the first second.

---

## 3. Module structure

Nine targets, each with a stated reason to exist. Swift Package Manager, one
public umbrella.

```
MiniMaxH3/                     umbrella: the public API, and nothing else
├─ H3Foundation/               config types, errors, geometry, lattice math,
│                              safetensors I/O, logging. No MLX in the API.
├─ H3Hardware/                 chip + memory detection, memory planner,
│                              thermal/pressure watch, single-instance guard
├─ H3Catalog/                  checkpoint discovery, vendor/partition/precision
│                              identification, integrity, resolution from config
├─ H3Attention/                the attention backend protocol + built-ins
├─ H3Modules/                  DiT blocks, VAEs, vision tower, text encoder
├─ H3Pipeline/                 conditioning assembly, packed layout, sampler,
│                              decode, mux. Phase-scoped residency lives here.
├─ H3Recipes/                  recipe registry + capability-aware validation
├─ H3Conformance/             a *small* retained golden suite (see §6)
└─ h3 (executable)             thin CLI over the public API
```

Design rules:

* **No `preconditionFailure` on any path a caller can reach.** The layout code
  currently crashes on a bad keyframe index. In a library that takes the host
  application down. Every one becomes a typed `throws`.
* **MLX types do not appear in the public API.** Callers pass URLs, structs and
  enums; they get back files, progress and typed errors. This is what lets the
  attention backend and the compute backend change without a breaking release.
* **Every module gets a header comment stating what it owns and what it must not
  know about.** The existing code already documents *why* heavily — that habit
  is the most valuable thing in the repo and should be preserved verbatim where
  the code moves unchanged.

---

## 4. What ships and what is stripped

### Strip (discovery scaffolding, ~3,900 lines)

`h3-parity`'s check subcommands — `Fl2VaCheck`, `Ref2VaCheck`,
`TemporalEncodeCheck`, `VaeEncodeCheck`, `VisionCheck`, `ImagePromptCheck`,
`EncodeCheck`, `DecodeCheck`, `OracleBlock`, `H3Parity`'s `Verify`/`Inspect`/
`Oracle`/`Run` — plus `GoldenBundle`, `BenchGemm`, and the whole `parity/`
directory of capture scripts and RunPod tooling.

### Keep, but move out of the product target

`FaceCheck`, `speech_check.py`, `anchor_check.py`, `audio_match.py` become a
`Tools/` directory. They are genuinely useful — they are the only oracles that
work without a golden — but they are QA, not product.

### Keep in the product

Everything in `H3Core`/`H3Model` except `GoldenBundle`, plus `MediaLoad`,
`Generate`'s pipeline (decomposed), `RenderPolicy`, `Recipe`,
`CheckpointInventory`.

### Keep the *knowledge*

`docs/FRAGILE_CONTRACTS.md` is 36 entries of "break this and parity goes with
it", each carrying its evidence. **This is the most valuable artefact in the
repo** and must migrate whole. Most entries should additionally become an
assertion or a test in `H3Conformance`, so the contract is enforced rather than
merely written down.

---

## 5. Hardware detection and automatic configuration

At startup, in order:

1. **Identify the machine** — `hw.model`, `sysctl machdep.cpu.brand_string`,
   `hw.memsize`, GPU core count via IOKit, and whether it is a laptop chassis
   (thermal ceiling differs from a Studio for a 20-minute sustained load).
2. **Measure, do not assume, free memory** — `hw.memsize` minus current
   footprint. A machine with 64 GB and Chrome open is not a 64 GB machine.
3. **Resolve a plan** — the largest checkpoint set that fits the sampling-phase
   peak for the requested shape, with a stated margin.
4. **Report the decision and the reason**, always. `h3 doctor` prints the whole
   resolution: what was detected, what was chosen, what was rejected and why.

The plan is a value the caller can inspect and override, not hidden state.
Overriding into a configuration that will not fit requires the same
`--allow-suboptimal` gesture the render policy already uses, and logs every
reason.

**Runtime pressure matters too, not just startup.** Today's SIGKILL at 196.2 GB
happened on a 275 GB machine because a second renderer was running. A
single-instance advisory lock and a memory-pressure watch (`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`)
that can abort cleanly with a diagnostic beats being killed with none.

---

## 6. Configuration files

TOML or JSON at `~/.config/minimax-h3/config.toml`, overridable by
`--config`, with environment variables above that and CLI flags above those.

```toml
[checkpoints]
root = "/Volumes/scratch_disk/models/MiniMax-H3"

[checkpoints.dit.fl2va]
bf16                = "MiniMax-H3-FL2VA_bf16.safetensors"
int8                = "MiniMax-H3-FL2VA_int8_convrot.safetensors"
pruned_bf16         = "MiniMax-H3-FL2VA-pruned_bf16.safetensors"

[checkpoints.dit.ref2va]
bf16                = "MiniMax-H3-Ref2VA_bf16.safetensors"
# …

[checkpoints.text_encoder]
bf16 = "…layer50_bf16.safetensors"
int8 = "…layer50_quanto_bf16_int8.safetensors"

[checkpoints.vae]
video = "MiniMax-H3-video_vae_fp16.safetensors"
audio = "MiniMax-H3-audio_vae_fp32.safetensors"

[policy]
allow_approximate_weights = false   # gates the `pruned` variants explicitly
max_render_seconds        = 10800

[attention]
backend = "auto"                    # auto | sdpa | sol
```

`h3 config validate` resolves every path, identifies vendor/partition/precision
from the header, checks sizes, and prints what is missing. Nothing is silently
absent.

---

## 7. Attention backend seam

```swift
public protocol H3AttentionBackend: Sendable {
    static var identifier: String { get }
    static func isAvailable(on: H3Hardware.Machine) -> Bool
    /// [S, heads, headDim] in, [S, heads, headDim] out. No batch axis —
    /// H3 is batch-size-1 by construction (reference model.py:509).
    func attend(q: MLXArray, k: MLXArray, v: MLXArray,
                scale: Float, mask: H3AttentionMask?) -> MLXArray
}
```

Selection is `config → hardware capability → fallback to SDPA`, resolved once at
model build and logged. `DiTBlock` holds a backend instance rather than calling
a free function.

**I need the specifics on sol-attn before designing further.** I do not want to
guess at its API, numerics or memory profile and build a seam that turns out to
be the wrong shape. Two questions: is it a drop-in SDPA replacement at the same
tensor layout, and does it change numerics enough to need its own conformance
run? If it does, the backend protocol needs a declared tolerance class, and the
conformance suite needs to run per-backend.

---

## 8. Recipe system, made capability-aware

The current registry knows resolution and aspect ratio, and tiers recipes as
`.baseRender` or `.upscaleTarget`. Productised, a recipe resolves against three
things it does not currently see:

1. **Mode** — which partition, which conditioning kinds are legal. Anchors and
   references are mutually exclusive in one payload; the recipe should encode
   that rather than leaving it to a validation function.
2. **Hardware** — a recipe that cannot fit the machine is rejected at selection
   with the measured numbers, not at minute 20.
3. **Feature availability** — 2K stays refused while the in-context upscaler is
   unreleased. `h3 recipes` lists everything with a status column, so
   "unavailable" is visible rather than discovered.

Every refusal keeps the current house style: the rule, the measurement behind
it, and the remedy. That style is why the render policy works, and it should be
the template for every new error type.

---

## 9. What I think the brief is missing

Ordered by how much I think each will cost if it is not decided now.

1. **The Ref2VA partition issue in §0.** It is a correctness bug in the current
   renders, not a productisation question, and it changes what we can claim.

2. **"Strip all the parity process" is half right.** Removing the CUDA capture
   tooling, RunPod scripts and golden bundles is correct — that was discovery.
   Removing *all* numerical checks leaves no defence against silent regression,
   and this codebase's failure mode is specifically silent: a wrong layout, a
   dropped label, a transposed qkv all keep every shape valid. I would keep a
   compact `H3Conformance` target — a handful of small recorded tensors
   (kilobytes, not gigabytes) pinning the packed layout, the tokenizer, the
   flow schedule and one block forward. Cheap, no GPU, and it is what makes the
   36 contracts enforceable instead of aspirational.

3. **What is this, exactly — a CLI, a library, or a service?** The module plan
   assumes library-first with a thin CLI. If a long-running service is intended,
   batch-size-1 means a queue and a single loaded model, and that changes the
   residency design substantially.

4. **Progress and cancellation.** A 20-minute call with no progress callback and
   no way to cancel is not shippable. This needs to be in the API from the
   start; retrofitting it through the sampler loop is invasive.

5. **Model acquisition.** 100+ GB of checkpoints. Auto-download with resume and
   hash verification, or documented manual placement? WanGP does the former.
   Either way there needs to be a disk-space preflight.

6. **Determinism story.** Seeds are reproducible *within* this port but cannot
   reproduce the CUDA reference — MLX's PRNG will never emit torch's bytes, which
   is why conditioning noise was fed in as a recorded tensor during parity work.
   For a product this is fine, but it must be stated: same seed, same machine,
   same version → same output; nothing stronger.

7. **Streaming decode.** Decoding all 124 frames before writing is a memory
   spike and gives no preview. Progressive decode caps the peak and enables a
   live preview, which is a visible product feature and a memory win at once.

8. **Thermals on laptops.** The estimator is calibrated on a Mac Studio. A
   MacBook Pro will throttle over a 20-minute sustained load and the estimate
   will be optimistic. Either measure a throttle factor per chassis or state the
   estimate is for sustained-cooling machines.

9. **Public API stability.** DocC, SemVer, and a decision about what is public
   vs `@_spi`. Once `H3AttentionBackend` is public it is a compatibility
   surface.

10. **Where the quality bar lives.** The stress set and the conditioning matrix
    are currently ad-hoc scripts. If image quality regressions matter, one
    curated prompt set plus the four no-golden oracles should run on a release
    checklist, with the outputs archived per version.

---

## 10. Sequencing

| phase | work | gate |
|---|---|---|
| **0** | Ref2VA partition experiment; decide on conformance scope | a decision, written down |
| **1** | Repo skeleton, SPM targets, CI, DocC, licence, README with the honest memory table | `swift build` + `swift test` green |
| **2** | `H3Foundation` + `H3Catalog` + config files + `h3 doctor` | resolves every checkpoint on this machine and names each one's vendor/partition/precision |
| **3** | `H3Modules` + `H3Pipeline` migrated, errors made throwing, phase-scoped residency | a bf16 render byte-comparable to the current tree's |
| **4** | `H3Hardware` + memory planner + recipe capability gating | correct plan on 128 / 64 / 32 GB, verified by simulating available memory |
| **5** | `H3Attention` seam + SDPA backend; sol-attn once specified | conformance passes per backend |
| **6** | int8 resident quantised matmul | measured peak and a quality comparison against bf16 |
| **7** | Progress/cancellation, streaming decode, model acquisition | — |

Phase 6 is the one that decides whether 64 GB Macs are supportable, and it is
the one with the most uncertainty. Worth prototyping early enough that the
answer can change the marketing rather than contradict it.
