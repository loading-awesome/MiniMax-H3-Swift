# h3 — MiniMax-H3 on Apple silicon

Type a sentence, get a video **with the sound already in it**. No GPU rental, no
Python environment, no node graph. One command, on your Mac.

```bash
h3 render --prompt "a red kite over a beach at sunset" --out kite.mp4
```

[![A drag race](demo-media/race.jpg)](demo-media/race.mp4)

<sub>**A drag race.** 864×480, 7 s. Two speakers — the dialogue, both engines and
the music were generated *together with the picture*, nothing was dubbed on
afterwards. Click any still for the clip; GitHub will not play video inline.</sub>

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is an open-weights
video model that is unusual in one specific way: **it generates the picture and
the soundtrack in a single pass.** It moves a character's lips and produces the
voice saying those words at the same time, from the same prompt. `h3` is a native
Swift and [MLX](https://github.com/ml-explore/mlx) port of it, with no Python
anywhere in it.

---

## Will it run on my Mac?

**Probably not**, and I would rather say so before you download 124 GB. The disk
usually stops people first.

| | you need |
|---|---|
| **memory** | 96 GB unified, realistically 128 GB |
| **disk** | 124 GB for text-to-video, 190 GB for everything |
| **chip** | any Apple silicon (M1 or later) |
| **macOS** | 14 or later |

A MacBook Air cannot *store* the model, never mind run it. 48–64 GB Macs are
**not yet** supported — two specific pieces of work would get there, both listed
under [Status](#status). 96 GB and up: yes.

`h3 doctor` answers this for your actual machine in about a second, reading file
headers rather than file bodies. It ends with `no problems found` when you are
ready.

---

## 1. Get `h3`

**Download the installer.** No compiler, no Xcode, no Terminal:

### [⬇ Download the latest release](https://github.com/loading-awesome/MiniMax-H3-Swift/releases/latest)

Take the **`.pkg`** and double-click it. Then:

```bash
h3 doctor
```

<details>
<summary>Or the <code>.zip</code>, if you would rather not run an installer</summary>

Unpack it and keep `h3` and `mlx.metallib` **in the same folder** — `h3` looks
for its GPU kernels beside itself and will not start without them.

```bash
chmod +x h3
./h3 doctor
```

To put it on your path, move **both** files together:

```bash
sudo mkdir -p /usr/local/lib/h3 && sudo cp h3 mlx.metallib /usr/local/lib/h3/
sudo ln -sf /usr/local/lib/h3/h3 /usr/local/bin/h3
```

The symlink is fine; `h3` resolves symlinks before looking for the kernels.
</details>

<details>
<summary><b>Or build it from source</b></summary>

Needs Xcode 16 or later.

```bash
git clone https://github.com/loading-awesome/MiniMax-H3-Swift.git
cd MiniMax-H3-Swift
swift build -c release
./Scripts/bootstrap-metal.sh
.build/release/h3 doctor
```

`bootstrap-metal.sh` is not optional. SwiftPM does not build MLX's Metal kernels,
so without it the first render dies with an unhelpful C++ error naming no file.
See [CONTRIBUTING.md](CONTRIBUTING.md).
</details>

> **If macOS says the file is damaged or from an unidentified developer**, you
> have an unsigned build. Clear the quarantine flag once:
> ```bash
> xattr -dr com.apple.quarantine h3
> ```

## 2. Get the model files

The weights are **not** in this repository and are licensed separately by
MiniMax. Take the single-file conversions published for
[WanGP](https://github.com/deepbeepmeep/Wan2GP) on Hugging Face:

| file | size | what it does |
|---|---|---|
| `MiniMax-H3-FL2VA_bf16.safetensors` | 66.3 GB | text-to-video, and frame anchors |
| `MiniMax-H3-Ref2VA_bf16.safetensors` | 66.3 GB | image, video and audio references |
| `Qwen3-VL-32B-Instruct-layer50_bf16.safetensors` | 51.5 GB | reads your prompt |
| `MiniMax-H3-video_vae_fp16.safetensors` | 5.2 GB | turns latents into pixels |
| `MiniMax-H3-audio_vae_fp32.safetensors` | 0.6 GB | turns latents into sound |

Plus a tokenizer folder holding `vocab.json`, `merges.txt` and
`tokenizer_config.json`.

**You do not need both of the big ones to start.** `FL2VA` alone gives you
text-to-video and frame anchors, which is most of what people want — 124 GB
instead of 190 GB. Put them anywhere; `h3` never moves or copies them.

## 3. Point `h3` at them

```bash
h3 config init
```

That writes `~/.config/minimax-h3/config.json` with the filenames already filled
in. Set two things — the folder your models are in, and the tokenizer:

```json
{
  "checkpoints": {
    "root": "/Users/you/models/MiniMax-H3",
    "tokenizer": "/Users/you/models/MiniMax-H3/tokenizer"
  }
}
```

Then `h3 doctor` again. Every file is identified **from its own header, not its
name**, so one that has been renamed, truncated or converted differently is
reported as what it actually is. When it says `no problems found`, you are ready.

## 4. Render something

```bash
h3 render --prompt "a red kite over a beach at sunset" --out kite.mp4
```

```
  making    text to video and audio
  size      1344 x 768, 5 s at 24 fps (124 frames)
  steps     20
  quality   balanced — an approximation, cache threshold 0.1
  model     MiniMax-H3-FL2VA_bf16.safetensors
  estimate  about 1h 16m of sampling

[1/4] reading the prompt
[2/4] sampling  ████████··············   8/20   61.2 s/step   12m 14s left
```

The countdown is measured from **your** render rather than read from a table, so
it settles after a couple of steps and then tracks reality. The one-line estimate
above it does not account for the cache, so on the default profile it reads high
until the countdown takes over.

```bash
h3 render --prompt "..." --out out.mp4 --dry-run          # what would this cost? (1 second)
h3 render --prompt "..." --out out.mp4 --quality faithful  # nothing approximated, ~2x slower
```

---

## What you can make

| you want | how |
|---|---|
| video from a description | just `--prompt` |
| a character who speaks | put the words in the prompt |
| start from your photo | `--first-frame photo.png` |
| start here, end there | `--first-frame a.png --last-frame b.png` |
| pass through a shot mid-clip | `--keyframes b.png@48` |
| a consistent character | `--reference-images face1.png face2.png` |
| match a clip's motion | `--reference-videos clip.mp4` |
| lip-sync to your own audio | `--reference-audio speech.wav` |

**Which model file gets loaded follows from what you asked for**, never from a
flag you might set wrongly. References load `Ref2VA`; prompts and frame anchors
load `FL2VA`. Point the config at the wrong one and `h3` refuses, rather than
producing something that looks fine and was made by weights never trained for
the job.

### Making a character talk

Put the dialogue in the prompt with a speaker tag. The lip movement and the voice
come out of the same pass:

```bash
h3 render --out hello.mp4 --prompt \
  'A woman in a sunlit kitchen looks at the camera. (S1) smiles and says
   <d>[English] This is running on Apple silicon now.</d>'
```

Measured on that clip: the audio transcribed back with Whisper at a **word error
rate of 0.00**, and a face detected with landmarks in **124 of 124 frames**.

[![A woman in a kitchen](demo-media/talking-head.jpg)](demo-media/talking-head.mp4)

[docs/PROMPTING.md](docs/PROMPTING.md) has the full grammar — camera moves,
soundscape, music — and is worth ten minutes if you want reliable results.

### Frame anchors

`--first-frame` and `--last-frame` pin a still to the two ends. `--keyframes
image.png@48` pins one anywhere, as `path@frame`.

Anchors land where you put them: measured on a three-anchor render, each anchor
reproduced its own frame at a normalised cross-correlation of **0.999**, against
a 0.850 baseline for how alike the anchors were to each other, and the frames
between them interpolated rather than cutting. That was one clip — the two ends
are what the model was trained on, and a middle anchor is out of distribution, so
it is measured to work rather than guaranteed to.

Frame numbers index the **rendered** timeline, which is your duration rounded up
onto the model's frame lattice: 5 s is 124 frames, so the last is 123, not 119.
`h3` refuses `seconds × fps − 1` for that reason rather than silently placing
your anchor early. Each anchor costs its rows twice — once in the packed sequence
and once as a `<Picture N>` block — so they are not free: three anchors measured
about 20% more sampling time than one, at 864×480×124.

---

## All the commands

```
h3 doctor            what this Mac can run, which files it found, what it chose
h3 config init       write a configuration file
h3 config validate   resolve every path and identify what is there
h3 render            make a video
h3 bench             compare recorded renders
```

Useful `render` options:

| | |
|---|---|
| `--out-audio out.wav` | also write the audio on its own |
| `--seconds 4…15` | duration. Use 5 or more; 4 lands under the model's trained floor |
| `--steps 20` | 20 is the verified value. **More is nearly free** — see below |
| `--seed 7` | same seed, machine and version → the same video |
| `--width` / `--height` | exact size, multiples of 32 |
| `--aspect-ratio 16:9` | or `9:16`, `21:9`, `4:3`, `3:4`, `1:1` |
| `--quality faithful` | turn off the cache; nothing approximated |
| `--cfg-scale 5` | stronger prompt adherence, at twice the work per step |
| `--negative-prompt "..."` | needs `--cfg-scale` above 1 to do anything at all |
| `--dry-run` | print the plan and the estimate, load nothing |
| `--verbose` | memory figures and backend choice while it runs |

`h3 render --help` lists every one.

---

## Speed and quality

Measured on a Mac Studio (M3 Ultra) at 864×480, 5 seconds, 20 steps. Ranges are
the spread across five prompts, not a single run.

| quality | time | what changes |
|---|---|---|
| `balanced` *(default)* | **11.4–11.8 min** | reuses work between steps — softer fine texture |
| `faithful` | **21.1–21.5 min** | nothing is approximated |

**`balanced` is the default and it is an approximation.** It reuses one step's
work in the next when the step barely moved, which is where roughly half the
render time goes. What it costs is fine texture — hair, foliage, fabric weave,
the lettering on a book spine. How much depends on the material: on a still
subject it is hard to find, and on high-frequency detail in fast motion you can
see it side by side. That is the trade almost everybody wants on a twenty-minute
render, and `--quality faithful` turns it off.

**The cache is not what degrades fast motion.** Measured on a deliberately
hostile case — a car crossing a wall of bookshelves, where every spine moves
between frames — detail fell by half between the stillest and fastest parts of
the clip *with the cache off*, several times anything the cache contributed. A
slow pan across the same shelves held its detail all the way through. Fast motion
costs texture because fast motion is blurred; the cache adds a little on top of
that, and does not cause it.

**Asking for more steps is close to free.** Doubling to 40 steps takes 12.2
minutes against 11.7 for 20 — about 4% more wall clock for twice the sampling. A
finer schedule moves the latent less per step, so more steps fall below the reuse
threshold: the cache skips 75% of them at 40 steps against 50% at 20.

**Larger costs more than proportionally.** Attention grows with the *square* of
the sequence length, so doubling the resolution roughly quadruples the time.

**A dry run prices a shape; it does not prove one will run.** It returns before
the memory admission check, so it will confidently price a configuration the
machine then refuses — 12 seconds at 576×1024 priced at "about 2h 04m" and was
refused wanting ~260 GB. The only way to know a shape fits is to start it and
watch for `H3-4001`, which costs seconds.

Measured peaks, for planning: **87.2 GB at 864×480×124**. Weights are a fixed
~66 GB and activations scale with packed tokens, so 576×1024×243 lands near
123 GB. A 96 GB Mac fits the first and not the second.

The `int8` files are half the size on disk and **exactly the same size in
memory** — they are expanded at load, so they save download, not RAM. The
`pruned` files genuinely are smaller in memory and they change the maths, so `h3`
refuses them unless you set `allow_approximate_weights` in the config.

The full write-up — sparse attention, partial cache refresh, quantised GEMM,
graph compilation, kernel fusion, the quality traps — is in
[docs/PERFORMANCE_GUIDE.md](docs/PERFORMANCE_GUIDE.md).

---

## For developers

It is a Swift package as well as a tool. The render API is an actor, so one
process cannot start a second render by accident — a model this size must never
be loaded twice:

```swift
import MiniMaxH3

let engine = RenderEngine(models: models)
let job = try await engine.start(RenderRequest(prompt: "...", videoOutput: url))
for await event in job.events { print(event) }
let result = try await job.value()
```

MLX types never appear in the public API, which is what lets the attention
backend, the quantisation scheme and the compute backend change without breaking
callers.

**On correctness.** This was ported against the CUDA reference with parity
established by measurement rather than hope: 225 gating taps inside CUDA-measured
equivalence classes at production shape, and 36 documented contracts in
[FRAGILE_CONTRACTS.md](FRAGILE_CONTRACTS.md), each carrying the evidence that
established it. That file exists because this codebase's failure mode is
*silent* — a wrong packed layout, a dropped label or a transposed qkv all keep
every tensor exactly the right shape.

Further reading: [CONTRIBUTING.md](CONTRIBUTING.md) for setup and house style,
[docs/adr/](docs/adr/) for design decisions, and
[docs/OPERATIONS.md](docs/OPERATIONS.md) for running it in anger.

<details>
<summary><b>Architecture</b></summary>

Dependencies point downward only, and the four lowest layers do not link MLX — so
geometry, the frame lattice, checkpoint identification and the memory planner all
test in milliseconds with no GPU and no 66 GB download. Those are exactly the
places where an error stays silent.

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

</details>

## Status

Verified end to end through this package: text-to-video with joint audio at
864×480×124, checked with two oracles that need no reference — speech at **WER
0.00** and a face with landmarks in **124/124 frames** — plus first-frame and
mid-timeline keyframe anchors, measured landing at their frames.

The three reference kinds were rendered end to end in the experimental tree this
was ported from, and have not yet been re-run here.

**Known defect.** On violent fast motion, renders can drop whole frames to black:
7 of 124 on one 20-step test, cache off. `docs/PROMPTING.md` reports these
disappearing above 6 steps, which holds for the prompt it was measured on but not
for this one. Under investigation.

Not done:

- **Smaller machines.** Two routes, neither built. Resident quantised weights
  was the assumed one. Block streaming is now the measured one — reading each
  transformer block from disk while the previous one computes costs 1.9–4.9%
  and holds 64 GB of weights in 2.4 GB, verified bit-identical. What that has
  not been tested on is a 48–64 GB Mac, which is the only machine the claim is
  about. `docs/PERF_ROADMAP.md` §7.
- **Chunked video decode.** The largest allocation in a render, and not a hard
  problem.
- **2K output.** Produced by a separate upscaler MiniMax has not released.
- **Fixtures for `H3Conformance`**, so the 36 contracts are documented but not
  yet enforced by a test.

## Licence

**Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Use it for anything, including commercially and in closed-source products.
Modify it, fork it, ship it. Two things travel with it: keep the copyright and
the `NOTICE` file in what you distribute, and state that you changed the files
you changed.

`Resources/mlx.metallib` is MLX's compiled Metal kernel library, redistributed
under its own MIT licence — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model weights are not in
this repository and are licensed separately by MiniMax.

> **If you verify the signature on a download**, `codesign -dv` reports
> `Developer ID Application: Tesserapps, LLC`, which is not the name on the
> copyright above. That is expected: the copyright is personal and Tesserapps is
> the Apple developer account whose certificates sign the binaries.
