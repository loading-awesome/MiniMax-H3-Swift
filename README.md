# h3 — MiniMax-H3 on Apple silicon

Type a sentence, get a video **with the sound already in it**. No GPU rental, no
Python environment, no node graph. One command, on your Mac.

<video src="https://github.com/loading-awesome/MiniMax-H3-Swift/raw/main/docs/media/stoplight.mp4" controls muted playsinline width="100%"></video>

<sub>Rendered by this tool: 864×480, 7 s, 20 steps. The dialogue, the engine
noise and the music were generated *together with the picture* — nothing was
dubbed on afterwards. [Download](docs/media/stoplight.mp4) if your browser will
not play it inline.</sub>

```bash
h3 render --prompt "a red kite over a beach at sunset" --out kite.mp4
```

---

## What this is

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is an open-weights
video model that is unusual in one specific way: **it generates the picture and
the soundtrack in a single pass.** Most video models hand you a silent clip and
leave you to add audio. H3 moves a character's lips and produces the voice
saying those words at the same time, from the same prompt.

`h3` is a native Swift and [MLX](https://github.com/ml-explore/mlx) port of it
for Apple silicon — a command-line tool and a Swift library, with no Python
anywhere in it.

**It is not a small, friendly download.** The model is 124 GB of weights and the
clip above took about twenty minutes on a Mac Studio. Please read the next
section before you get attached to the idea.

## Will it run on my Mac?

Probably not, and I would rather say so here than have you find out after a very
long download. Two things gate it, and **the disk usually stops people first**:

| | you need |
|---|---|
| **memory** | 96 GB unified memory, realistically 128 GB |
| **disk** | 124 GB free for text-to-video, 190 GB for everything |
| **chip** | any Apple silicon (M1 or later) |
| **macOS** | 14 or later |

Which works out as:

| Mac | verdict |
|---|---|
| MacBook Air, any configuration | **no.** A 16 GB Air with a 256 GB drive cannot *store* the model, never mind run it. |
| MacBook Pro or mini, 16–36 GB | no |
| M4 Pro, 48 GB | not yet — needs the work listed under [Status](#status) |
| M4 Max, 64 GB | not yet |
| **96 GB or more** | yes |
| **Mac Studio, 128 GB+** | yes, comfortably |

I would like this to reach further down the range, and it does not yet. Two
specific pieces of work would bring it to roughly 48 GB — keeping the weights
quantised in memory, and decoding the video in chunks. Neither is done, and
neither is hand-waving: they are the two items at the top of [Status](#status).

`h3 doctor` answers the question for your actual machine in about a second. It
reads file headers, not file bodies, so it is fast even against 66 GB files:

```
$ h3 doctor
machine
  Apple M3 Ultra (Mac15,14), 275 GB unified memory, 28 cores
  145 GB available right now (free + inactive + speculative)

memory plan at 15750 packed tokens
  bf16         peak   98.7 GB   FITS, +32 GB headroom
  int8         peak   98.7 GB   FITS, +32 GB headroom   [dequantised at load — saves disk, not memory]
  pruned_bf16  peak   73.8 GB   refused (approximate weights)

selected: bf16
  textEncode   53.5 GB
  vaeEncode     7.8 GB
  sampling     98.7 GB   <- peak
  decode       13.9 GB
  available   145.9 GB
  headroom     +32.4 GB after a 15% margin

no problems found
```

---

## 1. Get `h3`

**Download the installer.** No compiler, no Xcode, no Terminal:

### [⬇ Download the latest release](https://github.com/loading-awesome/MiniMax-H3-Swift/releases/latest)

Take the **`.pkg`** and double-click it. It puts `h3` on your path, and you can
then open Terminal and type:

```bash
h3 doctor
```

<details>
<summary>Or take the <code>.zip</code>, if you would rather not run an installer</summary>

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

> **If macOS says the file is damaged or from an unidentified developer**, you
> have an unsigned build — a build made from source, or from a release cut
> before signing was set up. Clear the quarantine flag once, in the folder you
> unpacked into:
>
> ```bash
> xattr -dr com.apple.quarantine h3
> ```

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

`bootstrap-metal.sh` is not optional. SwiftPM does not build MLX's Metal
kernels — mlx-swift compiles those in its Xcode project — so without it the
first render dies with an unhelpful C++ error naming no file.
See [CONTRIBUTING.md](CONTRIBUTING.md).
</details>

## 2. Get the model files

The weights are **not** in this repository and are licensed separately by
MiniMax. You need these, from the single-file conversions published for
[WanGP](https://github.com/deepbeepmeep/Wan2GP) on Hugging Face, which are
derived from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3):

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
instead of 190 GB.

Put them anywhere you like; `h3` never moves or copies them.

> You will also see `int8` and `pruned` versions with smaller file sizes. They
> are worth knowing about and mostly will not help yet — see
> [Speed and quality](#speed-and-quality).

## 3. Point `h3` at them

```bash
h3 config init
```

That writes `~/.config/minimax-h3/config.json` with the filenames already filled
in. Open it and set two things — the folder your models are in, and the
tokenizer:

```json
{
  "checkpoints": {
    "root": "/Users/you/models/MiniMax-H3",
    "tokenizer": "/Users/you/models/MiniMax-H3/tokenizer"
  }
}
```

Then check your work:

```bash
h3 doctor
```

Every file is identified **from its own header, not from its name**, so one that
has been renamed, truncated or converted differently is reported as what it
actually is rather than what it claims. Anything missing is listed with the path
that was checked. When it says **`no problems found`**, you are ready.

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
above it does not yet know about the cache, so on the default profile it reads
high until the countdown takes over.

Two flags worth knowing straight away:

```bash
h3 render --prompt "..." --out out.mp4 --dry-run           # what would this cost? (1 second)
h3 render --prompt "..." --out out.mp4 --quality faithful   # nothing approximated, ~2x slower
```

---

## What you can make

| you want | how |
|---|---|
| video from a description | just `--prompt` |
| a character who speaks | put the words in the prompt (below) |
| start from your photo | `--first-frame photo.png` |
| start here, end there | `--first-frame a.png --last-frame b.png` |
| a consistent character | `--reference-images face1.png face2.png` |
| match a clip's motion | `--reference-videos clip.mp4` |
| lip-sync to your own audio | `--reference-audio speech.wav` |

**Which model file gets loaded follows from what you asked for**, never from a
flag you might set wrongly. References load `Ref2VA`; prompts and frame anchors
load `FL2VA`. Point the config at the wrong one and `h3` refuses, rather than
producing something that looks fine and was made by weights never trained for
the job.

### Making a character talk

Put the dialogue in the prompt with a speaker tag. The lip movement and the
voice come out of the same pass:

```bash
h3 render --out hello.mp4 --prompt \
  'A woman in a sunlit kitchen looks at the camera. (S1) smiles and says
   <d>[English] This is running on Apple silicon now.</d>'
```

The clip at the top of this page uses two speakers, `(S1)` and `(S2)`.
[docs/PROMPTING.md](docs/PROMPTING.md) has the full grammar — camera moves,
soundscape, music — and is worth ten minutes if you want reliable results.

---

## Speed and quality

Measured on a Mac Studio (M3 Ultra) at 864×480, 5 seconds, 20 steps:

| quality | time | what changes |
|---|---|---|
| `balanced` *(default)* | ~13 min | reuses work between steps — **16% less fine detail** |
| `faithful` | ~23 min | nothing is approximated |
| `fast` | ~10 min | 28% less fine detail |

**`balanced` is the default, and it is an approximation.** It reuses one step's
work in the next when the step barely moved, which measured 1.8–1.9× faster for
16% less fine detail. On a twenty-minute render that is the trade almost
everybody wants, and finding out afterwards that a flag would have halved it is
worse than the 16%.

It says so on every run and it is recorded in the render's receipt, so nobody
discovers months later that their comparison was against a shortcut. That
disclosure is the part that matters, and it never depended on which profile was
the default.

```bash
h3 render --prompt "..." --out out.mp4 --quality faithful   # nothing approximated
```

Larger costs more than proportionally: attention grows with the *square* of the
sequence length, so doubling the resolution roughly quadruples the time.
`--dry-run` tells you before you commit twenty minutes.

The `int8` files are half the size on disk and **exactly the same size in
memory**, because they are expanded back to full precision at load. They save
download and storage, not RAM. The `pruned` files genuinely are smaller in
memory and they change the maths, so `h3` refuses them unless you set
`allow_approximate_weights` in the config — nobody should accidentally compare
one against the real thing.

---

## All the commands

```
h3 doctor            what this Mac can run, which files it found, what it chose
h3 config init       write a configuration file
h3 config validate   resolve every path and identify what is there
h3 render            make a video
```

Useful `render` options:

| | |
|---|---|
| `--out-audio out.wav` | also write the audio on its own |
| `--seconds 4…15` | duration. Use 5 or more; 4 lands under the model's trained floor |
| `--steps 20` | more is slower and slightly better; 20 is the verified value |
| `--seed 7` | same seed, machine and version → the same video |
| `--width` / `--height` | exact size, multiples of 32 |
| `--aspect-ratio 16:9` | or `9:16`, `21:9`, `4:3`, `3:4`, `1:1` |
| `--cfg-scale 5` | stronger prompt adherence, at twice the work per step |
| `--negative-prompt "..."` | needs `--cfg-scale` above 1 to do anything at all |
| `--dry-run` | print the plan and the estimate, load nothing |
| `--verbose` | memory figures and backend choice while it runs |

`h3 render --help` lists every one.

## Examples

**Speech generated with the picture.** 864×480, 5 s. The audio was transcribed
back with Whisper at a word error rate of **0.00** against the prompt's line,
and a face was detected with landmarks in **124 of 124 frames**.

<video src="https://github.com/loading-awesome/MiniMax-H3-Swift/raw/main/docs/media/talking-head.mp4" controls muted playsinline width="100%"></video>

<sub>[Download](docs/media/talking-head.mp4)</sub>

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
established by measurement rather than hope: 225 gating taps inside
CUDA-measured equivalence classes at production shape, and 36 documented
contracts in [FRAGILE_CONTRACTS.md](FRAGILE_CONTRACTS.md), each carrying the
evidence that established it. That file exists because this codebase's failure
mode is *silent* — a wrong packed layout, a dropped label or a transposed qkv
all keep every tensor exactly the right shape.

Further reading: [CONTRIBUTING.md](CONTRIBUTING.md) for setup and house style,
[docs/adr/](docs/adr/) for design decisions and their reasons, and
[docs/OPERATIONS.md](docs/OPERATIONS.md) for running it in anger.

<details>
<summary><b>Architecture</b></summary>

Dependencies point downward only, and the four lowest layers do not link MLX —
so geometry, the frame lattice, checkpoint identification and the memory planner
all test in milliseconds with no GPU and no 66 GB download. Those are exactly
the places where an error stays silent.

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
864×480×124, checked with two oracles that need no reference — speech at
**WER 0.00**, and a face with landmarks in **124/124 frames**.

Frame anchors and the three reference kinds were rendered end to end in the
experimental tree this was ported from, and have not yet been re-run here.

Not done:

- **Resident quantised weights.** The single item that decides whether 48–64 GB
  Macs are supported.
- **Chunked video decode.** Currently the largest memory allocation in an entire
  render, and not a hard problem.
- **2K output.** Produced by a separate upscaler that MiniMax has not released.
- **Fixtures for `H3Conformance`**, so the 36 contracts are documented but not
  yet enforced by a test.

## Licence

**Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Use it for anything, including commercially and in closed-source products.
Modify it, fork it, ship it. Two things travel with it: keep the copyright and
the `NOTICE` file in what you distribute, and state that you changed the files
you changed. That is the attribution.

`Resources/mlx.metallib` is MLX's compiled Metal kernel library, redistributed
here under its own MIT licence — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model weights are not in
this repository and are licensed separately by MiniMax.

> **If you verify the signature on a download**, `codesign -dv` reports
> `Developer ID Application: Tesserapps, LLC`, which is not the name on the
> copyright above. That is expected: the copyright is personal and Tesserapps is
> the Apple developer account whose certificates sign the binaries. Same author,
> two hats — checking the signature is the right instinct and this is the
> explanation.
