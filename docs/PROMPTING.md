# Writing prompts for MiniMax-H3

Refined against MiniMax's own guides:

* `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` — T2VA, I2VA, FL2VA, L2VA
* `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` — full-reference mode

Both are *rewriter output formats*: what MiniMax's own prompt rewriter emits
before the model sees it. Writing in that format directly is how you skip the
rewriter and address the model in the shape it was trained on.

**Every conditioning path now runs.** `--first-frame`, `--last-frame`,
`--reference-images`, `--reference-videos`, `--reference-video-audios` and
`--reference-audio` all reach the DiT: each asset is VAE-encoded into the packed
sequence *and* presented to the conditioning encoder — `<Picture N>` for images,
timestamped `<Video N>` frame pairs for clips, a bare `<Audio N>` label for
sound. The reference does both halves from the same asset, so doing only one
leaves the DiT conditioned on a prompt that never mentions what it is being
asked to follow. Section 8 has the details, including the two frame lattices
that round in opposite directions.

---

## 0. Defaults and the render policy

`h3 render` refuses configurations that are known to produce bad output, and
says why. The thresholds are measured, not guessed.

| setting | default | why |
| --- | --- | --- |
| `--seconds` | **5 s** | Frame counts snap up onto the 17k+5 lattice, so 4 s becomes **107 frames** — the only value in the whole 4-15 s range that lands under the model's 124-frame trained floor. 5 s gives exactly 124. |
| `--steps` | **20** | 6 steps measured 52 flash events and frame-to-frame luminance sd 0.0087; 20 measured **zero** events and sd 0.0009 on the same prompt. |
| `--resolution` | **768p** | 2K is not a base render size — see below. |

**2K is an upscale target, not a render size.** It is the output of In-Context
Regeneration, a second stage that takes a base render and enlarges it. The DiT
does not sample 2560x1440 directly, and that upscaler is **not released and not
ported**, so `--resolution 2k`
and every `h3_2k_*` recipe now refuse, and the recipe registry tags them
`.upscaleTarget` rather than pretending they are base configs. They are kept, not
deleted, because they are the real output sizes callers will ask for once the
upscaler exists.

**The verified configuration is 864x480, 124 frames, 20 steps.** It is
simultaneously the parity-gated shape (225 gating taps inside the CUDA-measured
class) and the shape proven to render a face detectable in 124/124 frames with
its dialogue transcribed correctly. Nothing else has both properties.

**It is no longer a recipe.** The ladder was re-derived for the decoder's tile
grid and 0.4 MP is now 832x448, so the verified shape is reachable only with an
explicit `--width 864 --height 480`. Verification does not travel with a name,
and no current rung has been through `Tools/ShapeVet/vet.sh`.

The banner prints a cost estimate — packed tokens, estimated sampling minutes,
and the multiple of the verified config. It is computed from the DiT's actual
FLOP count divided by the throughput a GEMM benchmark measured on this machine, and it
predicted 20 min against 21.3 min actual. Attention is quadratic in sequence
length, so resolution costs far more than linearly: 768p 16:9 is 2.5x the tokens
of the verified shape and about 3.7x the time.

`--allow-suboptimal` runs a rejected configuration anyway and logs every reason.
It exists because the out-of-distribution case *was* already a warning, and the
warning was ignored by the person who wrote it — costing an evening spent
hunting a decoder bug that did not exist.

---

## 1. The output shape, for T2VA

Three sections, in this order. **T2VA omits the image-alignment instruction**
that the other three modes open with.

```
integrated_multimodal_description
overall_soundscape
non_diegetic_music
```

Everything the audience *sees*, plus dialogue and synchronised sound, goes in
the first. Ambience goes in the second. Score goes in the third. The split is
not cosmetic — the model was trained on it.

**Length: 350–500 English words** for a generation task. Write the body in
English even when the dialogue is not.

---

## 2. Shots and time

The first shot carries **no timestamp**. Every later shot does:

```
[Shot 1] ...
[Shot 2] At 00:03.500, the camera cuts to ...
```

Number sequentially. A cut introduces new compositional information; if you
only want the framing to shift, that is camera motion, not a new shot.

Durations are written to exactly two decimals — `6.00 seconds`.

---

## 3. Camera motion: type + amplitude + speed

The grammar is fixed and the phrasings are literal.

| slot | allowed values |
|---|---|
| motion | Zoom In/Out · Push In/Out · Pan Left/Right · Truck Left/Right · Tilt Up/Down · Pedestal Up/Down · Arc Shot · Tracking Shot · Static Shot · Shake Slightly/Strongly · POV · Roll Clockwise/Counterclockwise |
| amplitude | `with small amplitude` · `with large amplitude` |
| speed | `at slow speed` · `at fast speed` |

> The camera pushes in with small amplitude at slow speed toward the folded letter.

**Do not** stack motion descriptors as labels at the end of a sentence. Put the
motion in the sentence.

---

## 4. Dialogue and speakers

Speaker ids are stable across the whole prompt: `(S1)`, `(S2)`, `(S1,S2)` when
two people speak together. Assign them in the order voices actually appear.

Identity and action live **outside** the tag; only language and words live
inside:

```
<Subject 2> (S1) turns toward the window and says <d>[English] It stopped raining.</d>
```

* Preserve the original words and punctuation exactly.
* Voiceover: `says in an off-screen voiceover`, and then state that the
  speaker's `lips remain completely closed`.
* Dialogue crossing a cut: mark `<scenetrans>` and say it continues seamlessly
  across the cut.
* Speech cut off by the end of the video: `<cutoff>`.

**On-screen text** goes in English double quotes, in its original form. Do not
translate it.

---

## 5. Soundscape and score

`overall_soundscape` — 1–4 sentences, one paragraph. Ambience, physical action
sounds, non-verbal human sounds: wind, rain, footsteps, fabric, breathing,
laughter. **No dialogue, no singing, no diegetic music** — those belong in the
description. `N/A` only if you are explicitly asking for silence.

`non_diegetic_music` — 1–3 sentences. Score the audience hears and the
characters do not. Name **instrumentation, speed, rhythm, dynamic change**.
Avoid abstract mood words. Music a character can hear — a radio, a busker, a
phone — is diegetic and belongs in the description instead. `N/A` when there is
none.

---

## 6. Negation goes in the CFG, never in the prompt

**Neither official guide has a negative-prompt block.** There is no "avoid
this" section in the format, because the format is a description of the video
you want. "no camera shake" in a prompt conditions the model *toward* the
tokens "camera shake".

Negation is a sampling-time operation:

    guided = neg + s · (cond − neg)

```bash
h3 render --out out.mp4 --prompt "<positive>" --negative-prompt "camera shake, subtitles, plastic skin" --cfg-scale 5.0 --width 864 --height 480 --seconds 6 --steps 20
```

`--cfg-scale 1.0` is the identity and skips the second pass. Useful values are
roughly **3.0–7.5**. Omit `--negative-prompt` and guidance runs against a null
context, which sharpens prompt adherence without subtracting anything specific.

`--negative-prompt` at scale 1.0 is refused rather than silently ignored.

**What it costs.** Two full forward passes per step, sequential — H3 is batch-1
only, so they cannot be batched. Measured at 864x480x158: **30.6 min without
guidance, 57.4 min with**. It is not a quality dial to leave on.

**Keep the negative short.** It is a full conditioning pass, not a filter; a
sprawling one starves the sampler and shows up as muddy texture. Roughly
20–30% of the positive by length. Raise `--cfg-scale` before lengthening it —
the CLI warns past 50%.

---

## 7. Duration, and what you actually get

* **Trained range is 124–362 frames**, 5.2–15.1 s at 24 fps. Below 124 renders
  fine and is out of distribution — do not read it as a quality signal.
* Frame counts **snap up onto a 17k+5 lattice**, so a request is a lower bound:
  4 s (96 frames) becomes 107, still under the floor. 6 s (144) becomes 158,
  which is 6.58 s. `generate` warns when you land outside the trained range.
* **Ask for 6 seconds or more.**
* `--width`/`--height` override the resolution tiers and must be multiples of
  32. The tiers cannot express 864x480, which is the shape every golden and
  every measured tolerance uses.

---

## 8. What this repo can and cannot run

Text in, one mp4 with sound out:

```bash
h3 render --out out.mp4 --prompt "..." --width 864 --height 480 --seconds 6 --steps 20
```

With a keyframe, add `--first-frame` (and `--last-frame`):

```bash
h3 render --out out.mp4 --prompt "..." --first-frame shot.png --width 864 --height 480 --seconds 6 --steps 20
```

Anchors elsewhere on the timeline go through `--keyframes`, as `path@frame`:

```bash
h3 render --out out.mp4 --prompt "..." --keyframes a.png@0 b.png@72 c.png@140 --width 864 --height 480 --seconds 6 --steps 20
```

Three things about those frame numbers. They index the **rendered** timeline,
which is the requested duration snapped up onto the 17k+5 lattice — 6 s at 24 fps
is 158 frames, so the last one is 157 and not the 143 the obvious arithmetic
gives. `seconds * fps - 1` is refused for that reason rather than accepted more
than half a second early; `--last-frame` resolves the end for you. And the
positions are exact but the behaviour is not gated: fl2va was trained with
conditioning at the two ends, so a middle anchor is out of distribution. It pins
the rows to the right instant; whether the picture lands there is the model's
call.

> **`--conditioning-noise` is for reproducing reference results, and most people
> should ignore it.** Keyframe rows are noise-augmented, and the reference draws
> that noise from `torch.Generator("cpu")`, which MLX's PRNG cannot reproduce —
> so matching the reference bit for bit means supplying the noise as a recorded
> tensor. The emitter that produces one lives in the parity tooling and is not
> shipped with this package. Without it the render happens anyway, from MLX's
> PRNG, and warns that it is outside the parity contract. For making videos that
> warning does not matter; for checking this port against CUDA it is the whole
> point.

**A keyframe travels two paths at once**, which is the thing to know here. The
same image is VAE-encoded into a `cond` segment of the packed sequence *and*
presented to the conditioning encoder as `<Picture 1>: `. The reference does
both from one image; doing only the first leaves the model reading a prompt
that never mentions the picture it is being asked to start from.

**References: `--reference-images`, `--reference-videos`, `--reference-audio`.**
Anchors and references are mutually exclusive in one payload — the reference
raises on the combination and so does this.

```bash
h3 render ... --reference-images subject.png prop.png location.png
h3 render ... --reference-videos clip.mp4 --reference-video-audios clip.wav
```

`--reference-video-audios` is **index-matched** to `--reference-videos`, which
is not cosmetic. A paired soundtrack's `<Audio j>` label is emitted immediately
*before* its `<Video k>` — so the audio ordinal can precede the video ordinal —
and the DiT block becomes `video_audio`, with the audio rows ahead of the video
rows inside one block, advancing the position cursor by `max(refAudioT,
videoTSpan)` rather than by either alone. A standalone `--reference-audio` does
none of that. Pass `''` to leave one video silent.

**Two frame lattices, rounding in opposite directions.** The generation length
snaps **up** onto `n % 17 == 5` (5 s -> 124 frames). A reference video's length
walks **down** onto the same lattice, after first being truncated to the
generation length. Both are in `nodes_minimax_h3.py` and they are three lines
apart. Getting the direction wrong on the reference side is silent: the VAE's
temporal chunker pads its tail by repeating the last frame, so an off-lattice
count encodes manufactured frames as if they were footage, with every shape
still valid.

**Give reference media to the renderer already at the canvas size.** The
reference resizes it with PIL LANCZOS on uint8 (`comfy.utils.lanczos`), which is
not ported — but `adapt_canvas(864, 480)` proposes 1376x768, a smaller source
loses, and the node falls back to the source's own 32-aligned size. So at
864x480 the reference's resize is the identity, and a file already on the canvas
skips the unported span *in both implementations* rather than having it
approximated. `videoHWC` refuses anything off a multiple of 32 and prints the
`ffmpeg -vf scale=...:flags=lanczos` line that fixes it.

Still refused, loudly:

* a prompt with no text encoder configured — run `h3 doctor` to see what is missing.
* batched classifier-free guidance — H3 is batch-size-1 by construction; CFG runs as two
  sequential forwards.

**What that leaves.** The 50-block DiT stack is shared across every modality and
is verified. So is the packed layout for every conditioning kind (7 fixtures,
full position ids), both VAE encoders, the vision tower, the image-prompt path
and the ref2va presentation through the language stack. The one span still
outside the gate is the LANCZOS resize, and the sizing rule above is how a real
render avoids needing it.

## 9. A worked T2VA prompt

```
integrated_multimodal_description
Cinematic 35mm film with warm low-contrast grading, shallow depth of field and
visible grain.
[Shot 1] A ceramic cup sits on a walnut table in front of a west-facing window,
backlit by late afternoon sun. Steam rises in a slow unbroken column. The camera
holds a Static Shot. <Subject 1> (S1), a woman in a grey sweater, enters frame
right and sits, then says <d>[English] I didn't think you'd still be here.</d>
[Shot 2] At 00:03.500, the camera cuts to a closer angle across the table and
pushes in with small amplitude at slow speed toward the cup. Her hand closes
around it. The window behind her reads "OPEN" in reversed lettering.

overall_soundscape
Quiet interior room tone with a faint clock tick, the soft scrape of a chair on
wood, and fabric shifting as she sits. Distant traffic sits low under everything.

non_diegetic_music
A sparse piano figure at a slow tempo, single notes with long decay, swelling
gently through the second shot and receding at the end.
```

Everything proscriptive — shake, subtitles, plastic skin, oversaturation —
stays out of this and goes to `--negative-prompt`.
