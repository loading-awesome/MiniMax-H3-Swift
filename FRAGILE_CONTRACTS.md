# Fragile Contracts

Break any of these and parity goes with it. Each entry carries the evidence
that established it. Add to this list whenever a debugging session ends in
"…and it must stay that way."

Status key: **[V]** verified by measurement · **[A]** assumed from source
reading, not yet measured.

---

### 1. Sigma shift is 12.0 (video) / 3.0 (audio) — in two places **[V]**

`comfy/ldm/minimax/model.py:418` (DiT constructor defaults) and
`MiniMaxH3SigmaShift` node defaults. The DiT reads
`transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video)`
(`model.py:527`), so the two must agree or the sampler's schedule and the
DiT's internal grid mapping will disagree silently.

The resulting back-loaded schedule (15 of 20 steps spanning sigma 1.0→0.8) is
**correct**. Verified analytically: `12t/(1+11t)` at t=0.5 = 0.9231, matching
the observed sigma. Do not "fix" it.

### 2. The latent is a `NestedTensor`, not a tensor **[V]**

`{"samples": comfy.nested_tensor.NestedTensor((video, audio))}`. Video is
`[B,24,latent_t,H//16,W//16]`, audio is `[B,32,2,audio_t]`. Order is
**(video, audio)** — `unbind()[0]` is video.

Any port that assumes `samples` is a plain tensor will appear to work and be
wrong.

### 3. Frame count lives on the 17k+5 lattice **[V]**

`latent_t = 5·(frames-5)/17 + 2`. Off-lattice frame counts are snapped up by
`align_frame_count`. Trained range **124–362**; 124 ≈ 5 s at 24 fps.

Rendering below 124 frames is out of distribution. Our 73-frame run was
below it — do not treat sub-124 output as a quality signal.

### 4. Text conditioning is the layer-50 hidden state, unnormalized **[V]**

`comfy/text_encoders/minimax.py:15`. The published checkpoint is truncated at
layer 50, so "last layer" and "layer 50" coincide *for that file only*. A port
that loads the full 66.71 GB official encoder must take layer 50 explicitly
and must **not** apply the final norm (`layer_norm_hidden_state=False`,
`minimax.py:105`).

### 5. Audio runs at 32 kHz with 40 latents/sec **[V]**

`comfy/ldm/minimax/audio_vae.py:391,397`. `audio_t = round(frames/24 · 40)`.
The audio VAE is fp32 and is the *only* component published at full
precision — do not downcast it to match the video VAE.

### 6. Audio must be muxed in encoder-frame-sized chunks **[V]**

AAC requires 1024-sample frames. One whole-waveform `AudioFrame` returns
EINVAL. See `docs/H3_PARITY.md` § "Fixed: audio muxing".

### 7. FL2VA serves T2VA **[A]**

`MiniMaxH3ImageToVideo` docstring: "t2va and fl2va". Text-to-video does **not**
use the Ref2VA partition. Ref2VA is for reference image/video/audio inputs
only. Not independently verified — flagged [A] because we have never produced
a correct render from either partition.

### 8. bf16 is the ceiling, not a compromise **[V]**

Official `MiniMaxAI/MiniMax-H3` transformer is 66.28 GB across 14 shards —
identical size to the single-file bf16 conversion. There is no fp32 DiT to
fall back to. If a parity diff shows bf16-level noise, that noise is in the
reference too.

The one genuine downcast in the ComfyUI ecosystem is the **video VAE**:
official is fp32 (10.42 GB), everyone ships fp16 (5.21 GB). If VAE-boundary
diffs ever look marginal, convert the fp32 original before suspecting the
math.

---

### 9. The two bf16 conversions are NOT interchangeable **[V]**

`attn.qkv_proj.weight` is stored in different memory layouts:

    Comfy-Org      (3, heads, headDim, hidden)   blocked  Q | K | V
    DeepBeepMeep   (heads, 3, headDim, hidden)   per-head interleaved Q,K,V

Same values, same shape `[21504, 5376]`, same dtype, **same std and mean to 8
decimal places** — only the arrangement differs. No shape, dtype, or statistics
check catches this. Loading one as the other yields output with the correct
magnitude and **cos 0.029 against the reference**: uncorrelated, but plausible
enough to look like a subtle numerical bug rather than a layout error.

Detect by metadata, which is unambiguous:

    Comfy-Org      __metadata__ = {"config": ...}
    DeepBeepMeep   __metadata__ = {"repo_id", "partition", "precision", "source_revision"}

`H3Core.CheckpointInventory.Vendor` does this, and `H3Weights` permutes on load.
A checkpoint whose vendor cannot be identified is **rejected** rather than
guessed at.

Corrects an earlier claim in `docs/reference/weights_inventory.md` that the two
bf16 DiTs were tensor-identical. That was inferred from a metadata-only sha
difference without comparing payloads, and it was wrong. `q_norm.weight` and
other unfused tensors *are* byte-identical; only fused tensors differ.

**Consequence for the local ComfyUI:** `models/diffusion_models/*.safetensors`
are hard links to DeepBeepMeep files, so ComfyUI on this Mac would misread
attention. Point it at Comfy-Org downloads for any real render.

---

### 10. The packed sequence is text | **audio** | video **[V]**

Audio comes before video, even though video is 36x longer at production shape.
`PackedLayout` appends "target audio then target video, always the last two
segments" (`comfy/ldm/minimax/model.py`), and the module docstring says so
outright: `[text | cond rows | audio | video]`.

Getting this backwards leaves **every tensor the right shape** and every value
in the wrong place — the packed sequence is `[S, hidden]` either way. There is
no shape check that catches it, and the taps a port would compare are all
`[S, hidden]` too.

Pinned by `LayoutTests` against a fixture dumped from the reference's own
`PackedLayout` (`parity/scripts/exporters/export_layout_fixture.py`), which
needs no weights and no GPU.

### 11. Audio velocity is scaled by d(sigma_a)/d(sigma_v) **[V]**

The DiT returns `[-video_out, -slope * audio_out]`. At sigma 1.0 that slope is
**4.0**, not 1.0 — the two streams' timesteps coincide there while their
schedules' *rates* do not.

Verified arithmetically against the tiny golden: `sampled.audio` equals
`noise.audio + 4 * audio_out` to 3.1e-02 (bf16 rounding), and equals
`noise.audio + 1 * audio_out` to only 1.3e+01. A port that drops the slope is
wrong by 4x on the audio stream and correct on video, which reads as an audio
VAE bug.

### 12. The final layer's AdaLN has ONE modality, blocks have three **[V]**

`AdalnProj(t_dim, hidden, expand=2, modalities=1)` in `FinalLayer` versus
`expand=6, modalities=3` in `DiTBlock`. So a block's AdaLN row is
`timestepRow * 3 + modalityTag`, and the final layer's row is the **timestep row
alone**. Indexing the final layer with a block-style row reads modulation
intended for another modality — plausible output, wrong values.

Shapes agree with either reading: `[10752, 2688]` is `2 * 5376 * 1`.

### 13. Video and audio share a temporal origin **[V]**

Both media streams begin their position-id cursor at `text_len`. They are
concurrent, not sequential. A port that advances the cursor past the audio span
before laying down video is off by 207 temporal units at 864x480x124 — a rope
error that leaves all shapes intact.

---

### 14. Parity tolerances are shape-specific and must be measured **[V]**

A tolerance measured at one shape does not transfer to another. Measured on
CUDA with `parity/scripts/validation/calibrate_l1_class.py`:

| tap | class at 25 tokens | class at 15,406 tokens |
|---|---|---|
| block_00 rel_rms | 3.6e-03 | 8.5e-03 |
| block_24 rel_rms | 7.0e-03 | 1.3e-02 |
| block_49 rel_rms | 2.8e-02 | 2.3e-02 |

Attention sums over 616x more keys at production shape, so the legitimate bf16
spread is wider there. The gate now travels with the golden as
`tolerances.json` rather than living in a table.

**Gate on `cos` and `rel_rms`, not `rel_max_abs`.** A single-element maximum
over tens of millions of values is a hopeless estimator: at block_00 two CUDA
attention kernels differ by 1.5e-04, at block_24 by 7.0e-03 — a factor of 46
from rounding alone, on the same run.

### 15. CUDA's bf16 DiT path is deterministic; its only variation is the SDPA backend **[V]**

Three things that might have given a second bf16 sample do not:

* fused vs unfused attention — **bit-identical** at both shapes (re-confirmed
  at production shape, not just tiny)
* `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` — no
  effect; cuBLAS accumulates bf16 GEMMs in fp32 on sm_120 either way
* wrapping a call in `sdpa_kernel(...)` — **silently ignored**. ComfyUI applies
  its own `SDPA_BACKEND_PRIORITY` with `set_priority=True` inside
  `comfy/ops.py`, which overrides any outer context. Override the module
  attribute instead.

Flash vs mem-efficient SDPA *is* a genuine second bf16 implementation
(3.08e-03 rel_rms apart on a standalone call at production shape), and it is
what the class is measured from. MATH is not usable at production shape — it
materialises the 15406^2 attention matrix, 49.5 GB.

---

### 16. AdaLN is ill-conditioned; compute it in fp32 **[V]**

`adaln_proj` is `linear(silu(t_emb))` over K=2688, and `silu(t_emb)` is tiny —
`t_emb` has std ~9e-03 at sigma 1.0 — so the reduction cancels heavily. The
**reference's own bf16-vs-fp32 spread on this op is 1.7e-03**, against 1.7e-04
for `attn.qkv_proj` and 1.8e-04 for `mlp.fc1` at the same shape. Three orders of
magnitude looser than any other matmul in the block.

Any difference in accumulation order lands at that level. In bf16 the MLX port
sat 1.55x outside the class here while every other operator sat at 0.00-0.23x.
Computing the projection in fp32 and rounding the result back to bf16 lands on
the reference's own fp32 result to 4 significant figures.

This is measured, not stylistic — `h3-parity oracle-block` reports all four
combinations. Padding the batch to force a gemm instead of a gemv changes
**nothing** (bit-identical), so it is not a kernel-dispatch effect.

Cost: AdaLN weights are 26 of the checkpoint's 66 GB, so the upcast adds about
50 GB of transient traffic per forward — under 1% of a 61 s production pass.

### 17. Op-level oracles must call the reference's own modules **[V]**

`h @ W.t()` is not the same computation as `F.linear(h, W)`. cuBLAS picks a
different kernel for the transposed view, and that difference alone put an
otherwise-correct replay of block 0 **1.5e+03** away from the real block output
at production shape. Reimplementing an operator in order to tap it defeats the
point of tapping it.

Likewise `F.rms_norm(x.float(), shape, w.float(), eps).to(bf16)` is not
`F.rms_norm(x, shape, w, eps)`.

Always self-test a replay against the real module output and refuse to report
numbers if it is not bit-exact. `oracle_block_prod.py` does; that check is what
caught both of the above.

---

### 18. The text encoder runs in fp32; the DiT runs in bf16 **[V]**

Same golden, same weights, only the compute dtype changed:

| compute | cos | rel_rms | std_ratio |
|---|---|---|---|
| fp32 | 1.000000000 | 1.8e-06 | 1.0000 |
| bf16 | 0.999985483 | 1.1e-02 | 0.9901 |

If the reference had run bf16, our bf16 would be the closer of the two. It is
worse by four orders of magnitude, and the bf16 error is a *systematic* 1%
deficit in magnitude rather than symmetric noise — cosine stays at 0.99999
while `std_ratio` drops to 0.99, which is the signature of a scale error, not
of rounding.

So `TextEncoder` defaults to fp32 while `H3Transformer` defaults to bf16. The
mechanism inside ComfyUI is not traced; the measurement is the fact.

Nearly free: the weights stay bf16 in memory and are upcast per op, moving peak
residency 48.8 -> 51.8 GB.

### 19. The conditioning encoder has no final norm and no lm_head **[V]**

Not "we skip them" — they are absent from the checkpoint. 902 tensors:
`model.embed_tokens.weight`, 50 layers, and the vision tower. Nothing else.
`text_cond` is the raw layer-50 hidden state, and it is byte-identical to the
`layers.49.0` tap.

The file's own metadata says so: `h3_language_layers: 50`. A port handed an
untruncated 64-layer Qwen3-VL would run 14 layers too many and apply a final
norm that H3 never sees.

### 20. Encoder attention is causal, and its RoPE theta is 5e6 **[V]**

It is a decoder being used as an encoder, so the mask is `triu(1)`, not none.
`rope_theta` is 5,000,000 for the VL variant, not the 1,000,000 of plain
Qwen3 — and the interleaved mRoPE the VL config also carries does **not**
engage for text: it needs three rows of position ids, which only image prompts
produce. Text gets `arange(S)` as a single row and plain split-half RoPE over
all 128 head channels.

### 21. Conditioning-row noise is a recorded input, not a seed **[V]**

`_cond_video_rows` mixes each visual condition with noise at `aug = 0.999`, and
draws it from `torch.Generator("cpu").manual_seed(seed)` **restarted for every
condition** — inside the model, not in the sampler. Audio sits at exactly 1.0,
so the `aug < 1.0` guard never fires and audio conditioning is never noised.

MLX has its own PRNG and will never emit those bytes. So, exactly as
`FixedNoise` already does for sampler noise: **a seed is not a shared input, a
tensor is**. `emit_cond_noise.py` produces the tensor from that one line of
torch — no model, no GPU — and `generate --cond-noise` consumes it.

**Measured portability.** The emitter was run on the CUDA box and on Apple
silicon, same torch 2.11.0, same seed, shapes 405x96 and 1024x96:

* **61% of values bit-identical**; the rest differ by at most **1.9e-06**
  (`rel_rms` 1.8e-07).
* MT19937 is integer arithmetic and is platform-independent — the uniform
  stream matches. Only the last bits of the normal transform move.
* At the 0.001 mixing weight that is **1.9e-09** on a conditioning row, against
  a measured equivalence class of 4.9e-03 for the same span: about **2e+06
  times smaller**.

So the emitter can run anywhere; it does not have to run on the reference box.
Note this is a statement about the *noise*, not a general licence — the same
test on a transform with more amplification could easily come out differently.

Without `--cond-noise` the rows are still produced, from MLX's PRNG, and the run
warns on stderr that it is outside the contract. Producing them silently would
read as parity.

### 22. The video VAE is a conv encoder and a ViT decoder **[V]**

They are not mirror images and porting one teaches you nothing about the other.
`EncoderFCN3D` is a 6-level causal-conv ResNet stack (`encoder.down.N.block.M.*`,
116 tensors); `ViT3DDecoder` is a transformer (440 tensors). Confirmed against
`minimax_h3_video_vae_fp16.safetensors`: 562 tensors, prefixes `encoder`,
`decoder`, `quant_conv`, `post_quant_conv`, `latents_mean`, `latents_std`.

Five details in `encode` are easy to miss and every one is silent when wrong.
All five were wrong in the first draft of the port and all five are now pinned
by `parity/goldens/vae_encode`:

* Input is renormalised **[-1,1] -> [0,1] -> ImageNet mean/std**, not fed raw.
* `encode` takes the **mean** of the moments. It does not sample. `logs_proj`
  is in the checkpoint and the reference never calls it — see #23.
* `group_norm_3d` is `eps=1e-6`, and it is *temporally isolated*: time is folded
  into the batch so statistics are per frame. In a channels-last framework this
  is also the wrong axis by default — MLX's `GroupNorm` normalises the last
  axis, the reference's the second.
* `Downsample3D` pads its extra row and column with **reflect**, not zeros.
* `CausalConv3d` reflects on the spatial axes and pads time **front-only at
  twice the nominal width**.

**Tiling is the normal path, not a fallback.** `tiling=True` in the reference's
own constructor and `tile_size` is 256. Measured: 256x256 is one untiled call;
**864x480 is fifteen** — a 5x3 grid, each tile a full pass over the conv stack,
seams cross-faded in latent space with the overlap rounded to whole `vae_ratio`
units. A single-shot pass over the whole frame differs everywhere, not just at
the seams, because the causal and reflect padding land in different places.

Still not ported: `encode_temporal`, the 17-frame chunker. So the encoder covers
**single frames at any resolution** — keyframes and reference images, which is
what the DiT payload path needs — and refuses multi-frame clips rather than
guessing.

### 23. Both VAE encoders return the posterior mean; neither samples **[V]**

`logs_proj` (audio) and the second half of the moments (video) are present in
the checkpoints and discarded at inference. The reference comment says so
outright for audio. An encoder that reparameterises produces a different latent
on every call, which is both wrong and unfalsifiable — no golden can pin it.

Encoding is therefore **deterministic and seed-independent**, which is worth
stating next to #21: conditioning-*augmentation* noise is unreproducible, but
conditioning *encoding* is exact.

Three more audio-side details, each of which was wrong before it was measured:

* `Snake1d` stores alpha **raw**, shape `[1, C, 1]`, and passes it as beta too.
  `SnakeBeta` — the decoder's activation — is the one that stores log scale.
  Exponentiating `Snake1d`'s alpha is a plausible and completely wrong guess.
* The three `ResidualUnit`s in a block run at dilation **1, 3, 9**, with padding
  `3 * dilation`. Running them all at dilation 1 leaves every shape correct.
* `CausalAttention` has **8 heads** (head dim 256) and finishes with
  `adaptive_avg_pool1d(mean_over_heads(x), 32)`. Picking 64 heads so the head
  dim lands on 32 and dropping the pool gives the right shape and wrong values.
* `EncoderBlock`'s downsample padding is `ceil(stride/2)`, which agrees with
  `(kernel - stride)/2` for even strides and is wrong by one at stride 5.
* `encode` right-pads the waveform to a multiple of the 800-sample hop.

Measured against `parity/goldens/vae_encode` (fp32 port vs the reference's fp16
video / fp32 audio): audio 13/13 taps, worst cos 0.99999972; video untiled 21/21,
worst cos 0.99994465 at `norm_out` — the loosest point in the stack because it
is the one tensor that is normalised to near-zero mean, and `conv_out`
immediately after it returns to 1.7e-03. Video tiled at 864x480: cos 0.99999566.

### 24. H3 is batch-size-1 by construction, so CFG cannot be batched **[V]**

Not a policy choice, not a limitation of this port, and not something a wrapper
can work around.

`comfy/ldm/minimax/model.py:509` raises `ValueError("MiniMax H3 supports batch
size 1")` before anything else runs. Below that, the stack has nowhere to put a
second sample: a DiT block takes `[S, hidden]` where `S` is the **packed
sequence**, and `H3Attention.forward` reads `s = x.shape[0]` as sequence length.
Batch was collapsed away by the packing design — text, audio and video already
share one axis.

So classifier-free guidance is **two sequential forwards**, and the cost is real
rather than an implementation detail to optimise away: measured at 864x480x158,
30.6 min without guidance and 57.4 min with. `--cfg-scale 1.0` skips the second
pass instead of multiplying by one.

A corollary worth stating, because the shapes tempt otherwise: batch support in
`H3Attention`, `H3MLP` and `SplitHalfRoPE` is unreachable through the DiT.
`packedForward` builds `[S, hidden]`, and the reference would reject a batched
input even if it did not. Those code paths are exercised only by
`Tests/H3ModelTests/BatchTests.swift`.

### 25. A keyframe goes through **both** the video VAE and the vision tower **[V]**

They are not alternatives, and assuming they were is the easiest mistake here to
make. `comfy_extras/nodes_minimax_h3.py` builds its keyframe list and then hands
*the same images* to `clip.tokenize(prompt, images=images)`. So one fl2va
keyframe becomes:

* a `cond` segment of the packed sequence, via the **video VAE**, and
* a `<Picture 1>: ` vision block in the **text conditioning**, via the tower.

Encode it only into the packed sequence and the DiT is conditioned on a prompt
that never mentions the picture it is being asked to start from. Vision-block
positions land in the text span with modality tag **0**, which is why
`ModulationIndex` takes `textTags` at all.

351 of the checkpoint's 902 tensors, under `visual.*`. Details that are silent
when wrong, all now pinned by `parity/goldens/vision`:

* `patch_embed.proj` is a `Conv3d` whose stride equals its kernel, applied to
  inputs that are already one patch each — so it is a matmul, and `[1152, 3, 2,
  16, 16]` flattens to `[1152, 1536]`, the width of a `flatten_patches` row.
* Tokens are **not in raster order**. A 9-way permutation interleaves 2x2 merge
  blocks so each consecutive run of 4 tokens is one block, which is what the
  merger's reshape assumes. The position embeddings and the RoPE row/col indices
  must be built in that same order.
* `pos_embed` is a learned **48x48** grid, bilinearly resampled to the image's
  patch grid — not an interpolation of the image.
* The two merger kinds differ in where the norm sits, and the checkpoint says
  so: `merger.norm.weight` is `[1152]` (normalise each patch *before* the
  merge), the three `deepstack_merger_list.N.norm.weight` are `[4608]`
  (normalise the merged vector *after*). Same field names, opposite semantics.
* Normalisation is mean/std **0.5**, i.e. [-1, 1]. Qwen2.5-VL used CLIP
  statistics; swapping them produces plausible garbage.
* GELU is the tanh approximation throughout.

Verified at 26/26 taps, worst cos 0.99999632, with `flatten_patches` **bit
exact**. The second case is 224x320 on purpose: a square image cannot tell a
correct implementation from one that has transposed h and w in the position
grid, the RoPE index, or the merge permutation.

### 26. An image prompt needs two more things the language stack does not do

The tower is complete; the *pipeline* it feeds is not. Both missing pieces are
in the language stack, and neither is reachable through a pure-text prompt —
which is why #20 could say "mRoPE does not engage for text" and be right.

1. **Interleaved mRoPE.** `rope_dims = [24, 20, 20]` summing to 64 = half of the
   128 head dim, and `interleaved_mrope = True`. Position ids become three rows
   (t, h, w); T frequencies are the default and H/W *replace every third
   dimension* — `slice(1, 60, 3)` and `slice(2, 60, 3)` — rather than occupying
   contiguous sections. The non-interleaved `mrope_section` path in the same
   function is the Qwen2-VL layout and is the wrong one here.
2. **DeepStack injection.** After language layer `i` for `i < 3`, the tower's
   three merged feature sets are *added* to the hidden state at the image's
   token positions only: `x[visual_pos_masks] += deepstack_embeds[i]`. Prefill
   only.

Plus tokenizer templating for the `<|vision_start|>` / `<|image_pad|>` /
`<|vision_end|>` span, and the position-id arithmetic in
`qwen2vl_mrope_position_ids`, which advances text positions past an image by
`max(grid)//2` rather than by the image's token count.


### 27. The image-prompt presentation is not chat-templated **[V]**

Raw prompt and label text, no special tokens, with vision blocks spliced in
(`comfy/text_encoders/minimax.py`):

    t2va    <prompt>
    fl2va   "<Picture 1>: " <vision> ["<Picture 2>: " <vision>] <prompt>
    ref2va  per condition in request order, 1-based ordinals per type

A vision block is `<|vision_start|>` (151652), the tower's merged embeddings,
then `<|vision_end|>` (151653). `<Picture 1>: ` is **ordinary text** — it
tokenises like any other string, it is not a special token.

`minimax_token_tags` is 1 for text and **0 for the whole vision block, the
flanking start/end tokens included** — the reference widens the span by one on
each side, so the tag run is `size + 2` long, not `size`.

Two things in the language stack engage only when an image is present, which is
why #20 could say mRoPE does not engage for text and be right:

* **Interleaved mRoPE.** `rope_dims = [24, 20, 20]`, summing to half the 128
  head dim. T is the default across the whole band; h and w then *replace every
  third dimension* — h at 1, 4, ... 58 and w at 2, 5, ... 59, both stopping at
  `rope_dims[axis] * 3`. Contiguous `mrope_section` is the Qwen2-VL layout and
  is present in the same function; it is the wrong branch here.
* **DeepStack injection.** After language layers 0, 1 and 2 the tower's
  layer-8/16/24 features are *added* at the image's token positions only.

In `qwen2vl_mrope_position_ids`, text after an image resumes at
`start + max(grid)/2` — the image advances the clock by its **merged extent**,
not by its token count, so the running offset goes negative. Inside the span,
t is constant, h is the merged row index and w the merged column. The h row's
repeat count is `ceil(size / merged_h)`, i.e. the merged **width** — it equals
`merged_h` only on a square image, so a square fixture cannot tell the two
apart. `parity/goldens/image_prompt` carries a 224x320 case for exactly that.

**Tap placement matters when reading these goldens.** The reference's per-layer
hook fires on the layer module's return, and DeepStack injection happens in the
enclosing loop *after* it. Record the tap after injecting and layers 0-2
disagree with the golden by exactly the injected term while every later layer
matches — which looks like an error that heals, and errors do not heal.

### 28. `encode_temporal` pads by repeating the LAST frame, and drops tokens from the tail **[V]**

The video VAE's multi-frame path is three steps, and each one has a plausible
wrong answer that keeps every shape correct:

1. Pad the frame axis up to a multiple of `clip_length` (17) by **repeating the
   last frame**. Zero-padding or reflecting gives the same latent shape.
2. Encode each 17-frame clip **independently**. The clips do not overlap and are
   not blended — unlike the spatial tiles, and unlike `decode_temporal`, which
   *does* overlap. Symmetry with the decoder is the trap.
3. Drop `token_drop` (3) tokens from the **tail after concatenation**, not per
   clip. With one clip the two are identical, which is why a single-clip fixture
   cannot catch it. `parity/goldens/temporal_encode` carries 1-, 3- and 5-clip
   cases for exactly this reason.

`encode` normalizes pixels *before* chunking, so the clips never see raw
pixels, and it returns the posterior mean of the concatenated moments.

Verified: `h3-parity temporal-encode-check`, 21 taps, worst cos 0.999973.

### 29. A reference video is truncated by the generation length **[V]**

`MiniMaxH3ReferenceToVideo` does `frames = frames[:frame_count]` before it does
anything else, then trims again to `n % 17 == 5`. A reference video longer than
the clip you are generating is silently cut down.

**The two lattices round in opposite directions**, and they are three lines
apart in the same file. `align_frame_count` walks the *generation* length **up**
(`n += 1`); the reference-video trim walks **down** (`n -= 1`). Copying either
loop to the other site is a one-character bug with no shape consequence at all:
`encode_temporal` pads its tail by repeating the last frame (contract 28), so an
off-lattice reference count just encodes manufactured frames as if they were
footage.

This is a **fixture** hazard more than a runtime one. Capturing ref2va at
`--length 5` reduces a 56-frame reference to 5 frames, which Qwen then sees as
a single frame pair rather than three, and which the VAE encodes as one clip
rather than four. Every tap still looks healthy. The first capture in this tree
did precisely that and had to be discarded.

### 30. A video's paired soundtrack is labelled BEFORE the video **[V]**

In the ref2va presentation the reference emits, in order: all images, then all
videos, then standalone audio. But a video's own soundtrack contributes its
`<Audio j>` label **immediately before** its `<Video k>` label — so with one
image, one video-with-sound and one standalone clip the ordinals run
`<Picture 1>`, `<Audio 1>`, `<Video 1>`, `<Audio 2>`, and the audio that is
*part of* the video is numbered ahead of the video.

On the DiT side the same pairing appears as a `ref_audio` segment packed
immediately before its `ref_img` segment, both sharing a cursor origin.

Two more rules in the same span:

* **`<Audio j>: ` is a label and nothing else.** Audio never enters Qwen; its
  latents ride in the DiT payload. Skipping the label because there is no
  embedding to splice shifts every position after it.
* **A block's timestamp is the midpoint of its two frames, at one decimal.**
  `"%.1f" % 0.25` is `"0.2"`, not `"0.25"` and not `"0.3"`. The string is
  tokenised, so a different string is a different sequence length.

Verified: `h3-parity ref2va-check`, 135 taps over 7 cases, 0 failing.

### 31. `_resize` is PIL Lanczos on uint8 — it quantises before it resizes **[V]**

Every keyframe and every reference image goes through `_resize`, which is
`comfy.utils.common_upscale(..., "lanczos", crop)`, which is
`PIL.Image.Resampling.LANCZOS` applied to a **uint8** array. The reference
therefore quantises to 8 bits before resizing, and the resampler is PIL's, not
a framework bilinear.

This is not portable by approximation, so it is not ported. Every fl2va and
ref2va check in this tree is fed the golden's **post-resize** pixels
(`resized.*`), the same rule `FixedNoise` applies to sampler noise. That keeps
the whole chain below it gated and leaves exactly one named gap, rather than
smearing an unmeasured resize error through every downstream tap.

The consequence for claims: file-in fl2va/ref2va cannot claim parity until this
is ported. Tensor-in can.

### 32. A degenerate equivalence class is not a gate **[V]**

The class arms in this tree are "the reference at bf16" versus "the reference at
fp32". For a span with no vision tower and no bf16 module in it — a text-only
prompt, a standalone audio reference — those two arms are **bit-identical** and
the measured class is exactly zero.

Zero does not mean "the reference is exact here, so the port must be too". It
means this pair of arms exercised no variation, so it bounds nothing. Gating on
it demands bit-equality between CUDA and MLX, which is the one thing every other
contract here says not to require.

`ref2va-check` detects `rel_rms == 0` and reports such taps with their numbers,
explicitly ungated, rather than failing them or borrowing another case's class.

### 33. The candidate writer must stream, or it costs 3x the bundle **[V]**

`Safetensors.write` originally accumulated every tensor into a `blob`, appended
that into an `out`, and wrote `out`. Counting the caller's own dictionary that
is **three full copies** of the bundle. At production shape the candidate is
33.4 GB, so writing it cost ~100 GB on top of a 64.8 GB resident checkpoint.

The failure mode is worth knowing independently of the fix, because it is a
good impression of a memory leak and is not one:

* RSS is **flat** for the entire forward — 117 consecutive 10 s samples at
  64.8 GB, about 19.5 minutes — and the whole blow-up arrives in the last
  ~25 seconds, after the model is done.
* A leak grows monotonically. A burst at the end of a run is the *writer*, or
  something else that runs once at the end. Look there first.
* It killed `--keep-ada-lnfp32-resident` at production shape three times out of
  three (SIGKILL, free memory to zero, compressor to 157.7 GB) while the
  baseline survived on the same machine — not because the baseline was doing
  anything different, but because it had ~48 GiB more headroom to absorb the
  same avoidable cost.

Now it streams: `data_offsets` are computable from the shapes alone, so the
header is built before any bytes move and each tensor goes straight through a
`FileHandle`.

**Two things that break when a writer streams**, both pinned in
`SafetensorsWriteTests`:

1. **Header order and payload order are now separate code.** They must both be
   the sorted key order. If they diverge the file is structurally perfect —
   every tensor present, every shape right, every offset resolving — and every
   tensor's bytes sit under another tensor's name. Nothing downstream catches
   that, so the test compares values *by name*.
2. **Overwriting no longer truncates for free.** `Data.write(to:)` replaced the
   file; a `FileHandle` opened on an existing longer file leaves the old tail
   behind, and a stale tail is invisible because the header still resolves. The
   writer removes the file first.

### 34. Audio rms is pinned at 0.2 by the decoder — it measures nothing **[V]**

The audio VAE decode ends with:

```
outStd = max(sqrt(unbiased_var) * 5.0, 1.0)
outWave = outWave / outStd
```

For any raw standard deviation at or above 0.2, that divisor makes the output
standard deviation **exactly 0.2**:

| raw std | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|---|---|---|
| output std | 0.050 | 0.100 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |

So "audio rms is 0.203" is a statement about the normalizer, not the model. Two
completely different prompts producing near-identical audio rms is the expected
behaviour, not a finding — and it was written up as a finding once already (see
`H3_PARITY.md`, MPS investigation, item 6, now retracted).

To say anything about audio content, look at the **spectrum**, not the level:
spectral flatness separates tonal content from noise, and left/right
correlation separates real stereo from a duplicated mono channel. A healthy MLX
render measured 2026-08-04 came out at flatness 0.0000 with 81% of its energy in
80-250 Hz and L/R correlation +0.738 — none of which the rms could have told
you.

The same warning applies to the whole class: **do not read quality off a
summary statistic that a normalisation step controls.** This is the audio twin
of the already-documented "mean/std said low contrast while the frames were
structureless — look at pixels".

### 35. Judge render quality only inside the trained envelope **[V]**

`generate` prints, and means, this:

```
warning: 107 frames is outside the trained range 124-362.
Output is out of distribution — ask for 6 s or more.
```

A render at 107 frames and 6 steps produced visible green blobs and severe
temporal flicker — 52 flagged events, worst per-tile luminance jump **0.559**
between adjacent frames of a shot prompted as static. It reads exactly like a
decoder or tiling bug.

It is neither. The same prompt at **124 frames and 20 steps** — the bottom of
the trained range, the shape the parity gate uses — is clean:

| | 6 steps, 107 frames | 20 steps, 124 frames |
|---|---|---|
| frame-to-frame luma sd | 0.0087 | **0.0009** |
| flash / blob events | 52 | **0** |
| green-excess peak | 6.85% of frame | **0.00%** |

Two rules follow, and the first one cost an evening:

1. **Do not diagnose the port from an out-of-distribution render.** The warning
   is not advisory. Frame count below 124 or a step count far under 20 produces
   artifacts that mimic real bugs and will send you into the VAE looking for
   something that is not there.
2. **The decoder was never a suspect worth chasing.** `decode-check` reproduces
   the production golden's decoded frames at cos 0.999999978. A span with a
   gate that tight does not also produce visible flashing; if the pixels look
   wrong and the gate is green, suspect the latent — which means steps, shape,
   or prompt — before the code.

### 36. At the canvas size the unported LANCZOS resize is the identity **[M]**

Contract 31 says the reference resizes reference media with PIL LANCZOS on
uint8 and that reproducing it elsewhere is its own job. That reads as a wall
across every file-in render. It is not, and the reason is worth writing down
because it is the difference between "ref2va cannot be run end to end" and "run
it, just size the files first".

`adapt_canvas(864, 480)` proposes **1376x768** — a 768-short-edge canvas capped
at 768x1344 pixels. The node then checks whether the source is *smaller* than
that proposal, and if it is, throws the proposal away and uses the source's own
32-aligned size instead:

```python
cw, ch = adapt_canvas(vw, vh)
if vw * vh < cw * ch:
    cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
```

864x480 = 414,720 px against 1,056,768, so a reference video already at the
verified render size resizes to itself. Reference *images* land the same way
from the other direction: at `ref_image_size="match"` the scale is
`sqrt((width * height) / (w * h))`, which is exactly 1.0 when the source is
already the generation's pixel area.

So the rule for real renders is: **hand the renderer reference media already at
864x480 (or 480x864), resized outside the process**, and the unported span never
executes — in either implementation. `MediaLoad.videoHWC` enforces the multiple
of 32 and prints the `ffmpeg -vf scale=W:H:flags=lanczos` line rather than
silently substituting CoreGraphics, which would be a different resampler wearing
the same name.

This does **not** retire contract 31. An arbitrary-sized asset still needs the
real Lanczos, and `ref2va-check` still runs on the golden's recorded post-resize
pixels. It narrows the gap to "assets you did not size yourself".
