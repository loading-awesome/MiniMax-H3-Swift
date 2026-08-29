#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture the FastH3 reference trajectory on a CUDA pod, for step-level diffing.

Run on a RunPod (or any CUDA) box with >= 80 GB VRAM — H100 80GB works with
layerwise offload, H200/B200 comfortably without:

    git clone https://github.com/hao-ai-lab/FastVideo && cd FastVideo
    git checkout 48a047c05ff4138f20cfa33351499c6ec5945f5d   # the checkpoint's pinned commit
    pip install -e .
    python runpod_capture.py --out /workspace/capture

It renders ONE clip with the Dense 4-step distill at our exact test point —
same prompt, 480x864, 5 s — and monkeypatches the denoising stage to dump,
per step: both latents in, both raw velocities, both timesteps. Plus the
initial noise for both streams, the prompt embeds, the final latents, and the
decoded mp4. Everything as safetensors in one tarball.

Why each item is captured:
  initial noise + prompt embeds   injected into our port so the whole
                                  trajectory is deterministic and comparable
  per-step tensors                localize a divergence to a step and a stream
  final audio latent              decoded through OUR audio VAE locally: if it
                                  sounds clean, our loop is the bug; if it
                                  sounds 'a bit off' exactly like ours, the
                                  checkpoint's 4-step audio is the ceiling
  their decoded mp4               the ear's ground truth
"""
import argparse, os, tarfile, json
from pathlib import Path

import torch
from safetensors.torch import save_file

MODEL = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
PROMPT = ("a woman with freckles talking directly to camera in a sunlit kitchen, "
          "explaining a recipe, natural hand gestures")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/capture")
    ap.add_argument("--offload", action="store_true", help="H100-80GB: layerwise offload")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}

    # --- hook the denoising stage before the pipeline is built -------------
    import fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising as den
    orig_fwd = den.MiniMaxH3DenoisingStage.forward
    step_counter = {"i": 0}

    def record(name, t):
        tensors[name] = t.detach().to(torch.float32).cpu().contiguous()

    # The stage calls self.transformer(...) once per DMD step. Wrap it.
    class TransformerTap:
        def __init__(self, inner): self.inner = inner
        def __getattr__(self, k): return getattr(self.inner, k)
        def __call__(self, *a, **kw):
            i = step_counter["i"]
            record(f"step{i}.video_latent_in", kw["hidden_states"][0])
            record(f"step{i}.audio_latent_in", kw["audio_hidden_states"][0])
            record(f"step{i}.timesteps", kw["timestep"])
            v, aud = self.inner(*a, **kw)
            record(f"step{i}.video_velocity", v[0])
            record(f"step{i}.audio_velocity", aud[0])
            step_counter["i"] += 1
            return v, aud

    def tapped_forward(self, batch, fastvideo_args):
        # initial state, before the loop touches it
        record("initial.video_noise", batch.latents)
        record("initial.audio_noise", batch.audio_latents)
        if getattr(batch, "prompt_embeds", None) is not None:
            pe = batch.prompt_embeds
            record("initial.prompt_embeds", pe[0] if isinstance(pe, (list, tuple)) else pe)
        self.transformer = TransformerTap(self.transformer)
        try:
            result = orig_fwd(self, batch, fastvideo_args)
        finally:
            self.transformer = self.transformer.inner
        record("final.video_latent", batch.latents)
        record("final.audio_latent", batch.audio_latents)
        return result

    den.MiniMaxH3DenoisingStage.forward = tapped_forward

    # --- run the reference pipeline ---------------------------------------
    from fastvideo import VideoGenerator
    from fastvideo.api.schema import GenerationRequest, SamplingConfig, OutputConfig
    kwargs = dict(num_gpus=1)
    if args.offload:
        kwargs["dit_layerwise_offload"] = True
    gen = VideoGenerator.from_pretrained(MODEL, **kwargs)
    request = GenerationRequest(
        prompt=PROMPT,
        sampling=SamplingConfig(
            seed=7, height=864, width=480, num_frames=124, fps=24,
            guidance_scale=1.0),
        output=OutputConfig(output_path=str(out), output_video_name="reference"),
    )
    gen.generate(request)

    save_file(tensors, str(out / "trajectory.safetensors"),
              metadata={"model": MODEL, "prompt": PROMPT, "seed": "7",
                        "commit": "48a047c05ff4138f20cfa33351499c6ec5945f5d"})
    manifest = {k: list(v.shape) for k, v in tensors.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with tarfile.open(out / "capture.tar", "w") as tar:
        for f in ("trajectory.safetensors", "manifest.json", "reference.mp4"):
            tar.add(out / f, arcname=f)
    print(f"capture complete: {out}/capture.tar")
    print(json.dumps(manifest, indent=1)[:1500])

if __name__ == "__main__":
    main()
