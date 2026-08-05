# Render operations

## Lifecycle

`RenderEngine` admits one job per process and an advisory user-scoped file lock
prevents a second CLI process or engine instance from loading another model.
Concurrent admission is refused with `H3-4002`; it is never silently queued.

The job exposes typed events and cooperative cancellation. Cancellation is
checked before expensive phases, between sampler steps, between decoders, and
while hashing checkpoint manifests. AVFoundation mux callbacks cannot be
interrupted safely once final encoding has begun.

## Preflight

Before checkpoint bodies are loaded, the engine checks:

- request and render-policy validity;
- Metal-library availability;
- input readability and input/output aliasing;
- existing-output policy and output-directory writability;
- tokenizer completeness;
- safetensors header bounds, shapes, offsets, and overlap;
- DiT partition and approximation policy;
- optional SHA-256 checkpoint identities;
- current available memory with configured margin;
- same-volume working disk capacity.

## Publication and recovery

Media is written beneath `.h3-<job-id>` in the final video's directory. Sidecar
audio is promoted first and the mp4 is promoted last as the transaction commit
marker. Each move is same-volume and atomic. Existing files are protected unless
`overwriteOutput` or `--overwrite` is explicit.

The pipeline always stages a recovery WAV. It is discarded after a successful
audio mux when no sidecar was requested, and published as `*.recovery.wav` if
muxing or later output work fails.

## Receipts

`<video-base>.h3-receipt.json` is written atomically on success, failure, or
cancellation whenever the output directory is writable. Schema version 1
contains:

- job and terminal status;
- dimensions, duration, steps, seed, and quality profile;
- prompt character count, never prompt contents;
- OS, machine, available memory, and library version;
- checkpoint filenames, sizes, and verified SHA-256 values when supplied;
- phase timings, output filenames and sizes, warnings, and stable error code.

Absolute media input paths and prompt text are intentionally absent.

## Memory pressure

Critical macOS memory pressure flips the render cancellation token. The render
stops at the next safe boundary and writes a terminal receipt. This is not a
promise of instant cancellation inside one multi-minute model forward; tearing
down an in-flight MLX graph is not a safe recovery mechanism.
