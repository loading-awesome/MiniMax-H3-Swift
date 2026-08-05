# ADR 0005: Checkpoints and media are untrusted inputs

- Status: accepted
- Date: 2026-08-05

## Context

Safetensors headers control offsets, shapes, and allocations. Media containers
are parsed by system frameworks. Checkpoints are commonly downloaded from
external stores and exceed 100 GB, making failure late in a render expensive.

## Decision

Preflight validates existence, type, size, header bounds, checkpoint identity,
output capacity, and input/output aliasing before model load. Safetensors
offset and shape arithmetic is checked for overflow and out-of-file access.
Cryptographic checkpoint manifests are supported by the model-set boundary;
verification is explicit in receipts.

## Consequences

Malformed inputs fail before expensive work with stable diagnostics. Integrity
verification may take time for very large files, so verified manifests can be
cached by size and modification identity in a later phase.
