# ADR 0004: Three conformance tiers

- Status: accepted
- Date: 2026-08-05

## Context

The model's common failures are silent: layouts, labels, QKV ordering, and
precision can be wrong while every tensor shape remains valid. Large CUDA
goldens cannot live in the product repository or run on ordinary CI.

## Decision

Conformance has three tiers:

1. CPU contracts run on every commit with no weights or GPU.
2. MLX numerical fixtures run on capable Apple Silicon hosts.
3. Production-shape parity and perceptual checks gate releases.

Every fragile contract has an ID, owner, evidence reference, tier, and
enforcement mechanism. Large fixtures are addressed by immutable digest and
kept outside Git.

## Consequences

Ordinary CI remains cheap while releases retain the evidence that justified
the port. A written contract without a mechanism is visible as a ledger gap.
