# MiniMax-H3-Swift — enterprise hardening plan

Updated 2026-08-05. This is the execution plan for the product repository.
Porting history and CUDA investigation remain in `/Volumes/scratch_disk/H3_Swift`;
they are evidence, not the product architecture.

## Definition of done

Version 1.0 is ready when a consumer can import only `MiniMaxH3`, submit a
validated request to an actor-isolated engine, observe and cancel the job, and
receive atomically published media plus a machine-readable receipt. Every
fragile numerical contract has an owner and enforcement tier. A clean checkout
builds and runs its CPU suite without weights or a GPU; numerical and
production-shape suites are separate release gates.

## Decisions

The binding decisions are recorded under `docs/adr/`:

- one supported public module and a library-first product;
- actor-owned, single-render execution;
- faithful numerics by default, approximations explicitly selected;
- three conformance tiers with immutable fixture provenance;
- checkpoint and media inputs treated as untrusted;
- atomic outputs, structured events, and durable render receipts.

## Work phases

| phase | outcome | acceptance gate |
|---|---|---|
| 0 | decisions, boundary inventory, contract ledger | ADRs accepted; no unresolved architectural ambiguity |
| 1 | stable `MiniMaxH3` facade and quality profiles | consumer imports one module; no MLX type crosses it |
| 2 | actor-isolated jobs and structured concurrency | concurrent work follows the declared policy; cancellation is tested |
| 3 | enforceable contract and conformance tiers | contracts 1...36 accounted for with provenance and an enforcement tier |
| 4 | preflight, atomic output, receipts, logging, recovery | every terminal job has media or a diagnostic receipt; partial output is never published |
| 5 | supply-chain and release engineering | reproducible signed artifacts, SBOM, compatibility checks |
| 6 | measured distribution and performance profiles | every optimization carries memory, speed, numerical, and quality evidence |

## Boundary inventory

The package graph remains useful: MLX-free policy and identity code stays cheap
to test, while MLX implementation targets remain replaceable. The visibility
model changes. `MiniMaxH3` is the supported product surface. Other targets are
implementation modules and may use `package` access internally; their current
`public` declarations are not compatibility promises.

The supported API consists of:

- render request, result, progress/event, receipt, and stable errors;
- model-set and engine configuration values;
- `RenderEngine`, `RenderJob`, cancellation, and explicit quality profiles.

The following are not API: MLX arrays, model layers, tap recorders, packed-row
implementation, checkpoint readers, attention kernels, and pipeline phases.

## Current baseline

- Debug build succeeds with Swift 6.3.3.
- 65 tests pass across policy, catalog, geometry, memory planning, attention,
  render validation, recipes, and Metal-library discovery.
- The retained conformance target previously declared zero covered contracts;
  Phase 3 replaces that placeholder with an auditable ledger and tests.
- Existing uncommitted README and `Doctor.swift` edits pre-date this work and
  must be preserved.

## Deferred beyond Phase 4

Resident quantised matmul, model acquisition, streaming VAE decode, signed
release artifacts, and production service deployment are deliberately outside
this slice. They must build on the API, lifecycle, and evidence model above
rather than defining a second one.
