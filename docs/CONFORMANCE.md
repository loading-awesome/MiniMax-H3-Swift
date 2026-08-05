# Conformance strategy

MiniMax-H3's dangerous regressions are usually shape-correct. Conformance is
therefore split by cost rather than reduced to one test command.

## Tier 1 — CPU contracts

Runs on every commit with no model weights or GPU. It covers geometry, schedule,
packing, request policy, checkpoint identity, memory planning, recipes,
safetensors safety, API lifecycle, and the completeness of the 36-contract
ledger in `H3Conformance`.

Command:

```bash
swift build --build-tests
./Scripts/bootstrap-metal.sh
swift test --skip-build
```

The bootstrap is currently needed because SwiftPM links all test products into
one bundle, including MLX tests. The CPU contracts themselves do not execute
MLX.

## Tier 2 — MLX numerical fixtures

Runs on Apple Silicon with the matching `mlx.metallib`. Small recorded fixtures
cover the text encoder, vision path, conditioning presentation, VAE boundaries,
one-block operators, attention backends, and sampler behavior. Fixtures must
record their source runtime, checkpoint identity, shape, dtype, capture script,
and SHA-256.

Large fixture bodies are not committed. A manifest without an immutable digest
is not a fixture.

## Tier 3 — release parity

Runs before a release that changes model math, MLX, Metal kernels, checkpoint
loading, media presentation, or sampling. It includes the production
864x480x124x20 trajectory, the operator matrix, conditioning modes, CFG, and the
no-golden perceptual oracles retained under `Tools/`.

Tolerance classes travel with the fixture and shape that measured them. A
zero-width class is reported as ungated, never treated as bit-exact evidence.

## Ownership

`H3Conformance.contracts` is the authoritative ledger. Its completeness test
requires IDs 1 through 36 exactly once, with an owner, tier, mechanism, and
evidence reference. Adding contract 37 requires adding its ledger entry in the
same change.
