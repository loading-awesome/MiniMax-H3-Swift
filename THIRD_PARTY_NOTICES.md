# Third-party notices

This repository's own source is Apache-2.0 (see [LICENSE](LICENSE) and
[NOTICE](NOTICE)). It also **redistributes a compiled third-party binary** and
depends on packages that carry their own terms. Both are listed here, because a
dependency is a link and a redistributed artifact is an obligation.

## Redistributed in this repository

### `Resources/mlx.metallib`

MLX's compiled Metal kernel library, 2.9 MB, checked in and copied beside every
built binary by `Scripts/bootstrap-metal.sh`. It is here because SwiftPM does
not build it — mlx-swift compiles its Metal kernels in its Xcode project, not on
the command line — and without it the first GPU operation of any render fails.
See `H3Hardware.MetalLibrary`.

It is a build artifact of MLX, not of this project:

```
MIT License
Copyright © 2023 Apple Inc.
```

Full text: <https://github.com/ml-explore/mlx/blob/main/LICENSE>

The MIT licence requires that this copyright notice travel with the binary. It
is version-coupled to the mlx-swift revision pinned in `Package.resolved`;
bumping one means replacing the other, and replacing it means checking this
notice still describes what is in the file.

## Package dependencies

Fetched by SwiftPM, not redistributed here. Listed so a downstream consumer can
see the whole licence surface without resolving the graph themselves.

| package | licence | holder |
|---|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | MIT | ml-explore |
| ├ mlx (vendored C++ core) | MIT | Apple Inc. |
| ├ mlx-c | MIT | Apple Inc. |
| ├ metal-cpp | Apache-2.0 | Apple Inc. |
| ├ fmt | MIT | Victor Zverovich |
| └ nlohmann/json | MIT | Niels Lohmann |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | Apache-2.0 with Runtime Library Exception | Apple Inc. |

All of the above are permissive and compatible with Apache-2.0 distribution of
this work.

## Model weights

**Not in this repository, and not covered by its licence.** The checkpoints are
published by MiniMax and licensed separately; `h3 doctor` resolves them from
paths you configure and this project never redistributes them. Their terms are
between you and MiniMax.

## Provenance of the port

This is a clean-room Swift/MLX implementation written against the published
reference behaviour, with numerical parity established by measurement — 225
gating taps inside CUDA-measured equivalence classes at production shape. No
reference source is vendored or copied here. Tensor names, layout conventions
and the documented contracts in `FRAGILE_CONTRACTS.md` describe an interoperable
file format and the observable behaviour required to match it.
