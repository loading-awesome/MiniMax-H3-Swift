#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
# Put MLX's Metal kernels where MLX will find them.
#
# `swift build` does not produce mlx-swift's default.metallib: the Metal kernels
# are excluded from the Cmlx target and compiled by mlx-swift's Xcode project
# instead. Without this step the build succeeds, the binary links, and the first
# GPU operation dies with an untyped C++ error carrying no path.
#
# MLX looks for `mlx.metallib` beside the running binary before it falls back to
# `default.metallib` in the current working directory. Beside-the-binary is the
# one worth targeting: it survives the user running the tool from somewhere else.
#
# Run after `swift build` and after `swift build --build-tests`. Cheap and
# idempotent, so running it every time is fine.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/Resources/mlx.metallib"

if [ ! -f "$src" ]; then
    echo "missing $src" >&2
    echo "It is version-coupled to the mlx-swift revision pinned in Package.resolved." >&2
    exit 1
fi

shopt -s nullglob
copied=0

# Both the executables and the test bundle's own MacOS directory: under
# `swift test` the running binary is the .xctest bundle's executable, so a copy
# in .build/debug alone does not help it.
for dir in "$root"/.build/*/debug "$root"/.build/*/release; do
    [ -d "$dir" ] || continue
    cp -f "$src" "$dir/mlx.metallib"
    copied=$((copied + 1))
    for bundle in "$dir"/*.xctest; do
        cp -f "$src" "$bundle/Contents/MacOS/mlx.metallib"
        copied=$((copied + 1))
    done
done

if [ "$copied" -eq 0 ]; then
    echo "no build directories found — run 'swift build' first" >&2
    exit 1
fi

echo "mlx.metallib -> $copied location(s)"

# The GEMM patch check rides along here rather than in a script of its own that
# nobody remembers to run. This is already the step that turns a bare
# `swift build` into a working one, so it is the only place a development build
# reliably passes through. It runs last: the metallib copy above is this
# script's actual job and should not be held up by a performance concern.
"$root/Scripts/check-mlx-patch.sh"
