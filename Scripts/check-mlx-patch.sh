#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
# Fail the build when the MLX large-M GEMM patch is not in the dependency
# checkout.
#
#     ./Scripts/check-mlx-patch.sh            verify, and fail if it is missing
#     ./Scripts/check-mlx-patch.sh --apply    apply it, then verify
#
# `patches/mlx-m3-ultra-large-m-gemm.patch` routes large bf16/fp16 matmuls with
# M >= 8192 on Ultra-class devices to the `32x64x16, 1x2sg` Steel kernel MLX
# already ships. It is worth 8.6% of block time and it is bit-identical — see
# section 14 of docs/PERF_ROADMAP.md for the measurement.
#
# **It lives in `.build/checkouts`, which is gitignored, so nothing in this
# repository can hold it.** `swift package reset`, `swift package update`, a
# clean checkout or a fresh clone all drop it, and every one of those is a
# routine thing to do. Nothing then fails: the build succeeds, the tests pass,
# the numbers are bit-identical, and the binary is 8.6% slower for the rest of
# its life. That is the entire reason this check exists — the failure has no
# other symptom.
#
# The patch is Ultra-class tuning and a machine that will never take that branch
# does not need it. Set H3_ALLOW_UNPATCHED_MLX=1 to downgrade the failure to a
# warning; the release script is where it matters, because that is the build
# that gets handed to someone else.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patch_file="$root/patches/mlx-m3-ultra-large-m-gemm.patch"
# The patch's paths are `a/mlx/backend/metal/matmul.cpp`, so they are relative
# to the vendored mlx submodule rather than to the mlx-swift checkout above it.
mlx_root="$root/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
target="$mlx_root/mlx/backend/metal/matmul.cpp"

apply=0
[ "${1:-}" = "--apply" ] && apply=1

fail() {
    echo "" >&2
    echo "$1" >&2
    if [ -n "${H3_ALLOW_UNPATCHED_MLX:-}" ]; then
        echo "" >&2
        echo "  continuing: H3_ALLOW_UNPATCHED_MLX is set." >&2
        exit 0
    fi
    echo "" >&2
    echo "  fix:    ./Scripts/check-mlx-patch.sh --apply" >&2
    echo "  ignore: H3_ALLOW_UNPATCHED_MLX=1 (this build will be ~8.6% slower)" >&2
    exit 1
}

[ -f "$patch_file" ] || {
    echo "missing $patch_file" >&2
    exit 1
}

# A check that passes because it could not find what it was checking is worse
# than no check: it reports the state it was asked about without ever having
# observed it. Unresolved dependencies are their own message.
[ -f "$target" ] || fail "The mlx-swift checkout is not there yet, so the GEMM patch cannot be
verified: $target

Run 'swift package resolve' first, then this again."

present() { git -C "$mlx_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; }
applies()  { git -C "$mlx_root" apply --check "$patch_file" >/dev/null 2>&1; }

if [ "$apply" -eq 1 ]; then
    if present; then
        echo "mlx large-M GEMM patch: already applied"
        exit 0
    fi
    applies || fail "The GEMM patch does not apply to this checkout, and is not already in it.

The pinned mlx-swift revision has probably moved under it — check
Package.resolved against the revision section 14 of docs/PERF_ROADMAP.md
was measured on, and re-cut the patch if upstream has changed matmul.cpp."
    git -C "$mlx_root" apply "$patch_file"
    echo "mlx large-M GEMM patch: applied"
fi

if present; then
    echo "mlx large-M GEMM patch: present"
    exit 0
fi

# Distinguish the two ways it can be absent. "Apply it" and "re-cut it against
# a moved dependency" are different jobs, and a check that says only "missing"
# sends you looking for the wrong one.
if applies; then
    fail "The mlx large-M GEMM patch is NOT applied to the dependency checkout.

A build without it is correct and about 8.6% slower per block, with nothing
in the output to say so. It applies cleanly, so it was almost certainly lost
to a 'swift package reset', an update, or a fresh clone."
else
    fail "The mlx large-M GEMM patch is neither applied nor applicable.

It does not reverse-apply (so it is not in the checkout) and does not apply
forward (so the file it patches has changed). The pinned mlx-swift revision
has probably moved: check Package.resolved, and re-cut the patch against
mlx/backend/metal/matmul.cpp if upstream now tunes this branch itself — in
which case this whole check should be deleted rather than fixed."
fi
