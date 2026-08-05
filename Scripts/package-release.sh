#!/bin/bash
# Build the release archive that gets attached to a GitHub release.
#
# The archive has to contain `mlx.metallib` next to `h3`. This is not optional
# packaging tidiness: SwiftPM does not build MLX's Metal kernels, and MLX looks
# for them beside the running binary. Ship `h3` alone and every download fails
# on its first render with an untyped C++ error naming no file. See
# `H3Hardware.MetalLibrary`.
#
#   ./Scripts/package-release.sh            -> dist/h3-<version>-macos-arm64.tar.gz
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

version="$(grep -o 'version = "[^"]*"' Sources/MiniMaxH3/MiniMaxH3.swift | head -1 | cut -d'"' -f2)"
[ -n "$version" ] || { echo "could not read the version from MiniMaxH3.swift" >&2; exit 1; }

arch="$(uname -m)"
name="h3-${version}-macos-${arch}"
stage="dist/${name}"

echo "building ${name}"
swift build -c release --disable-sandbox

rm -rf "$stage"
mkdir -p "$stage"
cp ".build/release/h3" "$stage/h3"
cp "Resources/mlx.metallib" "$stage/mlx.metallib"
cp LICENSE NOTICE THIRD_PARTY_NOTICES.md "$stage/"

cat > "$stage/README.txt" <<'TXT'
h3 — MiniMax-H3 for Apple silicon

Keep `h3` and `mlx.metallib` in the same folder. h3 looks for the Metal kernels
beside itself and will not run without them.

macOS will refuse to open a downloaded binary until you clear the quarantine
flag. From this folder, once:

    xattr -dr com.apple.quarantine h3
    chmod +x h3

Then:

    ./h3 doctor          what your Mac can run and which model files it found
    ./h3 config init     write a configuration file to fill in
    ./h3 render --help   every rendering option

To run it from anywhere, move both files together:

    sudo mkdir -p /usr/local/lib/h3 && sudo cp h3 mlx.metallib /usr/local/lib/h3/
    sudo ln -sf /usr/local/lib/h3/h3 /usr/local/bin/h3

The symlink is fine — h3 resolves symlinks before looking for the metallib.
TXT

tar -czf "dist/${name}.tar.gz" -C dist "$name"
rm -rf "$stage"

echo ""
echo "dist/${name}.tar.gz  ($(du -h "dist/${name}.tar.gz" | cut -f1))"
echo ""
echo "This binary is unsigned and unnotarised, so every downloader must clear"
echo "the quarantine flag by hand. Signing it with a Developer ID and running"
echo "notarytool would remove that step; the README.txt above documents it."
