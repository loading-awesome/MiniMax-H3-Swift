#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
# Build the artifacts a GitHub release attaches, and optionally sign, notarise
# and staple them.
#
# Unsigned (default) — anyone who downloads this must clear the quarantine flag
# by hand before macOS will run it:
#
#     ./Scripts/package-release.sh
#
# Signed and notarised — no quarantine step for the user at all:
#
#     H3_SIGN_APP="Developer ID Application: Your Name (TEAMID)" \
#     H3_SIGN_PKG="Developer ID Installer: Your Name (TEAMID)" \
#     H3_NOTARY_PROFILE=h3-notary \
#     ./Scripts/package-release.sh
#
# **This script never sees a secret.** `H3_NOTARY_PROFILE` names a keychain
# profile you create once, yourself:
#
#     xcrun notarytool store-credentials h3-notary \
#         --apple-id you@example.com --team-id TEAMID --password <app-specific>
#
# The app-specific password goes into your keychain at that moment and never
# appears here, in the repository, or in a process listing.
#
# Two certificates, not one. "Developer ID Application" signs the binary;
# "Developer ID Installer" signs the .pkg. Both are free with a paid developer
# account and both are created from Xcode: Settings -> Accounts -> Manage
# Certificates -> +.
#
# The hardened runtime is required for notarisation and is free here: measured
# against an unsigned build, 190 s to reach sampling step 3 either way, 60.5
# against 60.4 s/step. MLX compiles its Metal kernels out of process, so no JIT
# entitlement is needed.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

version="$(grep -o 'version = "[^"]*"' Sources/MiniMaxH3/MiniMaxH3.swift | head -1 | cut -d'"' -f2)"
[ -n "$version" ] || { echo "could not read the version from MiniMaxH3.swift" >&2; exit 1; }

arch="$(uname -m)"
name="h3-${version}-macos-${arch}"
stage="dist/${name}"
sign_app="${H3_SIGN_APP:-}"
sign_pkg="${H3_SIGN_PKG:-}"
notary="${H3_NOTARY_PROFILE:-}"

echo "building ${name}"
swift build -c release --disable-sandbox

rm -rf "$stage" "dist/pkgroot"
mkdir -p "$stage"
cp ".build/release/h3" "$stage/h3"
cp "Resources/mlx.metallib" "$stage/mlx.metallib"
cp LICENSE NOTICE THIRD_PARTY_NOTICES.md "$stage/"

# ---------------------------------------------------------------- sign

if [ -n "$sign_app" ]; then
    echo "signing with: ${sign_app}"
    # --options runtime is the hardened runtime, which notarisation requires.
    # --timestamp contacts Apple's timestamp server, so the signature stays
    # valid after the certificate expires.
    codesign --force --options runtime --timestamp \
             --sign "$sign_app" "$stage/h3"
    codesign --verify --strict --verbose=2 "$stage/h3"
else
    echo "NOT signing (set H3_SIGN_APP to sign)"
fi

cat > "$stage/README.txt" <<TXT
h3 — MiniMax-H3 for Apple silicon, ${version}

Keep 'h3' and 'mlx.metallib' in the same folder. h3 looks for the Metal kernels
beside itself and will not start without them.
$( [ -n "$sign_app" ] || printf '\n%s\n' "This build is not signed, so macOS will refuse to open it until you run:

    xattr -dr com.apple.quarantine h3" )
    chmod +x h3

Then:

    ./h3 doctor          what your Mac can run and which model files it found
    ./h3 config init     write a configuration file to fill in
    ./h3 render --help   every rendering option

To run it from anywhere, move both files together:

    sudo mkdir -p /usr/local/lib/h3 && sudo cp h3 mlx.metallib /usr/local/lib/h3/
    sudo ln -sf /usr/local/lib/h3/h3 /usr/local/bin/h3

The symlink is fine — h3 resolves symlinks before looking for the metallib.

Signature: the binary is signed by "Developer ID Application: Tesserapps, LLC",
which is the Apple developer account. The copyright is personal to Sean
Kammerich. Same author, two hats.
TXT

# ---------------------------------------------------------------- zip

# ditto, not zip(1). A bare Mach-O carries its signature embedded in __LINKEDIT,
# so this particular binary would survive either — but ditto is what Apple's
# notarisation documentation specifies, it preserves extended attributes, and it
# is correct for the bundles this will eventually also have to archive. Using
# the tool that is right in general beats using one that happens to work here.
ditto -c -k --keepParent "$stage" "dist/${name}.zip"
echo "dist/${name}.zip  ($(du -h "dist/${name}.zip" | cut -f1))"

# ---------------------------------------------------------------- pkg

# The installer exists because it is the only artifact that can be *stapled*.
# A stapled ticket means the first run works with no network; a notarised zip
# still has to phone Apple, which fails on a plane or behind a strict firewall.
# It also spares a non-technical user the Terminal entirely.
if [ -n "$sign_pkg" ]; then
    mkdir -p "dist/pkgroot/usr/local/lib/h3"
    cp "$stage/h3" "$stage/mlx.metallib" "dist/pkgroot/usr/local/lib/h3/"

    mkdir -p "dist/scripts"
    cat > "dist/scripts/postinstall" <<'POST'
#!/bin/bash
set -e
mkdir -p /usr/local/bin
ln -sf /usr/local/lib/h3/h3 /usr/local/bin/h3
exit 0
POST
    chmod +x "dist/scripts/postinstall"

    pkgbuild --root "dist/pkgroot" \
             --scripts "dist/scripts" \
             --identifier "com.loading-awesome.h3" \
             --version "$version" \
             --install-location "/" \
             --sign "$sign_pkg" \
             "dist/${name}.pkg"
    echo "dist/${name}.pkg  ($(du -h "dist/${name}.pkg" | cut -f1))"
else
    echo "NOT building a .pkg (set H3_SIGN_PKG to build and sign one)"
fi

# ---------------------------------------------------------------- notarise

if [ -n "$notary" ]; then
    for artifact in "dist/${name}.zip" "dist/${name}.pkg"; do
        [ -f "$artifact" ] || continue
        echo ""
        echo "notarising $(basename "$artifact") — this uploads it to Apple and waits"
        xcrun notarytool submit "$artifact" --keychain-profile "$notary" --wait
        # Only a .pkg can carry a stapled ticket. Stapling a zip is not a thing
        # that exists, so the zip relies on Gatekeeper's online check.
        case "$artifact" in
            *.pkg) xcrun stapler staple "$artifact"
                   xcrun stapler validate "$artifact" ;;
            *)     echo "  (zip notarised; tickets cannot be stapled to a zip)" ;;
        esac
    done
else
    echo ""
    echo "NOT notarising (set H3_NOTARY_PROFILE to notarise)"
fi

rm -rf "$stage" "dist/pkgroot" "dist/scripts"

echo ""
if [ -z "$sign_app" ]; then
    echo "This build is unsigned, so every downloader must clear the quarantine"
    echo "flag by hand. See the header of this script to sign and notarise."
fi
