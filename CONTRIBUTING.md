# Contributing

## Setup: `swift build` is not enough

```bash
swift build --build-tests && ./Scripts/bootstrap-metal.sh && swift test --skip-build
```

`--skip-build` on the test run is not optional: a plain `swift test` relinks the
test bundle, which deletes the `mlx.metallib` the bootstrap just put inside it.
Build, bootstrap, then run.

SwiftPM does not build MLX's Metal kernels. mlx-swift excludes them from its
`Cmlx` target and compiles them in its Xcode project instead, so a command-line
build links cleanly and then dies on its first GPU operation with an untyped
C++ error that names no path:

```
MLX error: Failed to load the default metallib. library not found library not found …
```

MLX's last-resort fallback is `default.metallib` **relative to the current
working directory**, which is why the experimental tree appeared to work for
months: a metallib had been committed at its repo root and everything was run
from there. `Scripts/bootstrap-metal.sh` instead puts `mlx.metallib` beside each
built binary, where MLX looks first and where the answer does not depend on
where the user is standing. `h3 doctor` reports which copy is in play, and
`H3Hardware.MetalLibrary.preflight()` turns a missing one into an error with a
remedy. Re-run the script after any build that relinks the test bundle.

`Resources/mlx.metallib` is version-coupled to the mlx-swift revision pinned in
`Package.resolved`. Bumping one means replacing the other.

### If the suite dies with SIGBUS and no failing test

Delete `.build` and rebuild. Adding or reordering a case in `H3Error` — or any
other enum crossing a module boundary — changes its memory layout, and SwiftPM's
incremental build does not always rebuild every dependent. The symptom is a
crash in `outlined destroy` of whatever holds the error, attributed to a test
that passes in isolation and has nothing to do with the change.

## The house style for comments

Explain **why**, with the measurement. This codebase's failure mode is silent —
a wrong packed layout, a dropped `<Audio>` label, a transposed qkv all keep
every tensor the right shape — so a comment saying what the code does is worth
very little and a comment saying what breaks if you change it is worth a lot.

Good:

```swift
// The last anchor sits on the *aligned* frame count, not the requested
// duration: the request is snapped up onto the 17k+5 lattice before anything
// else sees it.
```

Not useful:

```swift
// Set the last keyframe index.
```

## Errors

Nothing a caller can provoke may trap. `preconditionFailure` in a library takes
down the host application. Every refusal is a `throws` carrying three things:
the rule, the measurement that establishes it, and the remedy.

`precondition` still appears, and the line is this: **if the value came from
the caller, throw; if it came from our own arithmetic or from a checkpoint we
already identified, precondition.** A keyframe index, an image size and a frame
count are the caller's; "AdaLN must expand to 6" and "packed rows must equal the
layout's total" are ours, and if one of those is false the fix is a code change,
not a better error message.

## FRAGILE_CONTRACTS.md

36 entries, each one a thing that breaks parity silently. Adding to it is
expected when a debugging session ends in "…and it must stay that way". A
contract with no corresponding check in `H3Conformance` is written down but not
enforced; prefer to add both.

## Tests

Test through the public path. A test that builds its own `JSONDecoder` is
testing its own decoder — that is exactly how the `video_vae` key bug survived
a green suite, with `h3 doctor` cheerfully listing eight checkpoints and neither
VAE.

## Licensing of contributions

Contributions are accepted under the Apache License 2.0, the licence this
project ships under. Section 5 makes that the default for anything you
deliberately submit, so there is no separate CLA to sign.

If you add a file that came from somewhere else — a vendored binary, a kernel,
a table — it goes in `THIRD_PARTY_NOTICES.md` with its own licence and copyright
holder in the same commit. A dependency is a link; a redistributed artifact is
an obligation.

## Cutting a release

```bash
H3_SIGN_APP="Developer ID Application: NAME (TEAMID)" \
H3_SIGN_PKG="Developer ID Installer: NAME (TEAMID)" \
H3_NOTARY_PROFILE=h3-notary \
./Scripts/package-release.sh
```

Three things have to be true first, and each is a one-time setup:

1. **Two certificates**, not one. `Developer ID Application` signs the binary;
   `Developer ID Installer` signs the `.pkg`. Both come from Xcode: Settings →
   Accounts → Manage Certificates → **+**. An `Apple Development` certificate
   is *not* one of these and cannot be notarised.
2. **A stored notary profile.** Run this yourself; the script never handles a
   secret and there is nothing to paste into the repository:
   ```bash
   xcrun notarytool store-credentials h3-notary \
       --apple-id you@example.com --team-id TEAMID --password <app-specific-password>
   ```
   Generate the app-specific password at appleid.apple.com. It goes straight
   into your keychain.
3. **Attach both artifacts** to the GitHub release. The `.pkg` is the one the
   README points people at.

Why both: only a `.pkg` can carry a **stapled** notarisation ticket, so it works
on a machine with no network. A notarised `.zip` still has to reach Apple to be
checked on first run.

The hardened runtime that notarisation requires costs nothing here — measured
against an unsigned build, 190 s to reach sampling step 3 either way, 60.5
against 60.4 s/step. MLX compiles Metal kernels out of process, so no JIT
entitlement is needed.
