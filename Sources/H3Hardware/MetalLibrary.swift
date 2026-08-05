import Foundation
import H3Foundation

/// Finding MLX's compiled Metal kernel library, before anything tries to use it.
///
/// **This is a real deployment dependency, and it was discovered the hard way.**
/// `swift build` does not produce MLX's `default.metallib`. The Metal kernels
/// are excluded from the `Cmlx` target ("see PrepareMetalShaders — don't build
/// the kernels in place") and are compiled by mlx-swift's Xcode project, not by
/// SwiftPM on the command line. Nothing warns about this: the build succeeds,
/// the binary links, and the first GPU operation throws a C++ `runtime_error`
/// reading `Failed to load the default metallib. library not found library not
/// found library not found library not found` with no path in it.
///
/// MLX looks in four places, in this order (`mlx/backend/metal/device.cpp`):
///
///  1. `mlx.metallib` **beside the running binary**;
///  2. `Resources/mlx.metallib` beside the running binary;
///  3. `default.metallib` inside an `mlx-swift_Cmlx.bundle` in any loaded
///     bundle — the Xcode path, and empty for a SwiftPM build;
///  4. `default.metallib` **relative to the current working directory**, from
///     the compile-time `METAL_PATH` define.
///
/// The experimental tree survived for months on step 4 alone: a `default.metallib`
/// had been committed at the repo root, and every render and every test was run
/// from that directory. Move the file and MLX stops working; `cd` elsewhere and
/// it stops working. That is invisible while the working directory never
/// changes, and it fails the first time a user runs an installed binary from
/// their home directory.
///
/// So this package targets **step 1**, which depends on where the binary is
/// rather than where the user happens to stand: `Scripts/bootstrap-metal.sh`
/// copies `Resources/mlx.metallib` beside every built binary and into the test
/// bundle. `preflight()` turns a missing copy into an error that says which
/// file is missing, where it was looked for, and which command puts it there.
///
/// The metallib is version-coupled to the pinned mlx-swift revision in
/// `Package.resolved`. Bump one and you replace the other.
public enum MetalLibrary {

    /// The directory holding the binary this code is compiled into.
    ///
    /// **`dladdr`, not `Bundle.main`, and the difference is not pedantic.**
    /// MLX finds itself with `dladdr`; under `swift test` the process's main
    /// bundle is `swiftpm-testing-helper` inside the Xcode toolchain, while the
    /// image containing this code is `…PackageTests.xctest/Contents/MacOS/…`.
    /// Asking `Bundle.main` reports a directory MLX will never look in, so the
    /// preflight failed while MLX itself was loading the library perfectly
    /// well — a check that disagrees with the thing it is checking.
    ///
    /// `#dsohandle` is the current image's handle, so this resolves to the same
    /// binary MLX resolves to whether that is the CLI, a test bundle, or a host
    /// application that has linked this library.
    public static var binaryDirectory: URL? {
        var info = Dl_info()
        guard dladdr(#dsohandle, &info) != 0, let name = info.dli_fname else { return nil }
        return URL(fileURLWithPath: String(cString: name))
            .resolvingSymlinksInPath().deletingLastPathComponent()
    }

    /// Every path MLX will try, in the order it tries them.
    ///
    /// Reported verbatim by `h3 doctor`: a list of places checked is worth more
    /// to somebody stuck than a summary that says "not found".
    public static var searchPaths: [URL] {
        var paths: [URL] = []
        if let dir = binaryDirectory {
            paths.append(dir.appendingPathComponent("mlx.metallib"))
            paths.append(dir.appendingPathComponent("Resources/mlx.metallib"))
        }
        paths.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("default.metallib"))
        return paths
    }

    /// The first path that exists, or nil.
    public static func locate() -> URL? {
        searchPaths.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Whether the located copy is the cwd-relative fallback rather than one
    /// beside the binary.
    ///
    /// Not an error — it is what the experimental tree ran on — but it means
    /// the process works only from this directory, so it is worth saying out
    /// loud rather than letting somebody discover it after they install.
    public static func locatedOnlyViaWorkingDirectory() -> Bool {
        guard let found = locate() else { return false }
        return found.lastPathComponent == "default.metallib"
    }

    /// Throws before the first GPU operation, so the failure names its remedy.
    ///
    /// Call this ahead of any MLX work rather than letting the C++ exception
    /// surface: that one is untyped, unlocalised, and carries no path.
    public static func preflight() throws {
        guard locate() == nil else { return }
        let looked = searchPaths.map { "    \($0.path)" }.joined(separator: "\n")
        throw H3Error.unreadable(
            path: "mlx.metallib",
            reason: "MLX's compiled Metal kernels are not next to this binary, so the "
                  + "first GPU operation would fail with an untyped C++ error. `swift build` "
                  + "does not produce them — mlx-swift compiles its Metal kernels in its "
                  + "Xcode project, not on the SwiftPM command line.\n  looked in:\n\(looked)\n"
                  + "  remedy: run `Scripts/bootstrap-metal.sh` from the package root, which "
                  + "copies Resources/mlx.metallib beside every built binary. For an installed "
                  + "binary, place mlx.metallib in the same directory as the executable.")
    }
}
