import Foundation
import H3Catalog
import MLX
import H3Foundation

/// MLX-backed weight access for an H3 checkpoint.
///
/// Two-layer by design: `H3Core.CheckpointInventory` validates structure from
/// the header alone (instant on 62 GiB, no MLX), and this type provides the
/// tensors. Validation happens first and cheaply, so a wrong or truncated
/// checkpoint fails with a named mismatch instead of an allocation.
///
/// The DiT ships bf16 — the model's native precision, not a downcast. There is
/// no fp32 DiT to fall back to, so bf16 IS the reference and matching it is the
/// target. Keeping weights bf16 also matters for residency: fp32 would be
/// 132 GB.
package final class H3Weights {
    package let url: URL
    package let inventory: CheckpointInventory
    package let config: H3Config

    private let lock = NSLock()
    private var cache: [String: MLXArray] = [:]
    private var all: [String: MLXArray]?

    package enum Error: Swift.Error, CustomStringConvertible {
        case invalid([String])
        case missing(String)
        package var description: String {
            switch self {
            case .invalid(let p): "checkpoint failed validation:\n  " + p.joined(separator: "\n  ")
            case .missing(let n): "checkpoint has no tensor named \(n)"
            }
        }
    }

    /// Validates before touching data. `strict` refuses a checkpoint whose
    /// derived architecture disagrees with the reference — the default, because
    /// silently proceeding on a mismatched checkpoint wastes far more time than
    /// it saves.
    package init(url: URL, strict: Bool = true) throws {
        self.url = url
        let archive = try Safetensors.Archive(url: url)
        let inv = try CheckpointInventory(archive: archive,
                                          claimedName: url.lastPathComponent)
        if strict, !inv.isValid { throw Error.invalid(inv.problems) }
        self.inventory = inv
        self.config = inv.derived
    }

    /// Header-only view: shapes and dtypes without materialising anything.
    package func describe(_ name: String) throws -> Safetensors.TensorInfo {
        try Safetensors.Archive(url: url).info(name)
    }

    /// Materialises the checkpoint's arrays. MLX keeps them lazy and
    /// mmap-backed, so this is not a 62 GiB copy — but it is still the moment
    /// the process commits to the file, so it is deliberate rather than
    /// implicit in a subscript.
    package func loadAll() throws {
        lock.lock(); defer { lock.unlock() }
        if all == nil { all = try MLX.loadArrays(url: url) }
    }

    package func tensor(_ name: String) throws -> MLXArray {
        lock.lock(); defer { lock.unlock() }
        if let a = cache[name] { return a }
        if all == nil { all = try MLX.loadArrays(url: url) }
        guard var a = all?[name] else { throw Error.missing(name) }
        if inventory.vendor.needsQKVPermute, name.hasSuffix("attn.qkv_proj.weight") {
            a = Self.permuteQKV(a, heads: config.numHeads, headDim: config.headDim)
        }
        cache[name] = a
        return a
    }

    /// (heads, 3, headDim, hidden) -> (3, heads, headDim, hidden).
    ///
    /// DeepBeepMeep interleaves Q/K/V per head; Comfy-Org (and the model code)
    /// expect them blocked. Without this the projection returns the right
    /// shape, the right std, and the wrong answer — measured cos 0.029 against
    /// the reference, i.e. uncorrelated.
    static func permuteQKV(_ w: MLXArray, heads: Int, headDim: Int) -> MLXArray {
        let hidden = w.dim(1)
        return w.reshaped([heads, 3, headDim, hidden])
                .transposed(1, 0, 2, 3)
                .reshaped([3 * heads * headDim, hidden])
    }

    package subscript(name: String) -> MLXArray? { try? tensor(name) }

    /// Block-scoped accessor: `w.block(0, "attn.qkv_proj.weight")`.
    package func block(_ index: Int, _ suffix: String) throws -> MLXArray {
        try tensor("blocks.\(index).\(suffix)")
    }

    package func has(_ name: String) -> Bool {
        (try? Safetensors.Archive(url: url).info(name)) != nil
    }
}
