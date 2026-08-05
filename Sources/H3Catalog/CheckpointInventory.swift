// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import H3Foundation

/// Structural validation of an H3 checkpoint, from the safetensors header only.
///
/// No tensor data is read, so this is instant on the 62 GiB DiT and needs no
/// MLX. It does what `comfy/model_detection.py:362` does: **derive** the
/// architecture from tensor shapes rather than assume it. Every constant in
/// `H3Config` is re-derived here and compared, so a checkpoint that disagrees
/// with our assumptions is caught before a single tensor is multiplied — and a
/// mis-detected checkpoint (wrong partition, wrong quant, truncated download)
/// surfaces as a named mismatch rather than as garbage output.
package struct CheckpointInventory: Sendable {
    package let tensorCount: Int
    package let dtypes: [String: Int]
    package let totalBytes: Int
    package let derived: H3Config
    package let blockCount: Int
    package let refinerBlockCount: Int
    package let partition: Partition
    package let problems: [String]

    /// FL2VA serves text-to-video and first/last-frame; Ref2VA is
    /// reference-to-video. Same architecture, different weights — the file name
    /// is the only external hint, so we record what was claimed and let the
    /// caller cross-check.
    package enum Partition: String, Sendable { case fl2va, ref2va, unknown }

    /// Which conversion produced this file. **This is not cosmetic**: the two
    /// bf16 conversions store the fused `attn.qkv_proj.weight` in DIFFERENT
    /// memory layouts, and loading one as the other produces plausible-looking
    /// output with ~0 correlation to the reference.
    ///
    ///     Comfy-Org      (3, heads, headDim, hidden)   blocked  Q | K | V
    ///     DeepBeepMeep   (heads, 3, headDim, hidden)   per-head interleaved
    ///
    /// Same values, same std/mean to 8dp, different arrangement — so a shape,
    /// dtype or statistics check will NOT catch it. Detected by metadata:
    /// Comfy-Org writes only `config`; DeepBeepMeep writes `repo_id`,
    /// `partition`, `precision`, `source_revision`.
    package enum Vendor: String, Sendable {
        case comfyOrg, deepBeepMeep, unknown

        /// Does `attn.qkv_proj.weight` need permuting into Comfy-Org order?
        package var needsQKVPermute: Bool { self == .deepBeepMeep }

        package static func detect(metadata: [String: String]) -> Vendor {
            if metadata["repo_id"] != nil || metadata["partition"] != nil { return .deepBeepMeep }
            if metadata["config"] != nil { return .comfyOrg }
            return .unknown
        }
    }

    package var vendor: Vendor = .unknown

    package var isValid: Bool { problems.isEmpty }

    /// H3 is identified exactly as ComfyUI identifies it: the presence of both
    /// patch projections. This is the cheapest possible "is this even H3".
    package static func looksLikeH3(_ names: some Collection<String>) -> Bool {
        let s = Set(names)
        return s.contains("video_patch_proj.weight") && s.contains("audio_patch_proj.weight")
    }

    package init(archive: Safetensors.Archive, claimedName: String = "") throws {
        let t = archive.tensors
        var problems: [String] = []

        self.tensorCount = t.count
        var dt: [String: Int] = [:]
        var bytes = 0
        for (_, i) in t {
            dt[i.dtype, default: 0] += 1
            bytes += i.byteCount
        }
        self.dtypes = dt
        self.totalBytes = bytes

        guard Self.looksLikeH3(t.keys) else {
            throw Safetensors.Error.badHeader(
                "not a MiniMax-H3 DiT: video_patch_proj.weight / audio_patch_proj.weight absent")
        }

        func shape(_ name: String) -> [Int]? { t[name]?.shape }
        func need(_ name: String) -> [Int] {
            guard let s = shape(name) else {
                problems.append("missing tensor \(name)")
                return []
            }
            return s
        }

        // block counts: highest index + 1, and flag any gap
        func count(prefix: String) -> Int {
            var idx = Set<Int>()
            for k in t.keys where k.hasPrefix(prefix) {
                let rest = k.dropFirst(prefix.count)
                if let dot = rest.firstIndex(of: "."), let n = Int(rest[rest.startIndex ..< dot]) {
                    idx.insert(n)
                }
            }
            guard let mx = idx.max() else { return 0 }
            if idx.count != mx + 1 {
                problems.append("\(prefix) indices are not contiguous 0..<\(mx + 1) "
                                + "(found \(idx.count)) — truncated or corrupt checkpoint")
            }
            return mx + 1
        }
        self.blockCount = count(prefix: "blocks.")
        self.refinerBlockCount = count(prefix: "token_refiner.blocks.")

        var c = H3Config()
        c.numLayers = blockCount
        c.tokenRefinerLayers = refinerBlockCount
        if let s = shape("video_patch_proj.weight") { c.hiddenSize = s[0] }
        if let s = shape("final_layer.video_out.weight") { c.videoLatentDim = s[0] / 4 }
        if let s = shape("final_layer.audio_out.weight") { c.audioLatentDim = s[0] }
        if let s = shape("blocks.0.attn.q_norm.weight") { c.headDim = s[0] }
        let qkv = need("blocks.0.attn.qkv_proj.weight")
        if !qkv.isEmpty, c.headDim > 0 { c.numHeads = qkv[0] / (3 * c.headDim) }
        if let s = shape("blocks.0.mlp.fc1.weight") { c.ffnHidden = s[0] / 2 }
        if let s = shape("condition_proj.weight") { c.textDim = s[1] }
        if let s = shape("rope.inv_freq") { c.ropeInvFreqLen = s[0] }
        if let s = shape("time_embedder.proj_in.weight") {
            c.timestepInputDim = s[1]
            c.timeEmbedHidden = s[0]
        }
        if let s = shape("time_embedder.proj_out.weight") { c.timeEmbedDim = s[0] }
        self.derived = c

        // Cross-check against the documented reference values. A mismatch means
        // either a different checkpoint or a wrong assumption — both worth a
        // loud failure rather than a silent one.
        let expected = H3Config()
        func check(_ label: String, _ got: Int, _ want: Int) {
            if got != want { problems.append("\(label): derived \(got), reference \(want)") }
        }
        check("numLayers", c.numLayers, expected.numLayers)
        check("tokenRefinerLayers", c.tokenRefinerLayers, expected.tokenRefinerLayers)
        check("hiddenSize", c.hiddenSize, expected.hiddenSize)
        check("numHeads", c.numHeads, expected.numHeads)
        check("headDim", c.headDim, expected.headDim)
        check("ffnHidden", c.ffnHidden, expected.ffnHidden)
        check("videoLatentDim", c.videoLatentDim, expected.videoLatentDim)
        check("audioLatentDim", c.audioLatentDim, expected.audioLatentDim)
        check("textDim", c.textDim, expected.textDim)
        check("ropeInvFreqLen", c.ropeInvFreqLen, expected.ropeInvFreqLen)
        check("timeEmbedDim", c.timeEmbedDim, expected.timeEmbedDim)

        self.vendor = Vendor.detect(metadata: archive.metadata)
        if self.vendor == .unknown {
            problems.append("cannot identify the conversion vendor from metadata — "
                            + "qkv layout is ambiguous, refusing to guess")
        }
        let n = claimedName.lowercased()
        self.partition = n.contains("ref2va") ? .ref2va : (n.contains("fl2va") ? .fl2va : .unknown)
        self.problems = problems
    }

    /// Every tensor the model will ask for, derived from the config. Anything
    /// the checkpoint is missing is a load failure waiting to happen; anything
    /// extra usually means a different variant than intended.
    package func expectedTensorNames() -> Set<String> {
        var s: Set<String> = [
            "video_patch_proj.weight", "video_patch_proj.bias",
            "audio_patch_proj.weight", "audio_patch_proj.bias",
            "condition_proj.weight", "condition_proj.bias",
            "rope.inv_freq",
            "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
            "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
            "final_layer.norm.weight",
            "final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias",
            "final_layer.video_out.weight", "final_layer.video_out.bias",
            "final_layer.audio_out.weight", "final_layer.audio_out.bias",
        ]
        for i in 0 ..< refinerBlockCount {
            let b = "token_refiner.blocks.\(i)."
            s.formUnion([
                b + "norm1.weight", b + "norm2.weight",
                b + "attn.qkv_proj.weight", b + "attn.out_proj.weight",
                b + "attn.q_norm.weight", b + "attn.k_norm.weight",
                b + "mlp.fc1.weight", b + "mlp.fc2.weight",
            ])
        }
        s.insert("token_refiner.final_norm.weight")
        for i in 0 ..< blockCount {
            let b = "blocks.\(i)."
            s.formUnion([
                b + "norm1.weight", b + "norm2.weight",
                b + "attn.qkv_proj.weight", b + "attn.out_proj.weight",
                b + "attn.q_norm.weight", b + "attn.k_norm.weight",
                b + "mlp.fc1.weight", b + "mlp.fc2.weight",
                b + "adaln_proj.linear.weight", b + "adaln_proj.linear.bias",
            ])
        }
        return s
    }

    package func coverage(archive: Safetensors.Archive) -> (missing: [String], extra: [String]) {
        let have = Set(archive.tensors.keys)
        let want = expectedTensorNames()
        return (missing: want.subtracting(have).sorted(),
                extra: have.subtracting(want).sorted())
    }

    package var summary: String {
        let gb = Double(totalBytes) / 1e9
        let d = dtypes.map { "\($0.key)x\($0.value)" }.sorted().joined(separator: " ")
        return """
        tensors \(tensorCount) · \(String(format: "%.2f", gb)) GB · \(d)
        vendor \(vendor.rawValue) · partition \(partition.rawValue) · blocks \(blockCount) · refiner \(refinerBlockCount)
        derived: hidden \(derived.hiddenSize) heads \(derived.numHeads)x\(derived.headDim) \
        ffn \(derived.ffnHidden) text \(derived.textDim) \
        latents v\(derived.videoLatentDim)/a\(derived.audioLatentDim)
        """
    }
}
