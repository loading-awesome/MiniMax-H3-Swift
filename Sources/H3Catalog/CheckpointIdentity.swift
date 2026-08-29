// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import H3Foundation

/// What a checkpoint file actually is, read out of its safetensors header.
///
/// Every field here is established from `__metadata__` or from tensor shapes.
/// Nothing is inferred from the filename, because filenames are the least
/// reliable thing about this model family: `MiniMax-H3-FL2VA-pruned_bf16` is
/// not pruned, `…_int8_convrot` does not stay int8 in memory, and the same
/// `_bf16` suffix covers two mutually incompatible tensor layouts.
package struct CheckpointIdentity: Sendable, Equatable {

    /// Which conversion produced the file. Determines the fused-attention
    /// layout, and getting it wrong is silent — see `Vendor`.
    package enum Vendor: String, Sendable {
        case comfyOrg = "Comfy-Org"
        case deepBeepMeep = "DeepBeepMeep"
    }

    /// Which of the two DiT partitions this is.
    ///
    /// **This is the field the experimental tree ignored.** FL2VA and Ref2VA
    /// are separate 66.28 GB checkpoints with an identical architecture and
    /// different weights: FL2VA serves text-to-video and first/last-frame,
    /// Ref2VA serves image / video / audio references. Loading FL2VA for a
    /// reference render produces a picture, produces no error, and produces it
    /// from weights that were never trained to consume reference blocks.
    package enum Partition: String, Sendable {
        case fl2va = "FL2VA"
        case ref2va = "Ref2VA"
    }

    /// How the weights are stored, which is not how they are held.
    package enum Quantisation: Sendable, Equatable {
        case none
        /// Per-output-channel symmetric int8: `I8` weights with `F32`
        /// `weight_scale [out, 1]`, plus ComfyUI's `comfy_quant` descriptor.
        case int8ConvRot
        /// quanto: `_data` I8 with `_scale` BF16, plus input/output scales.
        case quanto
    }

    /// What kind of checkpoint this is, inferred from tensor names.
    ///
    /// This matters because **vendor identification is a DiT-only concern**.
    /// The two conversions differ in how they store fused attention weights,
    /// which is a property of `blocks.N.attn.qkv_proj` and of nothing else. A
    /// text encoder or a VAE with no recognised vendor metadata is not
    /// suspicious; the first version of this check reported the Qwen encoder as
    /// UNIDENTIFIED VENDOR and was wrong to.
    package enum Kind: String, Sendable {
        case dit, textEncoder, vae, unknown
    }

    package let url: URL
    package let kind: Kind
    package let vendor: Vendor?
    package let partition: Partition?
    package let quantisation: Quantisation
    /// True when AdaLN has been replaced by the low-rank curve approximation.
    package let isApproximate: Bool
    package let approximationDetail: String?
    package let sizeBytes: UInt64
    package let tensorCount: Int
    /// A distilled checkpoint's own denoising steps, out of 1000, when it
    /// declares them. Nil for the base model, which follows the flow schedule.
    ///
    /// These are a property of the weights, not a preference: a four-forward
    /// distill is trained to jump between specific noise levels and cannot be
    /// asked for twenty steps. The checkpoint is the only thing that knows,
    /// which is why it travels here rather than in a flag.
    package let distilledSteps: [Int]?

    /// Reads the header only — a few hundred kilobytes, not the 66 GB body.
    ///
    /// `headerOnly` matters more than it looks: identification runs over every
    /// configured checkpoint at startup, and mapping four 66 GB files to read
    /// their metadata would make `h3 doctor` a memory event.
    package static func identify(url: URL) throws -> CheckpointIdentity {
        let header: Safetensors.Archive
        do {
            header = try Safetensors.Archive(url: url, headerOnly: true)
        } catch {
            throw H3Error.unreadable(path: url.path, reason: String(describing: error))
        }
        let meta = header.metadata

        // Vendor is decided by which metadata keys are present, which is
        // unambiguous where the tensor statistics are not: the two bf16
        // conversions have the same shape, dtype, mean and standard deviation
        // to eight decimal places and differ only in the arrangement of fused
        // attention weights. See FRAGILE_CONTRACTS.md #9.
        var vendor: Vendor?
        if meta["config"] != nil {
            vendor = .comfyOrg
        } else if meta["repo_id"] != nil && meta["partition"] != nil {
            vendor = .deepBeepMeep
        }

        let partition = (meta["partition"]).flatMap(Partition.init(rawValue:))

        var quant: Quantisation = .none
        if meta["quantization_format"] == "int8_convrot" { quant = .int8ConvRot }
        else if meta["quantization_format"] == "quanto" { quant = .quanto }
        else if header.tensors.keys.contains(where: { $0.hasSuffix(".weight_scale") }) {
            quant = .int8ConvRot
        } else if header.tensors.keys.contains(where: { $0.hasSuffix(".weight._scale") }) {
            quant = .quanto
        }

        // The curve approximation announces itself twice: in metadata, and in
        // the shape of the AdaLN projection, which drops from [96768, 2688] to
        // [96768, rank]. Checking the shape as well as the metadata means a
        // re-exported file without metadata is still caught.
        var approximate = meta["adaln_curve_rank"] != nil || meta["adaln_curve_grid"] != nil
        var detail: String? = approximate
            ? "AdaLN replaced by a rank-\(meta["adaln_curve_rank"] ?? "?") curve"
            : nil
        if let shape = header.tensors["blocks.0.adaln_proj.linear.weight"]?.shape,
           shape.count == 2, shape[1] < 2688 {
            approximate = true
            detail = detail ?? "AdaLN projection is [\(shape[0]), \(shape[1])] rather than "
                             + "[\(shape[0]), 2688] — a low-rank approximation"
        }

        // Kind comes from the tensor names, which cannot be renamed without
        // breaking the loader, unlike metadata which is optional everywhere.
        let keys = header.tensors.keys
        let kind: Kind
        if keys.contains(where: { $0.hasPrefix("blocks.") && $0.contains(".attn.qkv_proj") }) {
            kind = .dit
        } else if keys.contains(where: { $0.hasPrefix("visual.") || $0.hasPrefix("model.layers.")
                                         || $0.hasPrefix("model.visual.") }) {
            kind = .textEncoder
        } else if keys.contains(where: { $0.hasPrefix("encoder.") || $0.hasPrefix("decoder.") }) {
            kind = .vae
        } else {
            kind = .unknown
        }

        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0

        let distilledSteps = meta["dmd_denoising_steps"]
            .map { $0.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) } }
            .flatMap { $0.isEmpty ? nil : $0 }

        return CheckpointIdentity(url: url, kind: kind, vendor: vendor, partition: partition,
                                  quantisation: quant, isApproximate: approximate,
                                  approximationDetail: detail, sizeBytes: size,
                                  tensorCount: header.tensors.count,
                                  distilledSteps: distilledSteps)
    }

    /// Refuses a checkpoint that cannot be used safely for `mode`.
    ///
    /// Two independent refusals, both of which the experimental tree lacked:
    /// an unidentifiable vendor (silently wrong attention), and a partition
    /// that does not match the requested mode (silently untrained conditioning).
    package func validate(forMode mode: RenderMode,
                         allowApproximate: Bool) throws {
        // Only the DiT carries fused attention weights, so only the DiT has a
        // layout that can be silently misread.
        if kind == .dit && vendor == nil {
            throw H3Error.unidentifiedCheckpoint(
                url: url,
                detail: "no recognised __metadata__ (expected Comfy-Org's `config` or "
                      + "DeepBeepMeep's `repo_id`/`partition`)")
        }
        if let partition, partition != mode.requiredPartition {
            throw H3Error.wrongPartition(needed: mode.requiredPartition.rawValue,
                                         loaded: partition.rawValue,
                                         mode: mode.rawValue)
        }
        if isApproximate && !allowApproximate {
            throw H3Error.approximateWeightsNotPermitted(
                variant: url.lastPathComponent,
                detail: approximationDetail ?? "weights differ from the released ones")
        }
    }

    package var summary: String {
        var bits = [String]()
        if kind == .dit {
            bits.append(vendor?.rawValue ?? "UNIDENTIFIED VENDOR")
        } else {
            bits.append(kind.rawValue)
        }
        if let partition { bits.append(partition.rawValue) }
        switch quantisation {
        case .none: bits.append("bf16")
        case .int8ConvRot: bits.append("int8_convrot")
        case .quanto: bits.append("quanto int8")
        }
        if isApproximate { bits.append("APPROXIMATE") }
        return String(format: "%-52@ %6.2f GB  %4d tensors  %@",
                      url.lastPathComponent as NSString, Double(sizeBytes) / 1e9,
                      tensorCount, bits.joined(separator: " / "))
    }
}

/// What the caller asked for, which decides which partition is required.
///
/// Modelled as an enum rather than a set of booleans so "text-to-video with a
/// reference image" cannot be expressed. The reference rejects anchors and
/// references in one payload, and a type that cannot represent the invalid
/// combination is a better guard than a validation function that must remember
/// to check for it.
public enum RenderMode: String, Sendable, CaseIterable {
    case textToVideo = "t2va"
    case firstLastFrame = "fl2va"
    case reference = "ref2va"

    package var requiredPartition: CheckpointIdentity.Partition {
        switch self {
        // FL2VA serves text-to-video as well as the keyframe modes; there is
        // no separate T2VA partition.
        case .textToVideo, .firstLastFrame: .fl2va
        case .reference: .ref2va
        }
    }
}
