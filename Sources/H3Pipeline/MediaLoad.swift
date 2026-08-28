// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import AVFoundation
import CoreGraphics
import ImageIO
import MLX
import H3Foundation

/// Decoding media files into the tensors the VAE encoders expect.
///
/// This is the one span with no golden behind it, and it is worth being blunt
/// about why: there is no reference tap for "this JPEG becomes these pixels".
/// The VAE encoders are pinned against `parity/goldens/vae_encode`, which feeds
/// them recorded pixel tensors — so everything downstream of `pixels` is
/// verified and everything upstream of it is convention. The conventions here
/// are chosen to match what the reference pipeline feeds its VAE: sRGB, linear
/// [-1, 1], no alpha, no colour management beyond what CoreGraphics does when
/// it draws into a device RGB context.
/// One bit of mutable state for `AVAudioConverter`'s `@Sendable` input block.
/// The block runs synchronously on this thread, so no locking is needed — it
/// only has to be a reference type to be capturable.
private final class FlagBox: @unchecked Sendable {
    var done = false
}

/// `AVAudioConverter` marks its synchronous pull callback `@Sendable` even
/// though the buffer is consumed before `convert` returns. This immutable box
/// documents and contains that imported-framework mismatch.
private final class AudioBufferBox: @unchecked Sendable {
    let value: AVAudioPCMBuffer
    init(_ value: AVAudioPCMBuffer) { self.value = value }
}

// `package` rather than internal so the external oracles in Tools/ can
// decode the same audio the pipeline does, rather than carrying a second
// decoder that could disagree with it.
package enum MediaLoad {
    enum Error: Swift.Error, CustomStringConvertible {
        case unreadable(String, String)
        case noTrack(String, String)
        var description: String {
            switch self {
            case .unreadable(let p, let why): "cannot read \(p): \(why)"
            case .noTrack(let p, let kind): "\(p) has no \(kind) track"
            }
        }
    }

    /// Image file -> `[1, 3, 1, H, W]` in [-1, 1].
    ///
    /// - Parameter fit: when given, the image is scaled to exactly these
    ///   dimensions. Keyframes must land on the target grid because they share
    ///   the video's spatial position ids; reference images keep their own size
    ///   and pass `nil`.
    ///
    /// Both axes are rounded down to a multiple of 32 — the VAE downsamples by
    /// 16 and the DiT patchifies by 2, so an off-grid size produces a latent
    /// the packed layout cannot describe.
    static func image(at path: String, fit: (width: Int, height: Int)? = nil) throws -> MLXArray {
        var w = fit?.width, h = fit?.height
        if w == nil || h == nil {
            let s = try imageSize(at: path)
            // Both axes down to a multiple of 32: the VAE downsamples by 16 and
            // the DiT patchifies by 2, so an off-grid size produces a latent the
            // packed layout cannot describe.
            w = max(32, s.width / 32 * 32)
            h = max(32, s.height / 32 * 32)
        }
        // [1, H, W, 3] in [0,1] -> [1, 3, 1, H, W] in [-1,1]
        return (try imageHWC(at: path, width: w!, height: h!) * 2.0 - 1.0)
            .squeezed(axis: 0)
            .transposed(2, 0, 1)                            // [3, H, W]
            .expandedDimensions(axes: [0, 2])               // [1, 3, 1, H, W]
    }

    /// Pixel dimensions without decoding the image body.
    static func imageSize(at path: String) throws -> (width: Int, height: Int) {
        let url = URL(fileURLWithPath: path)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else {
            throw Error.unreadable(path, "could not read image dimensions")
        }
        return (w, h)
    }

    /// Image file -> `[1, H, W, 3]` in **[0, 1]**, scaled to exactly `width` x
    /// `height`. This is the layout and range the vision tower's preprocessing
    /// expects; the VAE path converts from here.
    static func imageHWC(at path: String, width w: Int, height h: Int) throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw Error.unreadable(path, "not a decodable image")
        }
        // Draw into a known layout rather than trusting the source's: a skipped
        // alpha channel gives tightly packed RGBX bytes in device RGB.
        let bytesPerRow = w * 4
        var buffer = [UInt8](repeating: 0, count: bytesPerRow * h)
        guard let ctx = buffer.withUnsafeMutableBytes({ raw in
            CGContext(data: raw.baseAddress, width: w, height: h,
                      bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }) else {
            throw Error.unreadable(path, "could not create an RGB drawing context")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        return (MLXArray(buffer, [h, w, 4]).asType(.float32) / 255.0)[0..., 0..., 0 ..< 3]
            .expandedDimensions(axis: 0)                    // [1, H, W, 3]
    }

    /// Video file -> `([T, H, W, 3]` in **[0, 1]**, nominal fps)` at the file's
    /// own resolution and frame rate.
    ///
    /// **Nothing is resampled, in either axis, and that is deliberate.** Both
    /// rates are contracts the caller has to satisfy before the pixels get
    /// here:
    ///
    ///  * The reference resizes reference video with PIL LANCZOS on uint8
    ///    (`comfy.utils.lanczos`), which is not ported. But at the canvas size
    ///    that resize is the identity — `adapt_canvas(864, 480)` proposes
    ///    1376x768, the source is smaller, so the node falls back to the
    ///    source's own 32-aligned size. A file already on the grid therefore
    ///    skips the unported step *in both implementations* rather than having
    ///    it approximated here. Anything off the grid is refused, with the
    ///    ffmpeg line that fixes it, because a CoreGraphics rescale would be a
    ///    silently different resampler.
    ///  * Qwen's 2 fps sampling is `range(0, n, FPS // 2)` with a hardcoded
    ///    `FPS = 24`. Frames are taken by index, so a 30 fps file would be
    ///    stamped `<0.5 seconds>` for something 0.4 s in.
    static func videoHWC(at path: String, maxFrames: Int? = nil) async throws
        -> (frames: MLXArray, fps: Double) {
        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw Error.noTrack(path, "video")
        }
        let size = try await track.load(.naturalSize)
        let w = Int(size.width.rounded()), h = Int(size.height.rounded())
        guard w % 32 == 0 && h % 32 == 0 else {
            throw Error.unreadable(path, "\(w)x\(h) is not a multiple of 32 on both axes. "
                + "The reference would resize this with PIL LANCZOS, which is not ported, "
                + "so resize it yourself and keep the resampler out of the parity boundary: "
                + "ffmpeg -i \(path) -vf scale=\(w / 32 * 32):\(h / 32 * 32):flags=lanczos out.mp4")
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String:
                                kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        var planes: [Float] = []
        planes.reserveCapacity(w * h * 3 * (maxFrames ?? 32))
        var count = 0
        while let sample = output.copyNextSampleBuffer() {
            if let cap = maxFrames, count >= cap { break }
            guard let buf = CMSampleBufferGetImageBuffer(sample) else { continue }
            CVPixelBufferLockBaseAddress(buf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(buf) else { continue }
            let stride = CVPixelBufferGetBytesPerRow(buf)
            let bytes = base.assumingMemoryBound(to: UInt8.self)
            for y in 0 ..< h {
                let row = bytes + y * stride
                for x in 0 ..< w {
                    // BGRA on disk, RGB out.
                    planes.append(Float(row[x * 4 + 2]) / 255.0)
                    planes.append(Float(row[x * 4 + 1]) / 255.0)
                    planes.append(Float(row[x * 4 + 0]) / 255.0)
                }
            }
            count += 1
        }
        guard count > 0 else { throw Error.unreadable(path, "decoded no frames") }
        return (MLXArray(planes, [count, h, w, 3]),
                Double(try await track.load(.nominalFrameRate)))
    }

    /// Audio file -> `[1, 2, L]` in [-1, 1], resampled to 32 kHz stereo.
    ///
    /// Mono is duplicated across both channels rather than left silent on one
    /// side; anything above two channels keeps the first two.
    package static func audio(at path: String) throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) }
        catch { throw Error.unreadable(path, error.localizedDescription) }

        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: Double(H3Audio.sampleRate),
                                         channels: 2, interleaved: false) else {
            throw Error.unreadable(path, "could not build a 32 kHz stereo format")
        }
        let source = file.processingFormat
        guard let converter = AVAudioConverter(from: source, to: target) else {
            throw Error.unreadable(path, "no conversion from \(source) to 32 kHz stereo")
        }

        // Ceiling on the resampled length, plus a little slack for the
        // converter's own latency.
        let ratio = target.sampleRate / source.sampleRate
        let capacity = AVAudioFrameCount(Double(file.length) * ratio) + 4096
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            throw Error.unreadable(path, "could not allocate a \(capacity)-frame buffer")
        }

        // Read the whole file up front rather than pulling chunk by chunk. The
        // converter's input block is `@Sendable`, so anything it mutates has to
        // be a reference type; handing it one finished buffer keeps the only
        // mutable state to a single flag.
        guard let whole = AVAudioPCMBuffer(pcmFormat: source,
                                           frameCapacity: AVAudioFrameCount(file.length)) else {
            throw Error.unreadable(path, "could not allocate a \(file.length)-frame read buffer")
        }
        do { try file.read(into: whole) }
        catch { throw Error.unreadable(path, error.localizedDescription) }

        let fed = FlagBox()
        let sourceBuffer = AudioBufferBox(whole)
        let status = converter.convert(to: out, error: nil) { _, outStatus in
            if fed.done { outStatus.pointee = .endOfStream; return nil }
            fed.done = true
            outStatus.pointee = .haveData
            return sourceBuffer.value
        }
        guard status != .error, out.frameLength > 0, let data = out.floatChannelData else {
            throw Error.unreadable(path, "conversion produced no samples")
        }

        let n = Int(out.frameLength)
        let left = Array(UnsafeBufferPointer(start: data[0], count: n))
        let right = target.channelCount > 1 && source.channelCount > 1
            ? Array(UnsafeBufferPointer(start: data[1], count: n)) : left
        return concatenated([MLXArray(left, [1, 1, n]), MLXArray(right, [1, 1, n])], axis: 1)
    }
}
