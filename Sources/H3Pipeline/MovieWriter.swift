import Foundation
import AVFoundation
import CoreVideo
import Metal
import MLX
import H3Foundation

/// Cursors shared with the two mux queues. Each field is touched by exactly one
/// queue, and both are joined before the values are read.
private final class MuxCursor: @unchecked Sendable {
    var frame = 0
    var sample = 0
}

/// Joins AVFoundation's two required callback queues without blocking a Swift
/// concurrency worker. All mutable state is protected by the lock; the class is
/// the containment boundary for the callback API's non-Sendable objects.
private final class MuxCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private var remaining: Int
    private var firstError: String?
    private var continuation: CheckedContinuation<String?, Never>?

    init(remaining: Int, continuation: CheckedContinuation<String?, Never>) {
        self.remaining = remaining
        self.continuation = continuation
    }

    func finish(error: String? = nil) {
        lock.lock()
        if firstError == nil { firstError = error }
        remaining -= 1
        let result = remaining == 0 ? firstError : nil
        let done = remaining == 0 ? continuation : nil
        if remaining == 0 { continuation = nil }
        lock.unlock()
        done?.resume(returning: result)
    }
}

/// Writing the two streams out — as one mp4, and as a side-car wav.
///
/// H3 generates video and audio jointly, so handing back a silent video and a
/// loose wav throws away the thing that makes the model interesting. The mp4
/// carries both; the wav is a convenience and a fallback.
enum MovieWriter {

    /// 16-bit PCM stereo, written by hand because the alternative is an
    /// `AVAudioFile` and a format conversion for a file this simple.
    static func writeWAV(samples: [[Float]], to url: URL,
                         sampleRate: Int = H3Audio.sampleRate) throws {
        guard samples.count == 2 else {
            throw H3Error.invalidRequest(rule: "audio must be stereo",
                                         detail: "\(samples.count) channel(s)",
                                         remedy: "the audio VAE decodes to two channels.")
        }
        let channels = 2, bytesPerSample = 2
        let frames = samples[0].count
        let dataSize = frames * channels * bytesPerSample

        var data = Data()
        data.reserveCapacity(44 + dataSize)
        func u32(_ v: UInt32) { var x = v.littleEndian; data.append(withUnsafeBytes(of: &x) { Data($0) }) }
        func u16(_ v: UInt16) { var x = v.littleEndian; data.append(withUnsafeBytes(of: &x) { Data($0) }) }

        data.append(contentsOf: Array("RIFF".utf8))
        u32(UInt32(36 + dataSize))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        u32(16)                                   // PCM header size
        u16(1)                                    // PCM
        u16(UInt16(channels))
        u32(UInt32(sampleRate))
        u32(UInt32(sampleRate * channels * bytesPerSample))
        u16(UInt16(channels * bytesPerSample))
        u16(16)                                   // bits per sample
        data.append(contentsOf: Array("data".utf8))
        u32(UInt32(dataSize))

        for i in 0 ..< frames {
            for c in 0 ..< channels {
                let clamped = max(-1.0, min(1.0, samples[c][i]))
                u16(UInt16(bitPattern: Int16(clamped * 32767.0)))
            }
        }
        try data.write(to: url)
    }

    /// One mp4 carrying both streams, colour-tagged, with every failure path
    /// checked.
    ///
    /// Both tracks are driven by `requestMediaDataWhenReady`, which is the API
    /// built for this. Two hand-rolled push loops were tried first and both
    /// deadlocked: appending all video then all audio stalls immediately, and
    /// even a timestamp-interleaved loop stopped being served after 20 steps at
    /// 864x480 — `status 1`, no error, the input simply stopped accepting.
    /// AVFoundation pulls from each input at its own pace here, so there is no
    /// interleaving policy left to get wrong.
    ///
    /// Audio is still delivered in **1024-sample chunks**: AAC's frame size is
    /// 1024, and handing an encoder the whole waveform in one buffer is the same
    /// mistake that returned EINVAL from the ComfyUI mux path.
    ///
    /// - Parameter argb: `[T, H, W, 4]` uint8, alpha first.
    static func writeMovie(argb: MLXArray, waveform: [[Float]], to url: URL,
                           fps: Double, sampleRate: Int,
                           withAudio: Bool = true) async throws {
        let frameCount = argb.dim(0), height = argb.dim(1), width = argb.dim(2)
        guard frameCount > 0 else {
            throw H3Error.invalidRequest(rule: "nothing to write", detail: "no frames",
                                         remedy: "the decode produced an empty tensor.")
        }
        func fail(_ m: String) -> H3Error {
            H3Error.unreadable(path: url.path, reason: m)
        }

        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)

        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ])
        videoInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])
        guard writer.canAdd(videoInput) else { throw fail("AVAssetWriter refused the video input") }
        writer.add(videoInput)

        var audioInput: AVAssetWriterInput?
        var audioFormat: CMFormatDescription?
        if withAudio {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 2,
                AVEncoderBitRateKey: 192_000,
            ])
            input.expectsMediaDataInRealTime = false
            guard writer.canAdd(input) else { throw fail("AVAssetWriter refused the audio input") }
            writer.add(input)
            audioInput = input

            var asbd = AudioStreamBasicDescription(
                mSampleRate: Double(sampleRate), mFormatID: kAudioFormatLinearPCM,
                mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                mBytesPerPacket: 8, mFramesPerPacket: 1, mBytesPerFrame: 8,
                mChannelsPerFrame: 2, mBitsPerChannel: 32, mReserved: 0)
            guard CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault, asbd: &asbd,
                                                 layoutSize: 0, layout: nil, magicCookieSize: 0,
                                                 magicCookie: nil, extensions: nil,
                                                 formatDescriptionOut: &audioFormat) == noErr,
                  audioFormat != nil else {
                throw fail("could not describe the PCM source format")
            }
        }

        guard writer.startWriting() else {
            throw fail("startWriting failed: " + (writer.error?.localizedDescription ?? "unknown"))
        }
        writer.startSession(atSourceTime: .zero)

        let rowBytes = width * 4
        let frameStride = height * rowBytes
        let timescale: CMTimeScale = 90_000

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw fail("could not create a Metal device")
        }
        let contiguous = argb.contiguous()
        guard let mtlBuffer = contiguous.asMTLBuffer(device: device, noCopy: true) else {
            throw fail("could not wrap the video frames as a zero-copy MTLBuffer")
        }
        nonisolated(unsafe) let source = mtlBuffer.contents()

        // AVFoundation's writer objects are not Sendable, but each is touched by
        // exactly one serial queue here and `finishWriting` happens after both
        // have finished. That is the contract the API documents.
        nonisolated(unsafe) let vIn = videoInput
        nonisolated(unsafe) let vAdaptor = adaptor
        nonisolated(unsafe) let wr = writer
        let cursor = MuxCursor()

        let muxError = await withCheckedContinuation { continuation in
            let completion = MuxCompletion(remaining: withAudio ? 2 : 1,
                                           continuation: continuation)
            vIn.requestMediaDataWhenReady(on: DispatchQueue(label: "h3.video.mux")) {
                while vIn.isReadyForMoreMediaData {
                    let i = cursor.frame
                    if i >= frameCount {
                        vIn.markAsFinished(); completion.finish(); return
                    }
                guard let pool = vAdaptor.pixelBufferPool else {
                    vIn.markAsFinished(); completion.finish(error: "pixel buffer pool unavailable")
                    return
                }
                var pb: CVPixelBuffer?
                guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess,
                      let buffer = pb else {
                    vIn.markAsFinished()
                    completion.finish(error: "could not allocate a pixel buffer for frame \(i)")
                    return
                }
                CVPixelBufferLockBaseAddress(buffer, [])
                if let base = CVPixelBufferGetBaseAddress(buffer) {
                    let dstStride = CVPixelBufferGetBytesPerRow(buffer)
                    let frameBase = source.advanced(by: i * frameStride)
                    for y in 0 ..< height {
                        memcpy(base.advanced(by: y * dstStride), frameBase + y * rowBytes, rowBytes)
                    }
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                let pts = CMTime(value: CMTimeValue(Double(i) / fps * Double(timescale)),
                                 timescale: timescale)
                guard vAdaptor.append(buffer, withPresentationTime: pts) else {
                    vIn.markAsFinished()
                    completion.finish(error: "video append failed at frame \(i): "
                                      + (wr.error?.localizedDescription ?? "unknown"))
                    return
                }
                cursor.frame = i + 1
                }
            }

            if let audioIn = audioInput, let format = audioFormat {
                nonisolated(unsafe) let aIn = audioIn
                let total = waveform[0].count
                aIn.requestMediaDataWhenReady(on: DispatchQueue(label: "h3.audio.mux")) {
                    while aIn.isReadyForMoreMediaData {
                    let offset = cursor.sample
                    if offset >= total { aIn.markAsFinished(); completion.finish(); return }
                    let n = min(1024, total - offset)
                    var interleaved = [Float](repeating: 0, count: n * 2)
                    for j in 0 ..< n {
                        interleaved[j * 2] = waveform[0][offset + j]
                        interleaved[j * 2 + 1] = waveform[1][offset + j]
                    }
                    let byteCount = n * 2 * MemoryLayout<Float>.size
                    var block: CMBlockBuffer?
                    guard CMBlockBufferCreateWithMemoryBlock(
                            allocator: kCFAllocatorDefault, memoryBlock: nil,
                            blockLength: byteCount, blockAllocator: kCFAllocatorDefault,
                            customBlockSource: nil, offsetToData: 0, dataLength: byteCount,
                            flags: 0, blockBufferOut: &block) == noErr, let bb = block,
                          CMBlockBufferAssureBlockMemory(bb) == noErr else {
                        aIn.markAsFinished()
                        completion.finish(error: "could not allocate an audio block buffer")
                        return
                    }
                    interleaved.withUnsafeBytes { raw in
                        _ = CMBlockBufferReplaceDataBytes(with: raw.baseAddress!, blockBuffer: bb,
                                                          offsetIntoDestination: 0,
                                                          dataLength: byteCount)
                    }
                    var sample: CMSampleBuffer?
                    var timing = CMSampleTimingInfo(
                        duration: CMTime(value: 1, timescale: CMTimeScale(sampleRate)),
                        presentationTimeStamp: CMTime(value: CMTimeValue(offset),
                                                      timescale: CMTimeScale(sampleRate)),
                        decodeTimeStamp: .invalid)
                    var sizes = [8]
                    guard CMSampleBufferCreateReady(
                            allocator: kCFAllocatorDefault, dataBuffer: bb,
                            formatDescription: format, sampleCount: n,
                            sampleTimingEntryCount: 1, sampleTimingArray: &timing,
                            sampleSizeEntryCount: 1, sampleSizeArray: &sizes,
                            sampleBufferOut: &sample) == noErr, let sb = sample else {
                        aIn.markAsFinished()
                        completion.finish(error: "could not build an audio sample buffer")
                        return
                    }
                    guard aIn.append(sb) else {
                        aIn.markAsFinished()
                        completion.finish(error: "audio append failed at sample \(offset): "
                                          + (wr.error?.localizedDescription ?? "unknown"))
                        return
                    }
                    cursor.sample = offset + n
                    }
                }
            }
        }

        // Both queues read `source` straight out of MLX's storage. ARC's last
        // use of the array and its Metal wrapper is the `contents()` call above,
        // so without this it is free to release them — and free the storage —
        // while a queue is still copying.
        withExtendedLifetime(contiguous) { withExtendedLifetime(mtlBuffer) {} }
        if let muxError { throw fail(muxError) }

        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw fail("writing ended in status \(writer.status.rawValue): "
                       + (writer.error?.localizedDescription ?? "unknown"))
        }
    }
}
