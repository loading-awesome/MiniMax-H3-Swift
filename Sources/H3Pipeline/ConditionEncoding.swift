import Foundation
import MLX
import MLXRandom
import H3Foundation
import H3Modules

/// The conditioning rows the DiT packs, and the block table describing them.
struct EncodedConditions {
    /// Patchified, noise-augmented video conditioning rows, or nil when there
    /// are none.
    let videoRows: MLXArray?
    let audioRows: MLXArray?
    let keyframes: [KeyframeConfig]
    let refs: [ReferenceBlock]
    /// True when the noise augmentation came from MLX's PRNG rather than a
    /// recorded tensor, which puts the render outside the parity contract.
    let outsideParityContract: Bool
}

/// Phase 2: media becomes latents, and latents become packed rows.
///
/// **Everything here runs before the DiT is loaded.** The VAE encoders are
/// small — 5.2 GB video, 0.6 GB audio — but they allocate freely, and MLX pools
/// freed buffers rather than returning them. Encoding after the DiT would leave
/// those allocations sitting underneath 66 GB of weights for the whole render.
enum ConditionEncoder {

    /// Reference video frames, trimmed the two ways the reference trims them,
    /// in that order.
    ///
    /// The generation length truncates first, and then the count walks **down**
    /// to the nearest `n % 17 == 5`. Both matter: the VAE's temporal chunker
    /// pads the tail by repeating the last frame, so an off-lattice count would
    /// silently encode manufactured frames as if they were footage.
    static func loadReferenceVideo(at url: URL, frameCount: Int,
                                   log: (String) -> Void = { _ in }) async throws -> MLXArray {
        let (all, fps) = try await MediaLoad.videoHWC(at: url.path, maxFrames: frameCount)
        if abs(fps - Double(H3Video.fps)) > 0.5 {
            log(String(format: "warning: %@ is %.2f fps, not %d. Qwen samples reference video "
                       + "by INDEX (every %d frames) and stamps the result in seconds, so the "
                       + "timestamps this produces are wrong by %.2fx.",
                       url.lastPathComponent, fps, H3Video.fps, H3Video.fps / 2,
                       Double(H3Video.fps) / max(fps, 0.01)))
        }
        var n = all.dim(0)
        guard n >= 5 else {
            throw H3Error.invalidRequest(
                rule: "reference video too short",
                detail: "\(url.lastPathComponent) has \(n) frame(s)",
                remedy: "reference videos need at least 5 frames (~0.2 s at 24 fps).")
        }
        while n % 17 != 5 { n -= 1 }
        if n != all.dim(0) {
            log("    \(url.lastPathComponent): \(all.dim(0)) frames -> \(n) (reference video "
                + "lengths live on the same 17k+5 lattice, walking down)")
        }
        return all[0 ..< n]
    }

    /// Keyframes and reference images through the video VAE, then reference
    /// videos.
    ///
    /// Order matters and is the same order the layout declares its conditions:
    /// first frame, last frame, reference images, reference videos. The packed
    /// layout consumes these positionally, so a reordering here misaligns every
    /// conditioning row while leaving every shape valid.
    ///
    /// **Keyframes are fitted to the target resolution** because they share the
    /// video's spatial position ids — a keyframe on a different grid would place
    /// its rows at coordinates the target never visits. Reference images are
    /// independent blocks with their own grid, so they keep their own size.
    static func encodeVisual(request: RenderRequest, videoVAE url: URL,
                             width: Int, height: Int,
                             referenceVideoFrames: [MLXArray],
                             log: (String) -> Void = { _ in }) throws -> [MLXArray] {
        var paths: [(URL, Bool)] = []                       // (url, fitToTarget)
        if let f = request.firstFrame { paths.append((f, true)) }
        if let l = request.lastFrame { paths.append((l, true)) }
        paths += request.referenceImages.map { ($0, false) }
        guard !paths.isEmpty || !referenceVideoFrames.isEmpty else { return [] }

        log("Encoding \(paths.count + referenceVideoFrames.count) visual condition(s)...")
        let encoder = try VideoVAEEncoder(weights: try MLX.loadArrays(url: url))

        var out = try paths.map { item -> MLXArray in
            let pixels = try MediaLoad.image(at: item.0.path,
                                             fit: item.1 ? (width, height) : nil)
            let z = encoder.encode(pixels)
            eval(z)
            log("    \(item.0.lastPathComponent): \(pixels.dim(4))x\(pixels.dim(3)) -> \(z.shape)")
            return z
        }
        // `encode` routes T > 1 to the temporal path on its own: pad the tail by
        // repeating the last frame up to a multiple of 17, encode each clip
        // independently, concatenate, then drop the last 3 latent frames. That
        // drop is why the frame count had to walk down onto the lattice first.
        for (i, frames) in referenceVideoFrames.enumerated() {
            let pixels = (frames * 2.0 - 1.0)               // [T,H,W,3] -> [1,3,T,H,W]
                .transposed(3, 0, 1, 2)
                .expandedDimensions(axis: 0)
            let z = encoder.encode(pixels)
            eval(z)
            log("    \(request.referenceVideos[i].lastPathComponent): \(pixels.dim(2)) frames "
                + "-> \(z.shape)")
            out.append(z)
        }
        return out
    }

    /// Reference audio, decoded to 32 kHz stereo and run through the audio VAE.
    static func encodeAudio(request: RenderRequest, audioVAE url: URL,
                            log: (String) -> Void = { _ in }) throws -> [MLXArray] {
        let paths = request.orderedAudio
        guard !paths.isEmpty else { return [] }
        log("Encoding \(paths.count) audio condition(s)...")
        let encoder = try AudioVAEEncoder(weights: try MLX.loadArrays(url: url))
        return try paths.map { path in
            let wave = try MediaLoad.audio(at: path.path)
            let z = encoder.encode(wave)
            eval(z)
            log("    \(path.lastPathComponent): \(wave.dim(2)) samples -> \(z.shape)")
            return z
        }
    }

    /// The block table, and the rows that go with it.
    ///
    /// Building both together is deliberate: the table is derived from the
    /// latents' own shapes rather than from the source media, because the two
    /// can disagree — a VAE encode snaps its own frame lattice, and an odd
    /// latent axis gets rounded. If they disagree, the layout describes one
    /// thing while the rows carry another, and that misalignment is silent: every
    /// downstream shape stays valid because the segment table is self-consistent.
    static func assemble(request: RenderRequest, geometry: LatentGeometry,
                         visualLatents: [MLXArray], audioLatents: [MLXArray],
                         recordedNoise: [MLXArray],
                         log: (String) -> Void = { _ in }) throws -> EncodedConditions {

        var keyframes: [KeyframeConfig] = []
        if request.firstFrame != nil { keyframes.append(KeyframeConfig(resolvedFrameIndex: 0)) }
        // The last anchor sits on the *aligned* frame count, not on the
        // requested duration: the request is snapped up onto the 17k+5 lattice
        // before anything else sees it.
        if request.lastFrame != nil {
            keyframes.append(KeyframeConfig(resolvedFrameIndex: geometry.frameCount - 1))
        }

        let wantVisual = keyframes.count + request.referenceImages.count
            + request.referenceVideos.count
        guard visualLatents.count == wantVisual else {
            throw H3Error.invalidRequest(
                rule: "conditioning count mismatch",
                detail: "\(visualLatents.count) visual condition latent(s) for a payload that "
                      + "needs \(wantVisual) (\(keyframes.count) keyframe(s) + "
                      + "\(request.referenceImages.count) image(s) + "
                      + "\(request.referenceVideos.count) video(s))",
                remedy: "the order is keyframes, images, videos; the layout consumes them "
                      + "positionally.")
        }
        guard audioLatents.count == request.orderedAudio.count else {
            throw H3Error.invalidRequest(
                rule: "conditioning count mismatch",
                detail: "\(audioLatents.count) audio condition latent(s) for a payload that "
                      + "needs \(request.orderedAudio.count)",
                remedy: "paired soundtracks come first in reference-video order, then the "
                      + "standalone clips.")
        }

        // Keyframes consume the first latents and reuse the target frame grid,
        // so only the reference blocks need shapes read back.
        var cursor = keyframes.count
        var audioCursor = 0
        var refs: [ReferenceBlock] = []

        for _ in request.referenceImages {
            let (_, h, w) = try latentTHW(visualLatents[cursor])
            refs.append(ReferenceBlock(kind: .image, latentH: h, latentW: w))
            cursor += 1
        }
        for i in 0 ..< request.referenceVideos.count {
            let (t, h, w) = try latentTHW(visualLatents[cursor])
            // A paired soundtrack does not become its own block. It folds into
            // the video's, as `video_audio`, with its rows ahead of the video's
            // inside that one block — and it advances the position cursor by
            // `max(refAudioT, videoTSpan)` rather than by either alone.
            var refAudioT = 0
            if request.soundtrack(for: i) != nil {
                refAudioT = audioLatents[audioCursor].dim(-1)
                audioCursor += 1
            }
            refs.append(ReferenceBlock(kind: refAudioT > 0 ? .videoAudio : .video,
                                       latentH: h, latentW: w, latentT: t,
                                       refAudioT: refAudioT))
            cursor += 1
        }
        for _ in request.referenceAudio {
            // The audio latent is [.., 2, T]; the 2 is the stereo pair, not a
            // batch axis.
            refs.append(ReferenceBlock(kind: .audio,
                                       refAudioT: audioLatents[audioCursor].dim(-1)))
            audioCursor += 1
        }

        let patch = geometry.config.patchSize
        let video = try rows(visualLatents, aug: H3Cond.visualNoiseAug,
                             seed: request.seed, label: "visual", recorded: recordedNoise) {
            H3Packing.patchifyVideo($0.asType(.float32), patch: patch)
        }
        // Audio never takes this branch — `audioNoiseAug` is exactly 1.0 — so it
        // needs no recorded noise.
        let audio = try rows(audioLatents, aug: H3Cond.audioNoiseAug,
                             seed: request.seed &+ 1, label: "audio", recorded: []) {
            H3Packing.packAudio($0.asType(.float32))
        }

        let unrecorded = !visualLatents.isEmpty && recordedNoise.isEmpty
            && H3Cond.visualNoiseAug < 1.0
        if unrecorded {
            log("warning: visual conditioning rows are noise-augmented at "
                + "aug=\(H3Cond.visualNoiseAug) using MLX's PRNG, which cannot reproduce the "
                + "reference's torch.Generator(\"cpu\") bytes. This render is NOT covered by "
                + "the parity contract. Supply recorded conditioning noise to make it "
                + "reproducible.")
        }

        return EncodedConditions(videoRows: video, audioRows: audio,
                                 keyframes: keyframes, refs: refs,
                                 outsideParityContract: unrecorded)
    }

    /// Pre-encoded conditioning latents, in **key order**.
    ///
    /// safetensors is a dictionary, so the file itself carries no order — the
    /// sorted key order is the contract, and it has to line up with the order
    /// the payload declares its conditions. Name them `00_*`, `01_*` and it is
    /// unambiguous.
    static func loadLatents(_ url: URL?) throws -> [MLXArray] {
        guard let url else { return [] }
        let dict = try MLX.loadArrays(url: url)
        return dict.keys.sorted().compactMap { dict[$0] }
    }

    private static func latentTHW(_ a: MLXArray) throws -> (Int, Int, Int) {
        guard a.ndim >= 3 else {
            throw H3Error.invalidRequest(
                rule: "malformed conditioning latent",
                detail: "a video latent needs at least 3 axes (T, H, W); got shape \(a.shape)",
                remedy: "re-encode the condition, or check the key order of a pre-encoded file.")
        }
        return (a.dim(-3), a.dim(-2), a.dim(-1))
    }

    /// Patchified conditioning rows with the reference's noise augmentation.
    ///
    /// The reference draws this noise from `torch.Generator("cpu")`, restarted
    /// from the same seed for every condition. MLX has its own PRNG and will
    /// never emit those bytes, so the noise is supplied as a **recorded input**
    /// — the same rule the sampler noise follows, and the reason a seed is not a
    /// shared input while a tensor is.
    private static func rows(_ latents: [MLXArray], aug: Float, seed: UInt64,
                             label: String, recorded: [MLXArray],
                             pack: (MLXArray) -> MLXArray) throws -> MLXArray? {
        guard !latents.isEmpty else { return nil }
        guard aug < 1.0 else {
            // No augmentation at all: the rows are the latents.
            return concatenated(latents.map(pack), axis: 0)
        }
        if !recorded.isEmpty && recorded.count != latents.count {
            throw H3Error.invalidRequest(
                rule: "conditioning noise mismatch",
                detail: "\(recorded.count) recorded tensor(s) for \(latents.count) "
                      + "\(label) condition(s)",
                remedy: "emit one noise tensor per condition, in the same order.")
        }

        var out: [MLXArray] = []
        for (i, z) in latents.enumerated() {
            let r = pack(z)
            let noise: MLXArray
            if i < recorded.count {
                let n = recorded[i]
                guard n.shape == r.shape else {
                    throw H3Error.invalidRequest(
                        rule: "conditioning noise mismatch",
                        detail: "noise[\(i)] is \(n.shape); condition \(i) needs \(r.shape)",
                        remedy: "emit with shape \(r.dim(0))x\(r.dim(1)).")
                }
                noise = n.asType(.float32)
            } else {
                // Every condition restarts the same stream, as the reference
                // does, rather than reading on through one shared one.
                noise = MLXRandom.normal(r.shape, key: MLXRandom.key(seed))
            }
            out.append(MLXArray(aug) * r + MLXArray(1.0 - aug) * noise)
        }
        return concatenated(out, axis: 0)
    }
}
