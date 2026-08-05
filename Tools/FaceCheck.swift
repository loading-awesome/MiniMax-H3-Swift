import Foundation
import ArgumentParser
import AVFoundation
import CoreImage
import Vision

/// An **external** oracle for a rendered video: does a face detector find a
/// face, and does the video flash?
///
/// Every other check in this tree compares against a CUDA golden, which answers
/// "do we match the reference" and cannot answer "is the output any good". A
/// golden is also unavailable for anything we have not captured — and capturing
/// one for every prompt is not a plan.
///
/// Apple's `Vision` face detector knows nothing about H3, MLX, or our goldens.
/// If we render a talking head and it finds a face, in a stable place, in most
/// frames, that is evidence no amount of tap comparison provides. If it does
/// not, that is a real failure even with every gating tap green.
///
/// Two independent measurements, because they fail differently:
///
///   * **Face detection** — presence, count, position stability, and whether
///     landmarks (eyes/nose/mouth) resolve. A blob that reads as a face to a
///     human but has no landmarks is a different failure from no face at all.
///   * **Flash detection** — frame-to-frame luminance and per-channel jumps.
///     A render can be perfectly prompt-faithful and still strobe, and
///     "structure is present" checks cannot see it. The per-channel split
///     matters: a jump in one channel only is a different bug from a jump in
///     all three, and a green flash is exactly the former.
struct FaceCheck: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "face-check",
        abstract: "Detect faces and flashes in a rendered video — an oracle with no golden."
    )

    @Argument(help: "path to an .mp4 to analyse")
    var video: String

    @Option(help: "analyse every Nth frame (1 = all)")
    var stride: Int = 1

    @Option(help: "flag a frame when mean luminance jumps by more than this (0-1 scale)")
    var flashThreshold: Double = 0.06

    @Flag(help: "print a line per analysed frame")
    var verbose = false

    /// OCR is far slower than face detection, so it samples rather than
    /// covering every frame. Text that appears in only one frame of ten is
    /// itself a finding — legible text should persist.
    @Option(help: "run OCR on every Nth analysed frame (0 disables)")
    var ocrStride: Int = 10

    @Option(help: "text the prompt asked to appear on screen")
    var expectText: String?

    /// Whether this scene is supposed to contain a person.
    ///
    /// Off by default, because most renders are not talking heads and a check
    /// that fails every landscape is a check nobody reads.
    @Flag(help: "fail if a face is not present in at least 90% of frames")
    var expectFace = false

    struct FrameStat {
        let index: Int
        let luma: Double
        let r: Double, g: Double, b: Double
        /// Fraction of pixels where one channel runs far ahead of the other
        /// two. A whole-frame mean cannot see a saturated blob covering a few
        /// percent of the image — the first version of this check reported a
        /// clean video while a green patch was plainly visible in it. This is
        /// the measurement that catches that.
        var excessR: Double = 0, excessG: Double = 0, excessB: Double = 0
        /// Per-tile luminance, for localized flashes a frame mean averages away.
        var tiles: [Double] = []
        var faces: Int = 0
        var landmarks: Int = 0
        var box: CGRect? = nil
        /// Mean absolute luminance change from the previous analysed frame.
        /// Separates "static shot held" from "nothing is moving because the
        /// render collapsed", which look identical in every other statistic.
        var motion: Double = 0
        /// Variance of a Laplacian-like high-pass, as a sharpness proxy. A
        /// render that dissolves into mush keeps its mean and its colour and
        /// loses this.
        var detail: Double = 0
        var ocr: [String] = []
        var ocrConfidence: Float = 0
    }

    static let tileGrid = 8
    /// A channel is "in excess" when it exceeds the brighter of the other two
    /// by this much on a 0-1 scale. 0.15 is well beyond anything natural
    /// shading produces and well below a saturated artifact.
    static let excessMargin = 0.15

    func run() throws {
        let url = URL(fileURLWithPath: video)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("no such file: \(video)")
        }
        let asset = AVURLAsset(url: url)
        guard let track = try loadTrack(asset) else {
            throw ValidationError("no video track in \(video)")
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String:
                                kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        var stats: [FrameStat] = []
        var index = 0
        var prevLuma: [Double]? = nil
        var lumaW = 0, lumaH = 0
        while let sample = output.copyNextSampleBuffer() {
            defer { index += 1 }
            guard index % stride == 0,
                  let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            let (statBase, lumaPlane, lw, lh) = channelMeans(pixels, index: index)
            var stat = statBase
            lumaW = lw; lumaH = lh

            // Landmarks, not just rectangles: a face-shaped blob with no
            // resolvable eyes or mouth is a different (and more interesting)
            // outcome than no face.
            let handler = VNImageRequestHandler(cvPixelBuffer: pixels, options: [:])
            let request = VNDetectFaceLandmarksRequest()
            try? handler.perform([request])
            let found = request.results ?? []
            stat.faces = found.count
            if let best = found.max(by: { $0.confidence < $1.confidence }) {
                stat.box = best.boundingBox
                stat.landmarks = best.landmarks?.allPoints?.pointCount ?? 0
            }
            // Motion and detail relative to the previous analysed frame.
            if let prev = prevLuma, prev.count == lumaPlane.count {
                var acc = 0.0
                for i in 0 ..< prev.count { acc += abs(prev[i] - lumaPlane[i]) }
                stat.motion = acc / Double(prev.count)
            }
            stat.detail = detailScore(lumaPlane, width: lumaW, height: lumaH)
            prevLuma = lumaPlane

            if ocrStride > 0 && stats.count % ocrStride == 0 {
                let text = VNRecognizeTextRequest()
                text.recognitionLevel = .accurate
                text.usesLanguageCorrection = false
                try? handler.perform([text])
                for o in text.results ?? [] {
                    if let top = o.topCandidates(1).first, top.confidence > 0.3 {
                        stat.ocr.append(top.string)
                        stat.ocrConfidence = max(stat.ocrConfidence, top.confidence)
                    }
                }
            }

            stats.append(stat)
            if verbose {
                print(String(format: "    f%04d luma %.3f  faces %d  landmarks %d",
                             stat.index, stat.luma, stat.faces, stat.landmarks))
            }
        }
        guard !stats.isEmpty else { throw ValidationError("decoded no frames") }

        report(stats, totalFrames: index)
    }

    /// `loadTracks` is async and `run()` is not. A `final class` box crossing
    /// the boundary keeps the compiler's concurrency checking satisfied without
    /// making the whole command async for one call.
    private final class Box: @unchecked Sendable {
        var track: AVAssetTrack?
        var error: Error?
    }

    private func loadTrack(_ asset: AVURLAsset) throws -> AVAssetTrack? {
        let box = Box()
        let sem = DispatchSemaphore(value: 0)
        Task { [box] in
            do { box.track = try await asset.loadTracks(withMediaType: .video).first }
            catch { box.error = error }
            sem.signal()
        }
        sem.wait()
        if let e = box.error { throw e }
        return box.track
    }

    /// Returns the frame statistics plus a decimated luminance plane, which
    /// the motion and detail measures both need and which is not worth walking
    /// the pixels twice for.
    private func channelMeans(_ buf: CVPixelBuffer, index: Int)
        -> (FrameStat, [Double], Int, Int) {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
        let stride = CVPixelBufferGetBytesPerRow(buf)
        guard let base = CVPixelBufferGetBaseAddress(buf) else {
            return (FrameStat(index: index, luma: 0, r: 0, g: 0, b: 0), [], 0, 0)
        }
        let p = base.assumingMemoryBound(to: UInt8.self)
        var sr = 0.0, sg = 0.0, sb = 0.0
        var xr = 0.0, xg = 0.0, xb = 0.0
        let g = Self.tileGrid
        var tileSum = [Double](repeating: 0, count: g * g)
        var tileN = [Double](repeating: 0, count: g * g)
        var n = 0.0
        var plane: [Double] = []
        plane.reserveCapacity((h / 2 + 1) * (w / 2 + 1))
        var planeW = 0
        for y in Swift.stride(from: 0, to: h, by: 2) {
            let row = p + y * stride
            let ty = min(g - 1, y * g / h)
            for x in Swift.stride(from: 0, to: w, by: 2) {
                let o = x * 4                      // BGRA
                let bb = Double(row[o]) / 255, gg = Double(row[o + 1]) / 255
                let rr = Double(row[o + 2]) / 255
                sb += bb; sg += gg; sr += rr
                if rr - Swift.max(gg, bb) > Self.excessMargin { xr += 1 }
                if gg - Swift.max(rr, bb) > Self.excessMargin { xg += 1 }
                if bb - Swift.max(rr, gg) > Self.excessMargin { xb += 1 }
                let l = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
                plane.append(l)
                let t = min(g - 1, x * g / w) + ty * g
                tileSum[t] += l; tileN[t] += 1
                n += 1
            }
            if planeW == 0 { planeW = plane.count }
        }
        let r = sr / n, gm = sg / n, bl = sb / n
        var stat = FrameStat(index: index,
                             luma: 0.2126 * r + 0.7152 * gm + 0.0722 * bl,
                             r: r, g: gm, b: bl)
        stat.excessR = xr / n; stat.excessG = xg / n; stat.excessB = xb / n
        stat.tiles = (0 ..< g * g).map { tileN[$0] > 0 ? tileSum[$0] / tileN[$0] : 0 }
        return (stat, plane, planeW, planeW > 0 ? plane.count / planeW : 0)
    }

    /// Mean squared 4-neighbour Laplacian — high-frequency energy. Blur and
    /// temporal mush both collapse it while leaving mean, colour and even
    /// motion largely intact.
    private func detailScore(_ p: [Double], width: Int, height: Int) -> Double {
        guard width > 2, height > 2, p.count >= width * height else { return 0 }
        var acc = 0.0
        var n = 0.0
        for y in 1 ..< (height - 1) {
            for x in 1 ..< (width - 1) {
                let i = y * width + x
                let lap = 4 * p[i] - p[i - 1] - p[i + 1] - p[i - width] - p[i + width]
                acc += lap * lap
                n += 1
            }
        }
        return n > 0 ? acc / n : 0
    }

    private func report(_ s: [FrameStat], totalFrames: Int) {
        print("face-check \(video)")
        print("  \(totalFrames) frames, \(s.count) analysed (stride \(stride))")

        // ---- faces
        let withFace = s.filter { $0.faces > 0 }
        let rate = Double(withFace.count) / Double(s.count)
        let withLandmarks = s.filter { $0.landmarks > 0 }
        print("\n  faces")
        print(String(format: "    detected in       %d/%d frames (%.1f%%)",
                     withFace.count, s.count, 100 * rate))
        print(String(format: "    with landmarks    %d/%d frames (%.1f%%)",
                     withLandmarks.count, s.count,
                     100 * Double(withLandmarks.count) / Double(s.count)))
        if !withFace.isEmpty {
            let boxes = withFace.compactMap { $0.box }
            let cx = boxes.map { $0.midX }, cy = boxes.map { $0.midY }
            let area = boxes.map { Double($0.width * $0.height) }
            func sd(_ v: [CGFloat]) -> Double {
                let m = v.reduce(0, +) / CGFloat(v.count)
                return Double((v.map { ($0 - m) * ($0 - m) }.reduce(0, +) / CGFloat(v.count)).squareRoot())
            }
            print(String(format: "    centre drift      sd_x %.4f  sd_y %.4f (frame widths)",
                         sd(cx), sd(cy)))
            print(String(format: "    size              mean %.3f of frame, min %.3f max %.3f",
                         area.reduce(0, +) / Double(area.count), area.min()!, area.max()!))
        }

        // ---- flashes and localized artifacts
        //
        // Three measures, because they catch different things and the frame
        // mean alone catches almost nothing:
        //   1. whole-frame luminance jump  — a true global strobe
        //   2. worst per-tile jump          — a flash in part of the frame
        //   3. channel-excess fraction      — a saturated colour blob, which
        //      moves the frame mean by almost nothing
        print("\n  flashes and colour artifacts")
        let lumas = s.map { $0.luma }
        let lmean = lumas.reduce(0, +) / Double(lumas.count)
        let lsd = (lumas.map { ($0 - lmean) * ($0 - lmean) }.reduce(0, +)
                   / Double(lumas.count)).squareRoot()
        print(String(format: "    luma              mean %.3f  sd %.4f  min %.3f  max %.3f",
                     lmean, lsd, lumas.min()!, lumas.max()!))

        let exG = s.map { $0.excessG }, exR = s.map { $0.excessR }, exB = s.map { $0.excessB }
        func pk(_ v: [Double]) -> (Double, Int) {
            var m = 0.0, i = 0
            for (k, x) in v.enumerated() where x > m { m = x; i = k }
            return (m, s[i].index)
        }
        let (mg, fg) = pk(exG), (mr, fr) = pk(exR), (mb, fb) = pk(exB)
        print(String(format: "    channel excess    R peak %.2f%% (f%04d)  G peak %.2f%% (f%04d)  B peak %.2f%% (f%04d)",
                     100 * mr, fr, 100 * mg, fg, 100 * mb, fb))

        var jumps: [(Int, String, Double)] = []
        for i in 1 ..< s.count {
            let dl = abs(s[i].luma - s[i - 1].luma)
            if dl > flashThreshold { jumps.append((s[i].index, "global luma", dl)) }
            var worstTile = 0.0
            if s[i].tiles.count == s[i - 1].tiles.count {
                for t in 0 ..< s[i].tiles.count {
                    worstTile = Swift.max(worstTile, abs(s[i].tiles[t] - s[i - 1].tiles[t]))
                }
            }
            if worstTile > flashThreshold * 2 {
                jumps.append((s[i].index, "tile luma", worstTile))
            }
            let dg = s[i].excessG - s[i - 1].excessG
            let dr = s[i].excessR - s[i - 1].excessR
            let db = s[i].excessB - s[i - 1].excessB
            if dg > 0.01 { jumps.append((s[i].index, "green blob", dg)) }
            if dr > 0.01 { jumps.append((s[i].index, "red blob", dr)) }
            if db > 0.01 { jumps.append((s[i].index, "blue blob", db)) }
        }
        print(String(format: "    events            %d", jumps.count))
        for (idx, kind, v) in jumps.prefix(24) {
            print(String(format: "      f%04d  %-12@ %.3f", idx, kind as NSString, v))
        }
        if jumps.count > 24 { print("      ... \(jumps.count - 24) more") }

        // Any frame where a channel is in excess over more than 1% of the
        // image is worth naming outright, jump or not — a persistent blob
        // never produces a frame-to-frame delta.
        let blobFrames = s.filter { Swift.max($0.excessR, Swift.max($0.excessG, $0.excessB)) > 0.01 }
        if !blobFrames.isEmpty {
            print(String(format: "    frames with a channel blob over 1%% of image: %d/%d",
                         blobFrames.count, s.count))
        }

        // ---- motion, detail, text
        let motions = s.dropFirst().map { $0.motion }
        if !motions.isEmpty {
            let mm = motions.reduce(0, +) / Double(motions.count)
            let mx = motions.max() ?? 0, mn = motions.min() ?? 0
            print("\n  motion and detail")
            print(String(format: "    frame-to-frame    mean %.4f  min %.4f  max %.4f",
                         mm, mn, mx))
            // A render that collapses looks static. A held shot also looks
            // static. Detail separates them: mush loses high frequencies.
            let det = s.map { $0.detail }
            let dm = det.reduce(0, +) / Double(det.count)
            print(String(format: "    detail (lap var)  mean %.5f  min %.5f  max %.5f",
                         dm, det.min() ?? 0, det.max() ?? 0))
            if mm < 0.002 {
                print("    NOTE: almost no frame-to-frame change — either a held static "
                      + "shot or a collapsed render; check detail and look at pixels")
            }
        }

        let ocrFrames = s.filter { !$0.ocr.isEmpty }
        if ocrStride > 0 {
            print("\n  on-screen text")
            print(String(format: "    frames with text  %d/%d OCR'd",
                         ocrFrames.count, s.filter { $0.index % (stride * ocrStride) == 0 }.count))
            var seen: [String: Int] = [:]
            for f in ocrFrames { for t in f.ocr { seen[t, default: 0] += 1 } }
            for (t, c) in seen.sorted(by: { $0.value > $1.value }).prefix(8) {
                print("      x\(c)  \"\(t)\"")
            }
            if let want = expectText {
                let w = want.lowercased()
                let hits = seen.keys.filter { $0.lowercased().contains(w) || w.contains($0.lowercased()) }
                print("    expected \"\(want)\": " + (hits.isEmpty ? "NOT FOUND"
                      : "found as \(hits.map { "\"\($0)\"" }.joined(separator: ", "))"))
            }
        }

        // ---- verdict
        //
        // Two things this used to get wrong, both of which made the verdict
        // line unreadable and had to be talked around in every report:
        //
        //  1. **It demanded a face in every render.** A doorway has no face. A
        //     jacket shoulder has no face. "face in only 0% of frames" was the
        //     headline on renders that were entirely correct, and the matrix
        //     that knew which scenes had people never passed that knowledge in.
        //     Zero faces is uninformative without an expectation, so it is now
        //     reported and not judged unless `--expect-face` says otherwise.
        //     *Intermittent* detection still fails either way: a face found in
        //     some frames and lost in others is an instability the render owns,
        //     whatever the scene contains.
        //
        //  2. **The persistent colour-blob count never once indicated a real
        //     artifact.** It fired at 28% red on a render of a red door and on
        //     124/124 frames of teal hair under greenhouse light. A blob that
        //     does not change is indistinguishable from an object that is
        //     simply that colour, and no amount of threshold tuning fixes that
        //     — it needs a reference the check does not have. It stays as a
        //     reported measure and leaves the verdict.
        //
        //     The frame-to-frame *delta* is the one that caught the real
        //     artifact (green excess 6.85% of frame, 52 events at 6 steps;
        //     0.00% and 0 events at 20), and it stays in the verdict.
        var problems: [String] = []
        if expectFace {
            if rate < 0.9 {
                problems.append(String(format: "face expected but found in only %.0f%% of frames",
                                       100 * rate))
            }
        } else if rate > 0 && rate < 0.9 {
            problems.append(String(format: "face detected intermittently (%.0f%% of frames) — "
                                   + "stable presence or clean absence, not both", 100 * rate))
        }
        if !jumps.isEmpty { problems.append("\(jumps.count) flash/blob event(s)") }

        print("")
        if problems.isEmpty {
            var good = [String]()
            if rate >= 0.9 { good.append("face present and stable") }
            else if rate == 0 { good.append("no face (not expected)") }
            good.append("no flashes")
            print("  VERDICT: " + good.joined(separator: ", "))
        } else {
            print("  VERDICT: " + problems.joined(separator: "; "))
        }
        if !blobFrames.isEmpty {
            print(String(format: "  (note: %d frame(s) carry a saturated channel over 1%% of "
                         + "the image. Not judged — a persistent colour cannot be told from "
                         + "an object that colour without a reference.)", blobFrames.count))
        }
    }
}
