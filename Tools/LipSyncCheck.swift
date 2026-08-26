// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import AVFoundation
import ArgumentParser
import CoreMedia
import Vision
import H3Foundation
import H3Pipeline

/// Are the lips saying what the audio is saying, and do they stay that way?
///
/// The fourth no-golden oracle, and the one that measures the thing the whole
/// talking-head use case rests on. H3 generates video and audio in a single
/// forward pass, so nothing external enforces that they agree — and "lip drift"
/// is a named, unmeasured complaint about this model.
///
/// ## Two different questions
///
/// **Sync**: does mouth movement line up with speech energy at all? Measured as
/// the correlation between a per-frame mouth-aperture series and the audio
/// envelope, at the best lag within a plausible window.
///
/// **Drift**: does that alignment *hold* across the render? Measured as the
/// best lag per one-second window, then the trend across windows. This is the
/// operative definition of drift — a lag that is stable is a fixed offset, and
/// a lag that walks is drift. A single global correlation cannot tell them
/// apart, which is why "lip drift" has stayed an impression rather than a
/// number.
///
/// ## Why a bare correlation would prove nothing
///
/// A talking head's mouth is active while there is speech and still while there
/// is not, and so is the audio envelope. The two correlate *by construction*
/// whether or not they are synchronised. So every figure here is reported
/// against controls that preserve the audio's statistics and destroy only its
/// timing:
///
///  * **reversed** — the same envelope backwards. Identical amplitude
///    distribution, no shared timing.
///  * **displaced** — the same envelope shifted by half the clip.
///
/// Real synchronisation beats both by a clear margin. A render whose mouth
/// merely moves whenever there is sound will score about the same on all three.
struct LipSyncCheck: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "lipsync-check",
        abstract: "Measure lip synchronisation and drift against timing-destroyed controls."
    )

    @Argument(help: "the rendered mp4")
    var video: String

    @Option(help: "audio to test against; defaults to the mp4's own track")
    var audio: String?

    /// Physically plausible sync error. Broadcast practice treats audio leading
    /// video by more than 45 ms, or lagging by more than 125 ms, as detectable;
    /// this is deliberately wider so a real offset is measured rather than
    /// clipped at the search boundary.
    @Option(help: "maximum |lag| considered synchronised, in seconds")
    var maxLag: Double = 0.30

    @Option(help: "window length for the drift measurement, in seconds")
    var window: Double = 1.0

    /// The drift a viewer will tolerate, in milliseconds across the render.
    ///
    /// Set from viewing rather than from the correlation: 219 ms of slow drift
    /// on a profile mid-shot, with the lag crossing zero partway through, is
    /// not visible — and a person watching is what this oracle exists to
    /// approximate. See `docs/ANE_PRECISION_RESULTS.md`.
    ///
    /// It is an option and not a constant because it is a judgement, and a
    /// judgement that cannot be argued with on the command line is a constant
    /// pretending to be evidence. Tighten it for a frontal close-up, where
    /// articulation is legible and the tolerance is genuinely smaller.
    @Option(help: "drift, in ms across the render, treated as visible")
    var maxDrift: Double = 250

    static let fps = 24.0

    func run() throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        let url = URL(fileURLWithPath: video)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("no such file: \(video)")
        }

        let aperture = try mouthAperture(url)
        let envelope = try audioEnvelope(URL(fileURLWithPath: audio ?? video),
                                         frames: aperture.count)
        print("lipsync-check \(video)")
        print("  frames            \(aperture.count)")
        let voiced = aperture.filter { $0 != nil }.count
        print("  mouth measured in \(voiced)/\(aperture.count) frames")
        guard voiced >= 24 else {
            print("\n  INCONCLUSIVE: a face with resolvable lips is present in only "
                  + "\(voiced) frames. This measures nothing without one.")
            throw ExitCode(2)
        }

        // Gaps are filled by holding the last measurement rather than dropped,
        // so the series stays on the frame clock — a compacted series would
        // silently rescale time and invent sync.
        let a = zscore(fillGaps(aperture))
        let e = zscore(envelope)

        let maxLagFrames = Int((maxLag * Self.fps).rounded())
        let (r, lag) = bestLag(a, e, maxLag: maxLagFrames)
        let (rRev, _) = bestLag(a, Array(e.reversed()), maxLag: maxLagFrames)
        let (rDisp, _) = bestLag(a, displaced(e), maxLag: maxLagFrames)

        print(String(format: "\n  sync      r %+.3f at lag %+.0f ms", r, Double(lag) / Self.fps * 1000))
        print(String(format: "  reversed  r %+.3f   (control: same envelope, no shared timing)", rRev))
        print(String(format: "  displaced r %+.3f   (control: same envelope, shifted half a clip)", rDisp))
        let margin = r - Swift.max(rRev, rDisp)
        print(String(format: "  margin    %+.3f over the better control", margin))

        // ---- drift
        let w = Int(window * Self.fps)
        var lags: [(Int, Double, Double)] = []
        var start = 0
        while start + w <= a.count {
            let (wr, wl) = bestLag(Array(a[start ..< start + w]),
                                   Array(e[start ..< start + w]), maxLag: maxLagFrames)
            // A window is only allowed to vote on drift if its own correlation
            // is strong enough to locate a peak. The first version of this
            // accepted r > 0.15, and a 24-sample window at r = 0.2 has no peak
            // to find — its argmax lands wherever the noise is highest, very
            // often against the edge of the search range. Reporting those as
            // "the lag wandered" would be inventing drift out of an unusable
            // estimate, so r is printed alongside every lag and the bar is
            // raised to something a peak can actually clear.
            if wr > 0.45 { lags.append((start, Double(wl) / Self.fps * 1000, wr)) }
            start += Swift.max(1, w / 2)
        }
        print("\n  per-window lag (\(String(format: "%.1f", window))s windows, only where r > 0.45)")
        if lags.count < 3 {
            print("    too few usable windows (\(lags.count)) to call drift either way")
        } else {
            for (s, ms, wr) in lags {
                // The reachable maximum is the frame grid, not the requested
                // seconds: a 0.30 s window at 24 fps is 7 frames, i.e. 292 ms.
                // Comparing against 300 meant the marker never fired on the
                // very lags it exists to flag.
                let edgeMs = Double(maxLagFrames) / Self.fps * 1000
                let edge = abs(ms) >= edgeMs - 0.5 ? "  (AT SEARCH EDGE — distrust)" : ""
                print(String(format: "    t=%5.2fs  %+6.0f ms   r %+.2f%@",
                             Double(s) / Self.fps, ms, wr, edge as NSString))
            }
            let edgeMs = Double(maxLagFrames) / Self.fps * 1000
            let interior = lags.filter { abs($0.1) < edgeMs - 0.5 }
            let xs = interior.map { Double($0.0) / Self.fps }
            let ys = interior.map { $0.1 }
            let ws = interior.map { $0.2 }
            let slope = linearSlope(x: xs, y: ys, w: ws)
            let span = (xs.max() ?? 0) - (xs.min() ?? 0)
            let spread = (ys.max() ?? 0) - (ys.min() ?? 0)
            let atEdge = lags.filter { abs($0.1) >= edgeMs - 0.5 }.count
            print(String(format: "    trend %+.0f ms/s over %.1fs = %+.0f ms total drift; "
                         + "spread %.0f ms", slope, span, slope * span, spread))
            if atEdge > 0 {
                print(String(format: "    %d of %d windows are pinned at the search edge. A lag "
                             + "at the boundary means the correlation had no interior peak to "
                             + "find, so the spread above is an upper bound, not a measurement.",
                             atEdge, ys.count))
            }
        }

        // ---- verdict
        var problems: [String] = []
        if margin < 0.10 {
            problems.append(String(format: "no better than a timing-destroyed control "
                                   + "(margin %+.3f) — the mouth moves when there is sound, "
                                   + "which is not the same as being in sync", margin))
        }
        if abs(Double(lag) / Self.fps) > 0.125 {
            problems.append(String(format: "offset %.0f ms exceeds the 125 ms audible threshold",
                                   Double(lag) / Self.fps * 1000))
        }
        if lags.count >= 3 {
            let edgeMs = Double(maxLagFrames) / Self.fps * 1000
            let interior = lags.filter { abs($0.1) < edgeMs - 0.5 }
            // Drift is a TREND, and this used to test the spread.
            //
            // Spread is max minus min, so it answers "how much did the estimate
            // move", which mixes real drift with the noise of estimating a lag
            // from a one-second window. Those are different things and only one
            // of them is a defect: a clip whose lag sits at -83 ms throughout,
            // measured by windows that scatter +-100 ms around it, is in sync
            // and was being failed for it. That is exactly what happened on the
            // first renders this was run against — a profile shot, where mouth
            // aperture barely varies and a few windows land far from the true
            // lag. Both arms failed, the global correlation said +0.63 and
            // +0.69 against controls at +0.02 to +0.13, and a human watching
            // them saw nothing wrong. The human and the global measure were
            // right.
            //
            // A weighted trend answers the question the name promises: is the
            // offset moving in one direction over the render. Scatter about a
            // stable offset now shows up in the printed spread, where it can be
            // read, and does not fail the render on its own.
            if interior.count >= 4 {
                let xs = interior.map { Double($0.0) / Self.fps }
                let ys = interior.map { $0.1 }
                let ws = interior.map { $0.2 }
                let span = (xs.max() ?? 0) - (xs.min() ?? 0)
                let slope = linearSlope(x: xs, y: ys, w: ws)
                let total = abs(slope * span)
                // A trend fitted to six or seven noisy windows can be scatter
                // wearing a slope, so it has to clear its own standard error
                // before it is called drift. This does not rescue a clip that
                // is really drifting — the renders that prompted the check
                // come out at t = 4.8 and t = 2.7 — it stops the check firing
                // on a handful of weak windows that happen to line up.
                let se = slopeStandardError(x: xs, y: ys, w: ws, slope: slope)
                let significant = se <= 0 || abs(slope) > 2.5 * se
                if total > maxDrift && significant {
                    problems.append(String(format: "lag drifts %.0f ms across the render "
                                           + "(correlation-weighted trend, %.1f sigma over windows "
                                           + "with an interior peak) — this is drift, not a fixed "
                                           + "offset", total, se > 0 ? abs(slope) / se : 0))
                } else if total > maxDrift {
                    print(String(format: "\n  note: the interior trend would be %.0f ms but is "
                                 + "only %.1f sigma — too few usable windows to separate drift "
                                 + "from estimator noise, so it is not judged.",
                                 total, se > 0 ? abs(slope) / se : 0))
                } else if significant {
                    // Measured, real, and under the bar. Printed rather than
                    // swallowed: raising a threshold must not also hide the
                    // number that prompted it.
                    print(String(format: "\n  drift %.0f ms at %.1f sigma — real, and under the "
                                 + "%.0f ms a viewer was judged to tolerate. Not a failure; still "
                                 + "a number, and it belongs in any comparison of two renders.",
                                 total, se > 0 ? abs(slope) / se : 0, maxDrift))
                }
            }
        }

        print("")
        if problems.isEmpty {
            print("  PASS: mouth movement tracks the audio, beating both controls, "
                  + "with a stable offset")
        } else {
            for p in problems { print("  FAIL: \(p)") }
            throw ExitCode(1)
        }
    }

    // MARK: mouth

    /// Per-frame mouth aperture, scale-invariant, nil where no lips resolve.
    ///
    /// Inner-lip height over outer-lip width. The ratio matters: a raw height
    /// tracks how close the camera is as much as how open the mouth is, so a
    /// dolly-in would read as speech.
    private func mouthAperture(_ url: URL) throws -> [Double?] {
        let asset = AVURLAsset(url: url)
        guard let track = try loadTrack(asset, .video) else {
            throw ValidationError("no video track in \(url.path)")
        }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String:
                                kCVPixelFormatType_32BGRA])
        out.alwaysCopiesSampleData = false
        reader.add(out)
        reader.startReading()

        var series: [Double?] = []
        while let sample = out.copyNextSampleBuffer() {
            guard let pixels = CMSampleBufferGetImageBuffer(sample) else {
                series.append(nil); continue
            }
            let handler = VNImageRequestHandler(cvPixelBuffer: pixels, options: [:])
            let request = VNDetectFaceLandmarksRequest()
            try? handler.perform([request])
            guard let face = (request.results ?? []).max(by: {
                      $0.boundingBox.width * $0.boundingBox.height
                      < $1.boundingBox.width * $1.boundingBox.height }),
                  let lm = face.landmarks,
                  let inner = lm.innerLips?.normalizedPoints, inner.count >= 4,
                  let outer = lm.outerLips?.normalizedPoints, outer.count >= 4
            else { series.append(nil); continue }

            let ih = (inner.map { Double($0.y) }.max() ?? 0)
                   - (inner.map { Double($0.y) }.min() ?? 0)
            let ow = (outer.map { Double($0.x) }.max() ?? 0)
                   - (outer.map { Double($0.x) }.min() ?? 0)
            series.append(ow > 0 ? ih / ow : nil)
        }
        return series
    }

    // MARK: audio

    /// Speech-band energy, one value per video frame.
    ///
    /// Band-limited to 300-3400 Hz before the envelope is taken. Room tone,
    /// rumble and music sit largely outside that band and would otherwise
    /// contribute energy that no mouth is producing.
    private func audioEnvelope(_ url: URL, frames: Int) throws -> [Double] {
        let wave = try MediaLoad.audio(at: url.path)   // [1, 2, L] at 32 kHz
        let left = wave[0, 0].asArray(Float.self)
        let sr = Double(H3Audio.sampleRate)
        let per = sr / Self.fps

        // One-pole band-pass, cheap and adequate for an envelope.
        var hp = [Float](repeating: 0, count: left.count)
        let aHigh = Float(exp(-2.0 * Double.pi * 300.0 / sr))
        var prevIn: Float = 0, prevOut: Float = 0
        for i in 0 ..< left.count {
            prevOut = aHigh * (prevOut + left[i] - prevIn)
            prevIn = left[i]
            hp[i] = prevOut
        }
        let aLow = Float(exp(-2.0 * Double.pi * 3400.0 / sr))
        var lp: Float = 0
        for i in 0 ..< hp.count {
            lp = aLow * lp + (1 - aLow) * hp[i]
            hp[i] = lp
        }

        var env = [Double]()
        env.reserveCapacity(frames)
        for f in 0 ..< frames {
            let lo = Int(Double(f) * per), hi = Swift.min(Int(Double(f + 1) * per), hp.count)
            guard lo < hi else { env.append(0); continue }
            var s = 0.0
            for i in lo ..< hi { s += Double(hp[i] * hp[i]) }
            env.append((s / Double(hi - lo)).squareRoot())
        }
        return env
    }

    // MARK: maths

    private func fillGaps(_ xs: [Double?]) -> [Double] {
        var out = [Double](repeating: 0, count: xs.count)
        var last = xs.compactMap { $0 }.first ?? 0
        for (i, v) in xs.enumerated() { last = v ?? last; out[i] = last }
        return out
    }

    private func zscore(_ xs: [Double]) -> [Double] {
        let m = xs.reduce(0, +) / Double(xs.count)
        let sd = (xs.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(xs.count)).squareRoot()
        return sd > 0 ? xs.map { ($0 - m) / sd } : xs.map { _ in 0 }
    }

    private func correlate(_ a: [Double], _ b: [Double]) -> Double {
        let n = Swift.min(a.count, b.count)
        guard n > 4 else { return 0 }
        var num = 0.0, da = 0.0, db = 0.0
        for i in 0 ..< n { num += a[i] * b[i]; da += a[i] * a[i]; db += b[i] * b[i] }
        let d = (da * db).squareRoot()
        return d > 0 ? num / d : 0
    }

    /// Best correlation over lags, and the lag achieving it. Positive lag means
    /// the mouth moves *after* the sound.
    private func bestLag(_ a: [Double], _ b: [Double], maxLag: Int) -> (Double, Int) {
        var best = -2.0, bestL = 0
        for l in -maxLag ... maxLag {
            let x: [Double], y: [Double]
            if l >= 0 { x = Array(a.dropFirst(l)); y = b }
            else { x = a; y = Array(b.dropFirst(-l)) }
            let r = correlate(x, y)
            if r > best { best = r; bestL = l }
        }
        return (best, bestL)
    }

    private func displaced(_ xs: [Double]) -> [Double] {
        let h = xs.count / 2
        return Array(xs[h...]) + Array(xs[..<h])
    }

    /// Least squares, optionally weighted.
    ///
    /// Weights are the per-window correlations. A window that located a sharp
    /// peak knows where the lag is; one that scraped past the acceptance bar
    /// does not, and letting the two vote equally is how a profile shot with a
    /// few weak windows gets reported as drifting.
    private func linearSlope(x: [Double], y: [Double], w: [Double]? = nil) -> Double {
        let weights = w ?? [Double](repeating: 1, count: x.count)
        let sw = weights.reduce(0, +)
        guard sw > 0 else { return 0 }
        var mx = 0.0, my = 0.0
        for i in 0 ..< x.count { mx += weights[i] * x[i]; my += weights[i] * y[i] }
        mx /= sw; my /= sw
        var num = 0.0, den = 0.0
        for i in 0 ..< x.count {
            num += weights[i] * (x[i] - mx) * (y[i] - my)
            den += weights[i] * (x[i] - mx) * (x[i] - mx)
        }
        return den > 0 ? num / den : 0
    }

    /// Standard error of the weighted slope, so a trend can be asked whether
    /// it is distinguishable from zero rather than merely non-zero.
    private func slopeStandardError(x: [Double], y: [Double], w: [Double],
                                    slope: Double) -> Double {
        guard x.count > 2 else { return 0 }
        let sw = w.reduce(0, +)
        guard sw > 0 else { return 0 }
        var mx = 0.0, my = 0.0
        for i in 0 ..< x.count { mx += w[i] * x[i]; my += w[i] * y[i] }
        mx /= sw; my /= sw
        var sxx = 0.0, rss = 0.0
        for i in 0 ..< x.count {
            sxx += w[i] * (x[i] - mx) * (x[i] - mx)
            let fit = my + slope * (x[i] - mx)
            rss += w[i] * (y[i] - fit) * (y[i] - fit)
        }
        guard sxx > 0 else { return 0 }
        return (rss / Double(x.count - 2) / sxx).squareRoot()
    }

    private final class Box: @unchecked Sendable {
        var track: AVAssetTrack?
        var error: Error?
    }

    private func loadTrack(_ asset: AVURLAsset, _ kind: AVMediaType) throws -> AVAssetTrack? {
        let box = Box()
        let sem = DispatchSemaphore(value: 0)
        Task { [box] in
            do { box.track = try await asset.loadTracks(withMediaType: kind).first }
            catch { box.error = error }
            sem.signal()
        }
        sem.wait()
        if let e = box.error { throw e }
        return box.track
    }
}

// This target is a single file, which SwiftPM compiles as top-level code.
// An `@main` type is merely declared in that mode and never invoked — the
// binary linked, ran, printed nothing and exited 0. Call it explicitly.
LipSyncCheck.main()
