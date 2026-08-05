// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Where a render has got to, and the way to stop it.
///
/// **In the API from the start, because retrofitting it is invasive.** A
/// twenty-minute call with no progress and no way to cancel is not shippable,
/// and the only place a cancellation can be observed cheaply is between sampler
/// steps — which means the check has to be threaded through the loop rather
/// than bolted on around it.
public struct RenderProgress: Sendable {

    public enum Phase: String, Sendable, CaseIterable {
        case textConditioning
        case conditionEncoding
        case sampling
        case decoding
        case writing
    }

    public let phase: Phase
    /// Completed units within the phase, and the total. Only sampling has a
    /// meaningful count; the others report 0/0 and lean on `detail`.
    public let completed: Int
    public let total: Int
    public let detail: String
    /// Seconds since the render began.
    public let elapsed: TimeInterval

    public init(phase: Phase, completed: Int = 0, total: Int = 0,
                detail: String = "", elapsed: TimeInterval = 0) {
        self.phase = phase
        self.completed = completed
        self.total = total
        self.detail = detail
        self.elapsed = elapsed
    }

    public var fraction: Double? {
        total > 0 ? Double(completed) / Double(total) : nil
    }
}

/// A cancellation flag the caller can set from another thread.
///
/// Deliberately not `Task.isCancelled`: the sampler loop is synchronous and
/// compute-bound, so it is not inside a Swift concurrency task, and a caller
/// driving it from a UI thread needs something they can flip without one.
public final class RenderCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    public init() {}

    public func cancel() {
        lock.lock(); cancelled = true; lock.unlock()
    }

    public var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }; return cancelled
    }
}

/// Raised when a render stops because the caller asked it to.
///
/// A distinct type rather than an `H3Error` case: cancellation is not a
/// failure, and a caller catching `H3Error` to report a problem should not
/// report this one.
public struct RenderCancelled: Error, CustomStringConvertible, Sendable {
    public let phase: RenderProgress.Phase
    public let detail: String
    public init(phase: RenderProgress.Phase, detail: String) {
        self.phase = phase
        self.detail = detail
    }
    public var description: String { "render cancelled during \(phase.rawValue): \(detail)" }
}

/// What a completed render produced, and what it cost.
///
/// The timings are reported rather than merely printed because guessing at them
/// is what once made a hung writer look like a slow CPU.
public struct RenderResult: Sendable {
    public let video: URL
    public let audio: URL?
    public let frameCount: Int
    public let width: Int
    public let height: Int
    public let seconds: Double
    public let timings: Timings
    /// The cache's own account of what it skipped, when one was running.
    public let cacheSummary: String?
    /// False only when audio muxing failed and the pipeline deliberately
    /// retained a playable video plus a recovery WAV.
    public let muxedAudio: Bool

    public struct Timings: Sendable {
        public var textConditioning: TimeInterval = 0
        public var conditionEncoding: TimeInterval = 0
        public var sampling: TimeInterval = 0
        public var audioDecode: TimeInterval = 0
        public var videoDecode: TimeInterval = 0
        public var pixelPack: TimeInterval = 0
        public var mux: TimeInterval = 0
        public init() {}

        public init(textConditioning: TimeInterval, conditionEncoding: TimeInterval,
                    sampling: TimeInterval, audioDecode: TimeInterval,
                    videoDecode: TimeInterval, pixelPack: TimeInterval,
                    mux: TimeInterval) {
            self.textConditioning = textConditioning
            self.conditionEncoding = conditionEncoding
            self.sampling = sampling
            self.audioDecode = audioDecode
            self.videoDecode = videoDecode
            self.pixelPack = pixelPack
            self.mux = mux
        }

        public var total: TimeInterval {
            textConditioning + conditionEncoding + sampling
                + audioDecode + videoDecode + pixelPack + mux
        }
    }

    public init(video: URL, audio: URL?, frameCount: Int, width: Int,
                height: Int, seconds: Double, timings: Timings,
                cacheSummary: String?, muxedAudio: Bool = true) {
        self.video = video
        self.audio = audio
        self.frameCount = frameCount
        self.width = width
        self.height = height
        self.seconds = seconds
        self.timings = timings
        self.cacheSummary = cacheSummary
        self.muxedAudio = muxedAudio
    }
}
