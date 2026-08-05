import Foundation
import CryptoKit
import OSLog
import Darwin
import H3Foundation
import H3Hardware
import H3Catalog
import H3Pipeline

/// Owns admission and lifecycle for the process's model runtime.
public actor RenderEngine {
    typealias Operation = @Sendable (
        UUID, RenderRequest, ModelSet, Options, RenderCancellation,
        AsyncStream<RenderEvent>.Continuation
    ) async throws -> RenderResult

    public struct Options: Sendable {
        public var allowApproximateWeights: Bool
        public var memoryMarginFraction: Double
        public var verifySuppliedDigests: Bool

        public init(allowApproximateWeights: Bool = false,
                    memoryMarginFraction: Double = 0.15,
                    verifySuppliedDigests: Bool = true) {
            self.allowApproximateWeights = allowApproximateWeights
            self.memoryMarginFraction = memoryMarginFraction
            self.verifySuppliedDigests = verifySuppliedDigests
        }
    }

    private let models: ModelSet
    private let options: Options
    private let operation: Operation
    private var activeJobID: UUID?

    public init(models: ModelSet, options: Options = .init()) {
        self.models = models
        self.options = options
        operation = { id, request, models, options, cancellation, events in
            try await RenderOperation.execute(
                jobID: id, request: request, models: models, options: options,
                cancellation: cancellation, events: events)
        }
    }

    init(models: ModelSet, options: Options = .init(),
         operation: @escaping Operation) {
        self.models = models
        self.options = options
        self.operation = operation
    }

    public var activeJob: UUID? { activeJobID }

    /// Admit work immediately or refuse it with a stable error. A model this
    /// large must never create a second renderer merely because two callers
    /// raced to an async API.
    public func start(_ request: RenderRequest) throws -> RenderJob {
        if let activeJobID { throw H3Error.engineBusy(activeJob: activeJobID.uuidString) }

        let id = UUID()
        activeJobID = id
        let cancellation = RenderCancellation()
        let (events, continuation) = AsyncStream<RenderEvent>.makeStream(
            bufferingPolicy: .bufferingNewest(256))
        continuation.yield(.admitted(jobID: id))

        // The MLX forward is a long synchronous foreign-runtime call. Running
        // it on the actor would prevent the actor from rejecting concurrent
        // admission and reporting state. This is the single audited detached
        // boundary: admission above proves there is at most one, the job owns
        // its handle, and every terminal path returns to the actor to release
        // the slot.
        let task = Task.detached(priority: .userInitiated) {
            [models, options, operation, self] in
            defer { continuation.finish() }
            do {
                try Task.checkCancellation()
                let result = try await operation(
                    id, request, models, options, cancellation, continuation)
                continuation.yield(.completed(result))
                await self.finished(id)
                return result
            } catch {
                let reported = RenderOperation.report(error)
                continuation.yield(.failed(code: reported.code, message: reported.message))
                await self.finished(id)
                throw error
            }
        }
        return RenderJob(id: id, events: events, task: task, cancellation: cancellation)
    }

    public func render(_ request: RenderRequest) async throws -> RenderResult {
        let job = try start(request)
        return try await job.value()
    }

    private func finished(_ id: UUID) {
        if activeJobID == id { activeJobID = nil }
    }
}

public struct RenderJob: Sendable {
    public let id: UUID
    public let events: AsyncStream<RenderEvent>
    private let task: Task<RenderResult, Error>
    private let cancellation: RenderCancellation

    fileprivate init(id: UUID, events: AsyncStream<RenderEvent>,
                     task: Task<RenderResult, Error>,
                     cancellation: RenderCancellation) {
        self.id = id
        self.events = events
        self.task = task
        self.cancellation = cancellation
    }

    public func cancel() {
        cancellation.cancel()
        task.cancel()
    }

    public func value() async throws -> RenderResult { try await task.value }
}

private enum RenderOperation {
    struct ReportedError {
        let code: String
        let message: String
    }

    static func report(_ error: Error) -> ReportedError {
        if let h3 = error as? H3Error {
            return ReportedError(code: h3.code.rawValue, message: h3.description)
        }
        if error is RenderCancelled || error is CancellationError {
            return ReportedError(code: "H3-0001", message: String(describing: error))
        }
        return ReportedError(code: "H3-9000", message: String(describing: error))
    }

    static func execute(jobID: UUID, request: RenderRequest, models: ModelSet,
                        options: RenderEngine.Options,
                        cancellation: RenderCancellation,
                        events: AsyncStream<RenderEvent>.Continuation) async throws -> RenderResult {
        let logger = Logger(subsystem: "com.loading-awesome.MiniMaxH3", category: "render")
        let machine = Machine.detect()
        let dimensions = try request.dimensions()
        var receipt = RenderReceipt(
            jobID: jobID,
            request: .init(
                mode: request.mode.rawValue, width: dimensions.width, height: dimensions.height,
                seconds: request.seconds, steps: request.steps, seed: request.seed,
                promptCharacters: request.prompt.count,
                qualityProfile: request.qualityProfile.rawValue,
                cacheThreshold: request.cacheThreshold,
                referenceImages: request.referenceImages.count,
                referenceVideos: request.referenceVideos.count,
                referenceAudio: request.referenceAudio.count),
            environment: .init(
                libraryVersion: MiniMaxH3.version,
                operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                machine: machine.summary,
                availableMemoryBytes: Machine.availableBytes()))

        let receiptURL = receiptLocation(for: request.videoOutput)
        var transaction: OutputTransaction?
        do {
            let processLock = try ProcessRenderLock()
            defer { withExtendedLifetime(processLock) {} }
            let fingerprints = try Preflight.run(
                request: request, models: models, options: options)
            receipt.checkpoints = fingerprints
            if request.usesApproximateSampling {
                let warning = "approximate render: quality=\(request.qualityProfile.rawValue), "
                    + "cache_threshold=\(request.cacheThreshold)"
                receipt.warnings.append(warning)
                events.yield(.warning(warning))
            }
            if models.precision == .prunedBF16 || models.precision == .prunedInt8 {
                let warning = "approximate checkpoint: \(models.precision.rawValue) replaces AdaLN with a low-rank curve"
                receipt.warnings.append(warning)
                events.yield(.warning(warning))
            }

            let tx = try OutputTransaction(jobID: jobID, request: request)
            transaction = tx
            let pressure = MemoryPressureMonitor {
                cancellation.cancel()
                events.yield(.warning("critical system memory pressure; cancellation requested"))
            }
            pressure.start()
            defer { pressure.stop() }
            events.yield(.diagnostic("workspace prepared; final artifacts publish atomically"))
            logger.info("render admitted job=\(jobID.uuidString, privacy: .public)")

            let pipelineResult = try await PipelineRuntime.render(
                request: tx.stagedRequest,
                checkpoints: .init(
                    dit: models.dit.url, textEncoder: models.textEncoder.url,
                    tokenizer: models.tokenizer, videoVAE: models.videoVAE.url,
                    audioVAE: models.audioVAE.url),
                progress: { events.yield(.progress($0)) },
                cancellation: cancellation,
                log: {
                    logger.info("job=\(jobID.uuidString, privacy: .public) \($0, privacy: .private)")
                    events.yield(.diagnostic($0))
                })

            let result = try tx.commit(pipelineResult)
            transaction = nil
            receipt.status = .succeeded
            receipt.finishedAt = .now
            receipt.timings = timings(result.timings)
            receipt.outputs = outputRecords(result)
            if !result.muxedAudio {
                receipt.warnings.append("audio mux failed; recovery WAV retained")
            }
            try write(receipt, to: receiptURL)
            logger.info("render completed job=\(jobID.uuidString, privacy: .public)")
            return result
        } catch {
            let recovery = try? transaction?.recoverAudio()
            transaction?.discard()
            let reported = report(error)
            receipt.status = (error is RenderCancelled || error is CancellationError)
                ? .cancelled : .failed
            receipt.finishedAt = .now
            receipt.errorCode = reported.code
            receipt.errorMessage = receiptSafeMessage(
                reported.message, request: request, models: models)
            if let recovery {
                receipt.outputs.append(fileRecord(kind: "recovery-audio", url: recovery))
                receipt.warnings.append("recovery WAV retained after terminal failure")
            }
            try? write(receipt, to: receiptURL)
            logger.error("render failed job=\(jobID.uuidString, privacy: .public) code=\(reported.code, privacy: .public)")
            throw error
        }
    }

    private static func receiptLocation(for video: URL) -> URL {
        video.deletingPathExtension().appendingPathExtension("h3-receipt.json")
    }

    private static func write(_ receipt: RenderReceipt, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try encoder.encode(receipt).write(to: url, options: .atomic)
    }

    private static func timings(_ value: RenderResult.Timings) -> [String: TimeInterval] {
        ["text_conditioning": value.textConditioning,
         "condition_encoding": value.conditionEncoding,
         "sampling": value.sampling,
         "audio_decode": value.audioDecode,
         "video_decode": value.videoDecode,
         "pixel_pack": value.pixelPack,
         "mux": value.mux,
         "total": value.total]
    }

    private static func outputRecords(_ result: RenderResult) -> [RenderReceipt.Output] {
        var records = [fileRecord(kind: "video", url: result.video)]
        if let audio = result.audio { records.append(fileRecord(kind: "audio", url: audio)) }
        return records
    }

    private static func fileRecord(kind: String, url: URL) -> RenderReceipt.Output {
        let size = ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?
            .uint64Value ?? 0
        return .init(kind: kind, filename: url.lastPathComponent, sizeBytes: size)
    }

    private static func receiptSafeMessage(_ message: String, request: RenderRequest,
                                           models: ModelSet) -> String {
        var result = message
        var urls = request.referenceImages + request.referenceVideos + request.referenceAudio
        urls.append(contentsOf: request.referenceVideoSoundtracks.compactMap { $0 })
        urls.append(contentsOf: [request.firstFrame, request.lastFrame,
                                 request.conditioningNoise, request.audioOutput].compactMap { $0 })
        urls.append(request.videoOutput)
        urls.append(contentsOf: [models.dit.url, models.textEncoder.url,
                                 models.videoVAE.url, models.audioVAE.url, models.tokenizer])
        for url in urls.sorted(by: { $0.path.count > $1.path.count }) where !url.path.isEmpty {
            result = result.replacingOccurrences(of: url.path, with: url.lastPathComponent)
        }
        return result
    }
}

private enum Preflight {
    static func run(request: RenderRequest, models: ModelSet,
                    options: RenderEngine.Options) throws
        -> [RenderReceipt.Checkpoint] {
        try Task.checkCancellation()
        try request.validate()
        guard (0 ... 0.5).contains(options.memoryMarginFraction) else {
            throw H3Error.invalidRequest(
                rule: "memory margin out of range",
                detail: "\(options.memoryMarginFraction); expected 0 through 0.5",
                remedy: "use the default 0.15 unless a measured deployment profile says otherwise.")
        }
        try MetalLibrary.preflight()

        let outputParent = request.videoOutput.standardizedFileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: outputParent, withIntermediateDirectories: true)
        guard FileManager.default.isWritableFile(atPath: outputParent.path) else {
            throw H3Error.unreadable(path: outputParent.path, reason: "output directory is not writable")
        }

        var finalURLs = [request.videoOutput]
        if let audioOutput = request.audioOutput { finalURLs.append(audioOutput) }
        var inputURLs = request.referenceImages
        inputURLs.append(contentsOf: request.referenceVideos)
        inputURLs.append(contentsOf: request.referenceAudio)
        inputURLs.append(contentsOf: request.referenceVideoSoundtracks.compactMap { $0 })
        inputURLs.append(contentsOf: [request.firstFrame, request.lastFrame,
                                      request.conditioningNoise].compactMap { $0 })
        for input in inputURLs {
            guard FileManager.default.isReadableFile(atPath: input.path) else {
                throw H3Error.unreadable(path: input.path, reason: "file is missing or not readable")
            }
            if finalURLs.contains(where: { $0.standardizedFileURL == input.standardizedFileURL }) {
                throw H3Error.invalidRequest(
                    rule: "output aliases an input", detail: input.path,
                    remedy: "choose a distinct final output path; publishing is atomic but inputs are immutable.")
            }
        }
        for output in finalURLs where FileManager.default.fileExists(atPath: output.path)
            && !request.overwriteOutput {
            throw H3Error.outputExists(path: output.path)
        }

        let tokenizerFiles = ["vocab.json", "merges.txt", "tokenizer_config.json"]
        for name in tokenizerFiles {
            let url = models.tokenizer.appending(path: name)
            guard FileManager.default.isReadableFile(atPath: url.path) else {
                throw H3Error.checkpointMissing(role: "tokenizer/\(name)", path: url.path)
            }
        }

        let identity = try CheckpointIdentity.identify(url: models.dit.url)
        try identity.validate(forMode: request.mode,
                              allowApproximate: options.allowApproximateWeights)

        let dimensions = try request.dimensions()
        let geometry = LatentGeometry(width: dimensions.width,
                                      height: dimensions.height,
                                      length: request.seconds * H3Video.fps)
        let tokens = geometry.videoTokens + geometry.audioTokens + 512
        let precision = memoryPrecision(models.precision)
        let available = Machine.availableBytes()
        let plan = MemoryPlan.plan(precision: precision, packedTokens: tokens,
                                   availableBytes: available,
                                   marginFraction: options.memoryMarginFraction)
        let textEncoderSize = ((try? FileManager.default.attributesOfItem(
            atPath: models.textEncoder.url.path)[.size]) as? NSNumber)?.uint64Value ?? 0
        let actualTextPhase = textEncoderSize + 2_000_000_000
        let actualPeak = max(plan.peakBytes, actualTextPhase)
        let requiredWithMargin = UInt64(Double(actualPeak) * (1 + options.memoryMarginFraction))
        guard available > requiredWithMargin else {
            throw H3Error.insufficientMemory(
                needGB: Double(requiredWithMargin) / 1e9,
                availableGB: plan.availableGB,
                detail: "resolved \(models.precision.rawValue) plan at \(tokens) packed tokens; "
                    + "close other memory-heavy work or request a smaller supported shape.")
        }

        let frameCount = UInt64(geometry.frameCount)
        let rawVideoBytes = UInt64(dimensions.width * dimensions.height) * frameCount * 4
        let audioBytes = UInt64(request.seconds * H3Audio.sampleRate * 2 * 2)
        let requiredDisk = rawVideoBytes + audioBytes + 1_000_000_000
        let values = try outputParent.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        if let capacity = values.volumeAvailableCapacityForImportantUsage,
           capacity >= 0, UInt64(capacity) < requiredDisk {
            throw H3Error.unreadable(
                path: outputParent.path,
                reason: "needs at least \(requiredDisk) bytes of working disk capacity; "
                    + "only \(capacity) bytes are available")
        }

        return try [("dit", models.dit), ("text-encoder", models.textEncoder),
                    ("video-vae", models.videoVAE), ("audio-vae", models.audioVAE)]
            .map { try fingerprint(role: $0.0, file: $0.1,
                                   verify: options.verifySuppliedDigests) }
    }

    private static func memoryPrecision(_ value: ModelSet.Precision) -> MemoryPlan.Precision {
        switch value {
        case .bf16: .bf16
        case .int8: .int8
        case .prunedBF16: .prunedBF16
        case .prunedInt8: .prunedInt8
        }
    }

    private static func fingerprint(role: String, file: ModelFile, verify: Bool) throws
        -> RenderReceipt.Checkpoint {
        let attributes: [FileAttributeKey: Any]
        do { attributes = try FileManager.default.attributesOfItem(atPath: file.url.path) }
        catch { throw H3Error.checkpointMissing(role: role, path: file.url.path) }
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        guard size > 8 else {
            throw H3Error.unreadable(path: file.url.path, reason: "checkpoint is empty or truncated")
        }
        _ = try Safetensors.Archive(url: file.url, headerOnly: true)

        guard verify, let expected = file.expectedSHA256 else {
            return .init(role: role, filename: file.url.lastPathComponent,
                         sizeBytes: size, sha256: nil, verification: .notRequested)
        }
        guard expected.count == 64, expected.allSatisfy({ $0.isHexDigit }) else {
            throw H3Error.invalidRequest(
                rule: "invalid checkpoint digest", detail: "\(role) SHA-256 is not 64 hex characters",
                remedy: "copy the lowercase SHA-256 from the trusted model manifest.")
        }
        let actual = try sha256(file.url)
        guard actual == expected else {
            throw H3Error.unreadable(path: file.url.path,
                                     reason: "SHA-256 mismatch: expected \(expected), got \(actual)")
        }
        return .init(role: role, filename: file.url.lastPathComponent,
                     sizeBytes: size, sha256: actual, verification: .verified)
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 * 1024 * 1024), !chunk.isEmpty {
            try Task.checkCancellation()
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

final class OutputTransaction {
    let finalVideo: URL
    let finalAudio: URL?
    let recoveryAudio: URL
    let workspace: URL
    let stagedRequest: RenderRequest
    private let overwrite: Bool

    init(jobID: UUID, request: RenderRequest) throws {
        finalVideo = request.videoOutput.standardizedFileURL
        finalAudio = request.audioOutput?.standardizedFileURL
        overwrite = request.overwriteOutput
        let parent = finalVideo.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        workspace = parent.appending(path: ".h3-\(jobID.uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: false)
        recoveryAudio = workspace.appending(path: "recovery.wav")
        var staged = request
        staged.videoOutput = workspace.appending(path: "render.mp4")
        staged.audioOutput = recoveryAudio
        staged.overwriteOutput = true
        stagedRequest = staged
    }

    func commit(_ staged: RenderResult) throws -> RenderResult {
        var publishedAudio: URL?
        if let finalAudio {
            try promote(recoveryAudio, to: finalAudio)
            publishedAudio = finalAudio
        } else if !staged.muxedAudio {
            let recovery = uniqueRecoveryURL()
            try promote(recoveryAudio, to: recovery)
            publishedAudio = recovery
        }
        // The video is the transaction's commit marker and is promoted last.
        // Seeing the final mp4 therefore means every requested sidecar was
        // already published successfully.
        try promote(staged.video, to: finalVideo)
        discard()
        return RenderResult(video: finalVideo, audio: publishedAudio,
                            frameCount: staged.frameCount, width: staged.width,
                            height: staged.height, seconds: staged.seconds,
                            timings: staged.timings, cacheSummary: staged.cacheSummary,
                            muxedAudio: staged.muxedAudio)
    }

    func recoverAudio() throws -> URL? {
        guard FileManager.default.fileExists(atPath: recoveryAudio.path) else { return nil }
        let destination = uniqueRecoveryURL()
        try promote(recoveryAudio, to: destination)
        return destination
    }

    func discard() {
        try? FileManager.default.removeItem(at: workspace)
    }

    private func uniqueRecoveryURL() -> URL {
        let base = finalVideo.deletingPathExtension()
        let first = base.appendingPathExtension("recovery.wav")
        if !FileManager.default.fileExists(atPath: first.path) { return first }
        return base.appendingPathExtension("\(UUID().uuidString).recovery.wav")
    }

    private func promote(_ source: URL, to destination: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            guard overwrite else { throw H3Error.outputExists(path: destination.path) }
            _ = try fm.replaceItemAt(destination, withItemAt: source)
        } else {
            try fm.moveItem(at: source, to: destination)
        }
    }

    deinit { discard() }
}

/// Advisory lock shared by every process for the current user. The actor stops
/// races through one engine instance; this stops a second CLI process or a
/// second engine instance from loading another 66 GB DiT concurrently.
private final class ProcessRenderLock {
    private let descriptor: Int32

    init() throws {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "minimax-h3-render-\(getuid()).lock")
        descriptor = open(url.path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw H3Error.unreadable(path: url.path, reason: "could not create render lock")
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            throw H3Error.engineBusy(activeJob: "another MiniMax-H3 process")
        }
    }

    deinit {
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}

/// `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` is the platform's only pressure event
/// API. The callback does one thread-safe operation: flip the cancellation
/// token. Model work observes it at the next safe boundary.
private final class MemoryPressureMonitor: @unchecked Sendable {
    private let source: DispatchSourceMemoryPressure
    private let onCritical: @Sendable () -> Void

    init(onCritical: @escaping @Sendable () -> Void) {
        self.onCritical = onCritical
        source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: .global(qos: .utility))
        source.setEventHandler { [weak self] in
            guard let self, self.source.data.contains(.critical) else { return }
            self.onCritical()
        }
    }

    func start() { source.activate() }
    func stop() { source.cancel() }
}
