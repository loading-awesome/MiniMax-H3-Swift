import Foundation
import Testing
@testable import MiniMaxH3

@Suite("enterprise render facade")
struct RenderEngineTests {
    @Test("faithful output is the default")
    func faithfulDefault() {
        let request = RenderRequest(prompt: "test", videoOutput: URL(fileURLWithPath: "/tmp/out.mp4"))
        #expect(request.qualityProfile == .faithful)
        #expect(request.cacheThreshold == 0)
    }

    @Test("the actor refuses a second admitted render")
    func busyAdmission() async throws {
        let gate = Gate()
        let engine = RenderEngine(models: Self.dummyModels()) { _, request, _, _, _, _ in
            await gate.wait()
            return Self.result(for: request)
        }
        let request = RenderRequest(prompt: "test", videoOutput: URL(fileURLWithPath: "/tmp/out.mp4"))
        let first = try await engine.start(request)

        do {
            _ = try await engine.start(request)
            Issue.record("second render was admitted")
        } catch let error as H3Error {
            #expect(error.code == .engineBusy)
        }

        await gate.open()
        _ = try await first.value()
        #expect(await engine.activeJob == nil)
    }

    @Test("staged media publishes only at commit")
    func atomicCommit() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "h3-output-test-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }

        let video = root.appending(path: "final.mp4")
        let audio = root.appending(path: "final.wav")
        let request = RenderRequest(prompt: "test", videoOutput: video, audioOutput: audio)
        let transaction = try OutputTransaction(jobID: UUID(), request: request)
        try Data("video".utf8).write(to: transaction.stagedRequest.videoOutput)
        try Data("audio".utf8).write(to: transaction.stagedRequest.audioOutput!)

        #expect(!FileManager.default.fileExists(atPath: video.path))
        let published = try transaction.commit(Self.result(for: transaction.stagedRequest))
        #expect(try Data(contentsOf: published.video) == Data("video".utf8))
        #expect(try Data(contentsOf: published.audio!) == Data("audio".utf8))
        #expect(!FileManager.default.fileExists(atPath: transaction.workspace.path))
    }

    @Test("a mux fallback publishes recovery audio before the video commit marker")
    func recoveryCommit() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "h3-recovery-test-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }

        let request = RenderRequest(prompt: "test", videoOutput: root.appending(path: "final.mp4"))
        let transaction = try OutputTransaction(jobID: UUID(), request: request)
        try Data("video".utf8).write(to: transaction.stagedRequest.videoOutput)
        try Data("audio".utf8).write(to: transaction.stagedRequest.audioOutput!)
        let staged = RenderResult(video: transaction.stagedRequest.videoOutput,
                                  audio: transaction.stagedRequest.audioOutput,
                                  frameCount: 124, width: 864, height: 480, seconds: 5,
                                  timings: .init(), cacheSummary: nil, muxedAudio: false)

        let published = try transaction.commit(staged)
        #expect(FileManager.default.fileExists(atPath: published.video.path))
        #expect(published.audio?.lastPathComponent == "final.recovery.wav")
        #expect(try Data(contentsOf: published.audio!) == Data("audio".utf8))
    }

    @Test("a preflight failure writes a receipt without prompt contents")
    func failureReceipt() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "h3-receipt-test-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }

        let file = ModelFile(url: URL(fileURLWithPath: "/dev/null"))
        let engine = RenderEngine(models: ModelSet(
            dit: file, textEncoder: file, tokenizer: root,
            videoVAE: file, audioVAE: file))
        let output = root.appending(path: "failed.mp4")
        let request = RenderRequest(prompt: "SECRET PROMPT CONTENT", videoOutput: output,
                                    width: 864, height: 480)
        do {
            _ = try await engine.render(request)
            Issue.record("invalid model set reached the renderer")
        } catch {
            // The failure is the subject of the receipt assertions below.
        }

        let receiptURL = root.appending(path: "failed.h3-receipt.json")
        let data = try Data(contentsOf: receiptURL)
        let text = String(decoding: data, as: UTF8.self)
        #expect(!text.contains("SECRET PROMPT CONTENT"))
        #expect(!text.contains(root.path))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let receipt = try decoder.decode(RenderReceipt.self, from: data)
        #expect(receipt.status == .failed)
        #expect(receipt.errorCode != nil)
        #expect(receipt.request.promptCharacters == 21)
    }

    @Test("stable error codes do not depend on prose")
    func stableErrors() {
        let error = H3Error.outputExists(path: "/tmp/out.mp4")
        #expect(error.code.rawValue == "H3-3002")
        #expect(error.errorDescription?.contains("H3-3002") == true)
    }

    private static func dummyModels() -> ModelSet {
        let file = ModelFile(url: URL(fileURLWithPath: "/dev/null"))
        return ModelSet(dit: file, textEncoder: file,
                        tokenizer: URL(fileURLWithPath: "/tmp"),
                        videoVAE: file, audioVAE: file)
    }

    private static func result(for request: RenderRequest) -> RenderResult {
        RenderResult(video: request.videoOutput, audio: request.audioOutput,
                     frameCount: 124, width: 864, height: 480, seconds: 5,
                     timings: .init(), cacheSummary: nil)
    }
}

private actor Gate {
    private var isOpen = false
    private var continuations: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { continuations.append($0) }
    }

    func open() {
        isOpen = true
        let waiting = continuations
        continuations.removeAll()
        for continuation in waiting { continuation.resume() }
    }
}
