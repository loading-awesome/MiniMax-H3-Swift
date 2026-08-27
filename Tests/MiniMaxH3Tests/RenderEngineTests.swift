// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
import H3Foundation
@testable import MiniMaxH3

@Suite("enterprise render facade")
struct RenderEngineTests {
    /// The default is the cross-step cache, and it is an approximation.
    ///
    /// Reversed deliberately: the cache measures 1.79x for 16% less fine
    /// detail, which is the trade almost everyone wants on a twenty-minute
    /// render. What makes it safe is that the disclosure is independent of the
    /// default — `isApproximate` is asserted here alongside it, because a
    /// default that silently approximated would be the actual problem, not an
    /// approximate default that says so.
    @Test("cached output is the default, and it announces itself")
    func cachedDefault() {
        let request = RenderRequest(prompt: "test", videoOutput: URL(fileURLWithPath: "/tmp/out.mp4"))
        #expect(request.qualityProfile == .balanced)
        #expect(request.cacheThreshold == 0.10)
        #expect(request.qualityProfile.isApproximate)
    }

    /// And exactness is still reachable, unchanged.
    @Test("faithful remains available and exact")
    func faithfulStillExact() {
        let request = RenderRequest(prompt: "test", videoOutput: URL(fileURLWithPath: "/tmp/out.mp4"),
                                    qualityProfile: .faithful)
        #expect(request.cacheThreshold == 0)
        #expect(!request.qualityProfile.isApproximate)
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

    @Test("commit carries every field, not just the ones the initialiser takes")
    func commitCarriesEveryField() throws {
        // `commit` rebuilds the result to swap the staged URLs for published
        // ones, so anything the initialiser does not accept has to be copied
        // across by hand. Adding a field and forgetting that line compiles,
        // passes every other test, and silently zeroes the field — which is
        // exactly what happened to the benchmark trace: the record was assembled
        // from a published result whose trace had been dropped on the way, and
        // it would have written an empty `steps` array with no error anywhere.
        let root = FileManager.default.temporaryDirectory
            .appending(path: "h3-commit-test-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }

        let request = RenderRequest(prompt: "test", videoOutput: root.appending(path: "final.mp4"))
        let transaction = try OutputTransaction(jobID: UUID(), request: request)
        try Data("video".utf8).write(to: transaction.stagedRequest.videoOutput)
        try Data("audio".utf8).write(to: transaction.stagedRequest.audioOutput!)

        var staged = RenderResult(video: transaction.stagedRequest.videoOutput,
                                  audio: transaction.stagedRequest.audioOutput,
                                  frameCount: 124, width: 864, height: 480, seconds: 5,
                                  timings: .init(), cacheSummary: nil)
        staged.trace = SamplingTrace(steps: [], stepSeconds: [40, 20, 20, 40])
        staged.mlxPeakBytes = 53_000_000_000
        staged.mlxActiveBytesAtEnd = 1_234
        staged.attentionBackend = "sdpa"

        let published = try transaction.commit(staged)
        #expect(published.trace == staged.trace)
        #expect(published.mlxPeakBytes == 53_000_000_000)
        #expect(published.mlxActiveBytesAtEnd == 1_234)
        #expect(published.attentionBackend == "sdpa")
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

        // A failed render is still a render, and which arithmetic it was
        // running is exactly the sort of thing you want when working out why
        // it failed.
        #expect(receipt.compute?.ditDType == "bf16")
        #expect(receipt.compute?.aneRoutedProjections.isEmpty == true)
    }

    /// Two renders that differ only in arithmetic must not produce receipts
    /// that agree.
    ///
    /// This is the whole reason the field exists. A change of compute dtype, or
    /// of which projections ran on the Neural Engine, reselects the diffusion
    /// sample rather than degrading it — so the same seed and the same
    /// checkpoint give a visibly different video, and until schema 4 nothing in
    /// the receipt said which one you had.
    @Test("receipts distinguish renders that differ only in arithmetic")
    func arithmeticIsOnTheReceipt() throws {
        func receipt(dtype: String, routed: [String]) throws -> String {
            var r = RenderReceipt(
                jobID: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
                request: .init(mode: "t2va", width: 864, height: 480, seconds: 5,
                               steps: 20, seed: 42, promptCharacters: 10,
                               qualityProfile: "balanced", cacheThreshold: 0.25,
                               referenceImages: 0, referenceVideos: 0,
                               referenceAudio: 0, keyframeAnchors: 0),
                environment: .init(libraryVersion: "test", operatingSystem: "test",
                                   machine: "test", availableMemoryBytes: 0))
            r.compute = .init(ditDType: dtype, aneRoutedProjections: routed,
                              aneDeclinedProjections: [])
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            // `startedAt` is wall-clock, so compare everything else.
            var text = String(decoding: try encoder.encode(r), as: UTF8.self)
            if let range = text.range(of: "\"startedAt\":[^,]*,", options: .regularExpression) {
                text.removeSubrange(range)
            }
            return text
        }

        let production = try receipt(dtype: "bf16", routed: [])
        let fp16       = try receipt(dtype: "fp16", routed: [])
        let routed     = try receipt(dtype: "bf16", routed: ["fc1", "qkv"])

        #expect(production != fp16, "a dtype override must show on the receipt")
        #expect(production != routed, "engine routing must show on the receipt")
        #expect(fp16 != routed)

        // And it must survive the round trip a receipt exists to be read after.
        var original = RenderReceipt(
            jobID: UUID(),
            request: .init(mode: "t2va", width: 864, height: 480, seconds: 5, steps: 20,
                           seed: 42, promptCharacters: 10, qualityProfile: "balanced",
                           cacheThreshold: 0.25, referenceImages: 0, referenceVideos: 0,
                           referenceAudio: 0, keyframeAnchors: 0),
            environment: .init(libraryVersion: "test", operatingSystem: "test",
                               machine: "test", availableMemoryBytes: 0))
        original.compute = .init(ditDType: "bf16",
                                 aneRoutedProjections: ["fc1", "qkv"],
                                 aneDeclinedProjections: ["attn out"],
                                 aneCFGOverlap: true)

        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(RenderReceipt.self,
                                         from: try encoder.encode(original))
        #expect(decoded.schemaVersion == 5)
        #expect(decoded.compute?.ditDType == "bf16")
        #expect(decoded.compute?.aneRoutedProjections == ["fc1", "qkv"])
        #expect(decoded.compute?.aneDeclinedProjections == ["attn out"])
        #expect(decoded.compute?.aneCFGOverlap == true)
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
