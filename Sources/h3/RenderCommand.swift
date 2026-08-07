// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import ArgumentParser
import MiniMaxH3
import H3Foundation
import H3Catalog
import H3Recipes

extension H3RecipeID: ExpressibleByArgument {}
extension RenderRequest.AspectRatio: ExpressibleByArgument {}
extension RenderRequest.ResolutionTier: ExpressibleByArgument {}
extension RenderRequest.QualityProfile: ExpressibleByArgument {}

/// A render, from the command line.
///
/// Every option here is argument plumbing over `RenderRequest`, and every rule
/// about what a render may be lives in the library. That is not tidiness: in the
/// experimental tree the rules lived in the command's `validate()`, which meant
/// they could only be reached by spawning a process, and could not be tested at
/// all.
struct RenderCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "render",
        abstract: "Render video and audio together from a prompt."
    )

    @Option(help: "what to render, up to 7,000 characters")
    var prompt: String

    @Option(help: "where to write the mp4")
    var out: String

    @Option(help: "also write a side-car wav")
    var outAudio: String?

    @Option(help: "negative prompt; applied through guidance, so it needs --cfg-scale above 1")
    var negativePrompt: String?

    @Option(help: "duration in seconds, 4 to 15")
    var seconds: Int = 5

    @Option(help: "sampling steps")
    var steps: Int = 20

    @Option(help: "random seed")
    var seed: Int = 0

    @Option(help: "explicit width, a multiple of 32; overrides --recipe and --aspect-ratio")
    var width: Int?

    @Option(help: "explicit height, a multiple of 32")
    var height: Int?

    @Option(help: "a registered recipe, e.g. h3_768p_16_9")
    var recipe: H3RecipeID?

    @Option(help: "16:9, 9:16, 21:9, 4:3, 3:4 or 1:1")
    var aspectRatio: RenderRequest.AspectRatio = .r16x9

    @Option(help: "768p, or 2k which is an upscale target rather than a render size")
    var resolution: RenderRequest.ResolutionTier = .p768

    @Option(help: "first-frame anchor")
    var firstFrame: String?

    @Option(help: "last-frame anchor")
    var lastFrame: String?

    @Option(parsing: .upToNextOption, help: "reference images, up to 9")
    var referenceImages: [String] = []

    @Option(parsing: .upToNextOption, help: "reference videos, up to 3")
    var referenceVideos: [String] = []

    @Option(parsing: .upToNextOption,
            help: "soundtracks index-matched to --reference-videos; '' for a silent one")
    var referenceVideoAudios: [String] = []

    @Option(parsing: .upToNextOption, help: "standalone reference audio, up to 3")
    var referenceAudio: [String] = []

    @Option(help: "guidance scale; 1.0 turns guidance off")
    var cfgScale: Double = 1.0

    @Option(help: "quality profile: faithful or balanced")
    var quality: RenderRequest.QualityProfile = .balanced




    @Option(help: "recorded conditioning noise, for a render inside the parity contract")
    var conditioningNoise: String?

    @Option(help: "attention backend: auto, or a registered identifier")
    var attention: String?

    @Option(help: "checkpoint precision key from the configuration")
    var precision: String = "bf16"

    @Option(help: "configuration file (default: ~/.config/minimax-h3/config.json)")
    var config: String?

    @Flag(help: "render even if the policy rejects this configuration, logging every reason")
    var allowSuboptimal = false

    @Flag(help: "replace an existing final output")
    var overwrite = false

    @Flag(help: "print the plan and the time estimate, then stop without rendering")
    var dryRun = false

    @Flag(name: .shortAndLong, help: "show memory figures and backend selection while rendering")
    var verbose = false

    func run() async throws {
        // Line-buffer stdout. Redirected to a file it is block-buffered by
        // default, and a render that is SIGKILLed — which is how an
        // out-of-memory render ends, with no signal handler and no unwinding —
        // takes the whole buffer with it. A 26-minute failure once left a
        // 272-byte log holding one stderr warning and nothing else.
        setvbuf(stdout, nil, _IOLBF, 0)

        let (cfg, _) = try H3Configuration.load(from: config.map(URL.init(fileURLWithPath:)))

        let request = RenderRequest(
            prompt: prompt,
            videoOutput: URL(fileURLWithPath: out),
            audioOutput: outAudio.map(URL.init(fileURLWithPath:)),
            negativePrompt: negativePrompt,
            seconds: seconds, steps: steps, seed: UInt64(max(0, seed)),
            width: width, height: height, recipe: recipe,
            aspectRatio: aspectRatio, resolution: resolution,
            firstFrame: firstFrame.map(URL.init(fileURLWithPath:)),
            lastFrame: lastFrame.map(URL.init(fileURLWithPath:)),
            referenceImages: referenceImages.map(URL.init(fileURLWithPath:)),
            referenceVideos: referenceVideos.map(URL.init(fileURLWithPath:)),
            // An empty string is "silent", so a later video can still have one.
            referenceVideoSoundtracks: referenceVideoAudios.map {
                $0.isEmpty ? nil : URL(fileURLWithPath: $0)
            },
            referenceAudio: referenceAudio.map(URL.init(fileURLWithPath:)),
            cfgScale: cfgScale,
            qualityProfile: quality,
            conditioningNoise: conditioningNoise.map(URL.init(fileURLWithPath:)),
            attentionBackend: attention ?? cfg.attention.backend,
            allowSuboptimal: allowSuboptimal,
            overwriteOutput: overwrite)
        try request.validate()

        // Which partition is needed follows from the mode, so the catalog is
        // asked for that mode's checkpoint and refuses a mismatch rather than
        // rendering with weights that were never trained for it.
        let resolution = try Catalog(config: cfg).resolve(mode: request.mode,
                                                          precision: precision)
        guard let tokenizer = resolution.tokenizerDirectory else {
            throw H3Error.checkpointMissing(
                role: "tokenizer",
                path: "no tokenizer directory configured (vocab.json, merges.txt, "
                    + "tokenizer_config.json)")
        }

        let (w, h) = try request.dimensions()
        let geometry = LatentGeometry(width: w, height: h,
                                      length: request.seconds * H3Video.fps)
        let packedTokens = geometry.videoTokens + geometry.audioTokens
        RenderReporter.plan(
            request: request, width: w, height: h, frameCount: geometry.frameCount,
            fps: H3Video.fps, packedTokens: packedTokens, checkpoint: resolution.dit.url.lastPathComponent,
            estimateSeconds: H3RenderPolicy.estimatedSeconds(tokens: packedTokens,
                                                             steps: request.steps))

        // Everything above is header-only — safetensors headers, arithmetic, no
        // tensor bodies — so `--dry-run` can answer "what would this cost" in
        // about a second rather than after a checkpoint load.
        if dryRun {
            print("dry run: nothing was loaded and nothing was written.")
            return
        }

        guard let modelPrecision = ModelSet.Precision(rawValue: precision) else {
            throw H3Error.invalidRequest(
                rule: "unknown precision", detail: precision,
                remedy: "use bf16, int8, pruned_bf16, or pruned_int8.")
        }
        let engine = RenderEngine(
            models: ModelSet(
                dit: ModelFile(url: resolution.dit.url),
                textEncoder: ModelFile(url: resolution.textEncoder.url),
                tokenizer: tokenizer,
                videoVAE: ModelFile(url: resolution.videoVAE.url),
                audioVAE: ModelFile(url: resolution.audioVAE.url),
                precision: modelPrecision),
            options: .init(
                allowApproximateWeights: cfg.policy.allowApproximateWeights,
                memoryMarginFraction: cfg.policy.memoryMarginFraction))
        let job = try await engine.start(request)
        let eventPrinter = Task {
            var reporter = RenderReporter()
            for await event in job.events {
                switch event {
                case .progress(let p):
                    reporter.observe(p)
                case .warning(let message):
                    reporter.finishLine()
                    FileHandle.standardError.write(Data(("warning: " + message + "\n").utf8))
                case .diagnostic(let message):
                    // Memory figures and backend selection: useful when a render
                    // goes wrong, noise when it does not.
                    if verbose {
                        reporter.finishLine()
                        print("  " + message)
                    }
                case .failed(let code, let message):
                    reporter.finishLine()
                    FileHandle.standardError.write(Data(("\n\(code): \(message)\n").utf8))
                case .admitted, .completed:
                    break
                }
            }
            return reporter
        }
        let result = try await job.value()
        var reporter = await eventPrinter.value
        reporter.finished(result)

    }
}
