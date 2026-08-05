import Foundation
import ArgumentParser
import MiniMaxH3

extension H3RecipeID: ExpressibleByArgument {}
extension RenderRequest.AspectRatio: ExpressibleByArgument {}
extension RenderRequest.ResolutionTier: ExpressibleByArgument {}

/// A render, from the command line.
///
/// Every option here is argument plumbing over `RenderRequest`, and every rule
/// about what a render may be lives in the library. That is not tidiness: in the
/// experimental tree the rules lived in the command's `validate()`, which meant
/// they could only be reached by spawning a process, and could not be tested at
/// all.
struct RenderCommand: ParsableCommand {
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

    @Option(help: "cross-step cache threshold; 0 is faithful, 0.10 is the measured knee")
    var cacheThreshold: Double = 0.10

    @Option(help: "most consecutive cached steps before a full one is forced")
    var cacheMaxSkips: Int = 3

    @Flag(help: "probe the whole packed sequence instead of per stream")
    var cacheWholeSequenceProbe = false

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

    func run() throws {
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
            cacheThreshold: cacheThreshold, cacheMaxSkips: cacheMaxSkips,
            cacheWholeSequenceProbe: cacheWholeSequenceProbe,
            conditioningNoise: conditioningNoise.map(URL.init(fileURLWithPath:)),
            attentionBackend: attention ?? cfg.attention.backend,
            allowSuboptimal: allowSuboptimal)
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
        print("render")
        print("  mode        \(request.mode.rawValue) — \(request.modeDescription)")
        print("  shape       \(w)x\(h), \(seconds)s at \(H3Video.fps) fps, \(steps) steps")
        print("  dit         \(resolution.dit.url.lastPathComponent)")
        print("  guidance    \(cfgScale > 1 ? String(format: "%.2f", cfgScale) : "off")")

        var lastPhase: RenderProgress.Phase?
        let result = try H3Pipeline.render(
            request: request,
            checkpoints: H3Pipeline.Checkpoints(
                dit: resolution.dit.url, textEncoder: resolution.textEncoder.url,
                tokenizer: tokenizer, videoVAE: resolution.videoVAE.url,
                audioVAE: resolution.audioVAE.url),
            progress: { p in
                if p.phase != lastPhase {
                    print("\(p.phase.rawValue): \(p.detail)")
                    lastPhase = p.phase
                } else if p.phase == .sampling, p.total > 0 {
                    print("  \(p.detail)")
                }
            },
            log: { print($0) })

        // Where the wall clock actually went. Sampling and both decodes are GPU,
        // the pixel pack is GPU, the mux is VideoToolbox plus memcpy. Guessing
        // at this split is what once made a hung writer look like a slow CPU.
        let t = result.timings
        print(String(format: """
            timings
              text conditioning %7.1fs
              condition encode  %7.1fs
              sampling          %7.1fs   %d step(s)%@
              audio decode      %7.1fs
              video decode      %7.1fs
              pixel pack        %7.1fs
              mux + encode      %7.1fs
            """, t.textConditioning, t.conditionEncoding, t.sampling, steps,
            cfgScale > 1.0 ? ", 2 forwards each" : "",
            t.audioDecode, t.videoDecode, t.pixelPack, t.mux))

        print(String(format: "wrote %@ — %d frames, %.2fs",
                     result.video.path, result.frameCount, result.seconds))
        if let audio = result.audio { print("wrote \(audio.path)") }
    }
}
