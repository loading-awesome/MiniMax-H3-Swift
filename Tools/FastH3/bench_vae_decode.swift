// Where the 75 seconds of video decode actually go.
//
// A 480x864x124 render decodes as 6 temporal chunks x 15 spatial tiles = 90
// separate decoder passes, each one a 36-block, 2048-wide transformer over
// 1792 tokens with a batch of one. This measures whether that batch of one is
// the problem, by running the identical tile shape at several batch sizes.
import Foundation
import MLX
import H3Modules

func seconds(_ body: () -> MLXArray) -> Double {
    let began = Date()
    let out = body()
    eval(out)
    return Date().timeIntervalSince(began)
}

let path = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/Volumes/scratch_disk/models/MiniMax-H3/MiniMax-H3-video_vae_fp16.safetensors"

MLXRandom.seed(0)
let vae = try VideoVAE(url: URL(fileURLWithPath: path))
print("loaded \(path)")

// One tile as the pipeline builds it: 7 temporal tokens (5 + 2 overlap),
// 16x16 latent = a 256x256 pixel tile.
func tile(_ batch: Int) -> MLXArray {
    MLXRandom.normal([batch, 24, 7, 16, 16]).asType(.bfloat16)
}

// Warm up: first call pays for graph build and weight residency.
_ = seconds { vae.decodePixels(tile(1)) }

let quick = ProcessInfo.processInfo.environment["H3_BENCH_QUICK"] == "1"

// Same shape three times: with the table cached, any difference between the
// first and later calls is what the per-call preamble actually costs.
for r in 0 ..< (quick ? 0 : 3) {
    let z = tile(1)
    eval(z)
    print(String(format: "repeat %d  %.4f s", r, seconds { vae.decodePixels(z) }))
}

for batch in (quick ? [] : [1, 5, 15, 30, 90]) {
    let z = tile(batch)
    eval(z)
    let t = seconds { vae.decodePixels(z) }
    let per = t / Double(batch)
    print(String(format: "batch %3d  total %6.3f s  per tile %6.3f s  90 tiles would be %6.1f s",
                 batch, t, per, per * 90))
}

// Whole decodes at several frame sizes. 256-pixel tiles with a 64-pixel
// minimum overlap mean the tile count steps, not scales: 480 needs three
// tiles where 448 needs two, and 864 needs five where 832 needs four. A frame
// sitting just past both cliffs pays for almost twice the decode it needs.
for (w, h) in (quick ? [(480, 864), (448, 832)]
                             : [(480, 864), (448, 832), (448, 768), (512, 896)]) {
    let z = MLXRandom.normal([1, 24, 31, h / 16, w / 16]).asType(.bfloat16)
    eval(z)
    let t = seconds { vae.decode(z) }
    let (_, yl, _) = vae.splitTiles(inputLen: h)
    let (_, xl, _) = vae.splitTiles(inputLen: w)
    let out = vae.decode(z)
    eval(out)
    let mean = out.mean().item(Float.self)
    let rms = sqrt((out * out).mean()).item(Float.self)
    print(String(format: "%d x %d  %d x %d = %2d tiles  decode %6.2f s  mean %+.6f  rms %.6f",
                 w, h, xl.count, yl.count, xl.count * yl.count, t, mean, rms))
}

print("routed:   \(ANELinearBackend.routedProjections)")
print("declined: \(ANELinearBackend.declinedProjections)")
