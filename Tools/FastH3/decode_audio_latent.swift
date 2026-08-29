// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich
//
// Decode a captured audio latent through THIS port's audio VAE.
//
// The point is to separate two explanations for 4-step audio that is "a bit
// off": our sampling loop reaching a worse latent than the reference, or the
// checkpoint's four-step audio simply sounding like this. Feeding the
// reference pipeline's own final audio latent through our decoder answers it
// in one step -- if it is clean, the fault is upstream of the decoder and ours
// to fix; if it is equally off, the ladder is the ceiling and no sampler
// change recovers it.
//
//   swift run h3-decode-audio --latent their_audio.npy --out reference.wav
import Foundation
import MLX
import H3Modules
import H3Foundation

// npy is the interchange the capture writes for a single tensor: numpy header,
// then raw little-endian float32. Only the shape line needs parsing.
func loadNPY(_ url: URL) throws -> MLXArray {
    let data = try Data(contentsOf: url)
    guard data.count > 10, data[0] == 0x93 else { throw Err.bad("not a .npy") }
    let headerLen = Int(data[8]) | (Int(data[9]) << 8)
    let header = String(decoding: data[10 ..< (10 + headerLen)], as: UTF8.self)
    guard let shapeRange = header.range(of: #"\((\d+,\s*)*\d+,?\)"#, options: .regularExpression)
    else { throw Err.bad("no shape in npy header: \(header)") }
    let dims = header[shapeRange]
        .trimmingCharacters(in: CharacterSet(charactersIn: "()"))
        .split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let body = data.dropFirst(10 + headerLen)
    let floats: [Float] = body.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: Float32.self))
    }
    let expected = dims.reduce(1, *)
    guard floats.count >= expected else {
        throw Err.bad("npy short: \(floats.count) floats for shape \(dims)")
    }
    return MLXArray(Array(floats.prefix(expected)), dims)
}

enum Err: Error { case bad(String) }

func writeWAV(_ samples: [Float], channels: Int, rate: Int, to url: URL) throws {
    let frames = samples.count / channels
    var d = Data()
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + frames * channels * 2))
    d.append("WAVEfmt ".data(using: .ascii)!); u32(16); u16(1); u16(UInt16(channels))
    u32(UInt32(rate)); u32(UInt32(rate * channels * 2)); u16(UInt16(channels * 2)); u16(16)
    d.append("data".data(using: .ascii)!); u32(UInt32(frames * channels * 2))
    for s in samples { u16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
    try d.write(to: url)
}

let args = CommandLine.arguments
func flag(_ name: String) -> String? {
    guard let i = args.firstIndex(of: "--\(name)"), i + 1 < args.count else { return nil }
    return args[i + 1]
}
guard let latentPath = flag("latent"), let outPath = flag("out") else {
    print("usage: h3-decode-audio --latent <audio.npy> --out <out.wav> [--vae <path>]")
    exit(2)
}
let vaePath = flag("vae")
    ?? "/Volumes/scratch_disk/models/MiniMax-H3/MiniMax-H3-audio_vae_fp32.safetensors"

let z = try loadNPY(URL(fileURLWithPath: latentPath))
print("latent \(z.shape)")
let vae = try MiniMaxH3AudioVAE(url: URL(fileURLWithPath: vaePath))
let wave = vae.decode(z)
eval(wave)
print("waveform \(wave.shape)")

// [1, channels, L] -> interleaved
let ch = wave.dim(1), n = wave.dim(2)
let flat = wave.reshaped([ch, n]).asType(.float32).asArray(Float.self)
var interleaved = [Float](repeating: 0, count: ch * n)
for c in 0 ..< ch { for i in 0 ..< n { interleaved[i * ch + c] = flat[c * n + i] } }
try writeWAV(interleaved, channels: ch, rate: H3Audio.sampleRate,
             to: URL(fileURLWithPath: outPath))
print("wrote \(outPath)")
