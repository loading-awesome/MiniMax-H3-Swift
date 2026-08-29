// Whether a quantised matmul beats dense bf16 at the DiT's production shapes.
//
// int8 checkpoints in this repo dequantise at load — they save disk and
// nothing else (MemoryPlan.isResidentQuantised). Making quantisation worth
// anything at render time means keeping weights quantised and computing with
// MLX's quantizedMatmul, and whether that is faster is a property of the
// shapes: quantised kernels win when the GEMM is bound by streaming weights,
// and the DiT's sequence lengths are large enough that these GEMMs may be
// compute-bound instead, where dequant-on-the-fly is pure overhead.
import Foundation
import MLX
import MLXRandom

func best(_ n: Int, _ body: () -> MLXArray) -> Double {
    var t = Double.infinity
    for _ in 0 ..< n {
        let began = Date()
        eval(body())
        t = min(t, Date().timeIntervalSince(began))
    }
    return t
}

// (s, k, n): the four DiT projections at the 448x832x124 sequence length the
// calibration trace reports, plus the text-encoder MLP shape at a typical
// prompt length.
let cases: [(String, Int, Int, Int)] = [
    ("dit qkv     ", 13938, 5376, 16128),
    ("dit attn out", 13938, 7168, 5376),
    ("dit fc1     ", 13938, 5376, 28672),
    ("dit fc2     ", 13938, 14336, 5376),
    ("text mlp    ", 300, 5120, 25600),
]

print("shape          bf16      q8       q4     q8/bf16  q4/bf16")
for (name, s, k, n) in cases {
    let x = MLXRandom.normal([s, k]).asType(.bfloat16)
    let w = MLXRandom.normal([n, k]).asType(.bfloat16)
    eval(x, w)
    let dense = best(3) { matmul(x, w.T) }

    var line = String(format: "%@ %7.1f ms", name, dense * 1e3)
    var ratios = ""
    for bits in [8, 4] {
        let (wq, scales, biases) = MLX.quantized(w, groupSize: 64, bits: bits)
        eval(wq, scales, biases)
        let q = best(3) {
            quantizedMatmul(x, wq, scales: scales, biases: biases,
                            transpose: true, groupSize: 64, bits: bits)
        }
        line += String(format: " %7.1f", q * 1e3)
        ratios += String(format: "   %.2fx", dense / q)
    }
    print(line + ratios)
}
