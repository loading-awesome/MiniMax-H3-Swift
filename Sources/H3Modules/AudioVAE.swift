// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXNN
import H3Foundation

package func snake(_ x: MLXArray, alpha: MLXArray, beta: MLXArray) -> MLXArray {
    let t = sin(alpha * x)
    return (t * t) / (beta + 1e-9) + x
}

package struct VaeConv1d {
    package let weight: MLXArray
    package let bias: MLXArray?
    package let stride: Int
    package let padding: Int
    package let dilation: Int
    package let groups: Int
    
    package init(weight: MLXArray, bias: MLXArray? = nil, stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1) {
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    }
    
    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias {
            out = out + bias.reshaped([1, 1, bias.dim(0)])
        }
        return out
    }
}

package struct VaeConvTransposed1d {
    package let weight: MLXArray
    package let bias: MLXArray?
    package let stride: Int
    package let padding: Int
    package let dilation: Int
    package let groups: Int
    
    package init(weight: MLXArray, bias: MLXArray? = nil, stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1) {
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    }
    
    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = convTransposed1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias {
            out = out + bias.reshaped([1, 1, bias.dim(0)])
        }
        return out
    }
}

package class SnakeBeta: Module {
    package let alpha: MLXArray
    package let beta: MLXArray

    package init(alpha: MLXArray, beta: MLXArray) {
        self.alpha = alpha
        self.beta = beta
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha).reshaped([1, 1, alpha.dim(0)])
        let b = exp(beta).reshaped([1, 1, beta.dim(0)])
        return snake(x, alpha: a, beta: b)
    }
}

package class UpSample1d: Module {
    package let ratio: Int
    package let stride: Int
    package let pad: Int
    package let padLeft: Int
    package let padRight: Int
    package let filter: MLXArray

    package init(ratio: Int = 2, kernelSize: Int = 12, filter: MLXArray) {
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernelSize / ratio - 1
        self.padLeft = self.pad * ratio + (kernelSize - ratio) / 2
        self.padRight = self.pad * ratio + (kernelSize - ratio + 1) / 2
        self.filter = filter
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padWidths: [IntOrPair] = [
            0,
            [pad, pad],
            0
        ]
        let paddedX = padded(x, widths: padWidths, mode: .edge)
        
        let C = x.dim(2)
        let weight = broadcast(filter, to: [C, filter.dim(1), filter.dim(2)]).transposed(0, 2, 1)
        
        var out = convTransposed1d(paddedX, weight.asType(x.dtype), stride: stride, padding: 0, groups: C)
        out = out * Float(ratio)
        
        let s = padLeft
        let e = out.dim(1) - padRight
        let indices: [any MLXArrayIndex] = [0 ..< out.dim(0), s ..< e, 0 ..< out.dim(2)]
        return out[indices]
    }
}

package class LowPassFilter1d: Module {
    package let padLeft: Int
    package let padRight: Int
    package let stride: Int
    package let filter: MLXArray

    package init(cutoff: Float = 0.5, halfWidth: Float = 0.6, stride: Int = 1, kernelSize: Int = 12, filter: MLXArray) {
        self.padLeft = kernelSize / 2 - (kernelSize % 2 == 0 ? 1 : 0)
        self.padRight = kernelSize / 2
        self.stride = stride
        self.filter = filter
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padWidths: [IntOrPair] = [
            0,
            [padLeft, padRight],
            0
        ]
        let paddedX = padded(x, widths: padWidths, mode: .edge)
        
        let C = x.dim(2)
        let weight = broadcast(filter, to: [C, filter.dim(1), filter.dim(2)]).transposed(0, 2, 1)
        
        return conv1d(paddedX, weight.asType(x.dtype), stride: stride, padding: 0, groups: C)
    }
}

package class DownSample1d: Module {
    package let lowpass: LowPassFilter1d

    package init(ratio: Int = 2, kernelSize: Int = 12, filter: MLXArray) {
        self.lowpass = LowPassFilter1d(stride: ratio, kernelSize: kernelSize, filter: filter)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        return lowpass(x)
    }
}

package class Activation1d: Module {
    package let act: SnakeBeta
    package let upsample: UpSample1d
    package let downsample: DownSample1d

    package init(activation: SnakeBeta, upFilter: MLXArray, downFilter: MLXArray) {
        self.act = activation
        self.upsample = UpSample1d(ratio: 2, kernelSize: 12, filter: upFilter)
        self.downsample = DownSample1d(ratio: 2, kernelSize: 12, filter: downFilter)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = upsample(x)
        h = act(h)
        return downsample(h)
    }
}

package class AMPBlock1: Module {
    package let convs1: [VaeConv1d]
    package let convs2: [VaeConv1d]
    package let activations: [Activation1d]

    package init(channels: Int, kernelSize: Int, dilation: [Int],
                convs1Weights: [MLXArray], convs1Biases: [MLXArray],
                convs2Weights: [MLXArray], convs2Biases: [MLXArray],
                alphas: [MLXArray], betas: [MLXArray],
                upFilters: [MLXArray], downFilters: [MLXArray]) {
        
        var c1: [VaeConv1d] = []
        var c2: [VaeConv1d] = []
        var acts: [Activation1d] = []
        
        let numLayers = dilation.count
        for i in 0 ..< numLayers {
            let d = dilation[i]
            let p = (kernelSize * d - d) / 2
            
            let conv1 = VaeConv1d(weight: convs1Weights[i].transposed(0, 2, 1), bias: convs1Biases[i], stride: 1, padding: p, dilation: d)
            c1.append(conv1)
            
            let conv2 = VaeConv1d(weight: convs2Weights[i].transposed(0, 2, 1), bias: convs2Biases[i], stride: 1, padding: (kernelSize - 1) / 2, dilation: 1)
            c2.append(conv2)
            
            let act1 = SnakeBeta(alpha: alphas[2 * i], beta: betas[2 * i])
            let blockAct1 = Activation1d(activation: act1, upFilter: upFilters[2 * i], downFilter: downFilters[2 * i])
            acts.append(blockAct1)
            
            let act2 = SnakeBeta(alpha: alphas[2 * i + 1], beta: betas[2 * i + 1])
            let blockAct2 = Activation1d(activation: act2, upFilter: upFilters[2 * i + 1], downFilter: downFilters[2 * i + 1])
            acts.append(blockAct2)
        }
        
        self.convs1 = c1
        self.convs2 = c2
        self.activations = acts
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for i in 0 ..< convs1.count {
            let a1 = activations[2 * i]
            let a2 = activations[2 * i + 1]
            let c1 = convs1[i]
            let c2 = convs2[i]
            
            var xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        }
        return x
    }
}

package class BigVGAN: Module {
    package let convPre: VaeConv1d
    package let ups: [VaeConvTransposed1d]
    package let resblocks: [AMPBlock1]
    package let activationPost: Activation1d
    package let convPost: VaeConv1d
    package let numKernels: Int
    package let numUpsamples: Int

    package init(numMels: Int, upsampleInitialChannel: Int,
                upsampleRates: [Int] = [5, 5, 2, 2, 2, 2, 2],
                upsampleKernelSizes: [Int] = [9, 9, 4, 4, 4, 4, 4],
                resblockKernelSizes: [Int] = [3, 7, 11],
                resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                weights: [String: MLXArray]) throws {
        
        func get(_ name: String) throws -> MLXArray {
            guard let a = weights[name] else {
                throw H3Weights.Error.missing(name)
            }
            return a
        }

        self.numKernels = resblockKernelSizes.count
        self.numUpsamples = upsampleRates.count
        
        let preW = try get("decoder.conv_pre.weight")
        let preB = try get("decoder.conv_pre.bias")
        self.convPre = VaeConv1d(weight: preW.transposed(0, 2, 1), bias: preB, stride: 1, padding: 3)
        
        var upModules: [VaeConvTransposed1d] = []
        for i in 0 ..< numUpsamples {
            let u = upsampleRates[i]
            let k = upsampleKernelSizes[i]
            let p = (k - u) / 2
            
            let upW = try get("decoder.ups.\(i).0.weight")
            let upB = try get("decoder.ups.\(i).0.bias")
            
            let up = VaeConvTransposed1d(weight: upW.transposed(1, 2, 0), bias: upB, stride: u, padding: p)
            upModules.append(up)
        }
        self.ups = upModules
        
        var blocks: [AMPBlock1] = []
        for i in 0 ..< numUpsamples {
            let ch = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))
            for j in 0 ..< numKernels {
                let k = resblockKernelSizes[j]
                let d = resblockDilationSizes[j]
                let idx = i * numKernels + j
                let p = "decoder.resblocks.\(idx)."
                
                var convs1W: [MLXArray] = []
                var convs1B: [MLXArray] = []
                var convs2W: [MLXArray] = []
                var convs2B: [MLXArray] = []
                var alphas: [MLXArray] = []
                var betas: [MLXArray] = []
                var upFilters: [MLXArray] = []
                var downFilters: [MLXArray] = []
                
                for l in 0 ..< d.count {
                    convs1W.append(try get(p + "convs1.\(l).weight"))
                    convs1B.append(try get(p + "convs1.\(l).bias"))
                    convs2W.append(try get(p + "convs2.\(l).weight"))
                    convs2B.append(try get(p + "convs2.\(l).bias"))
                }
                for l in 0 ..< 2 * d.count {
                    alphas.append(try get(p + "activations.\(l).act.alpha"))
                    betas.append(try get(p + "activations.\(l).act.beta"))
                    upFilters.append(try get(p + "activations.\(l).upsample.filter"))
                    downFilters.append(try get(p + "activations.\(l).downsample.lowpass.filter"))
                }
                
                let block = AMPBlock1(
                    channels: ch, kernelSize: k, dilation: d,
                    convs1Weights: convs1W, convs1Biases: convs1B,
                    convs2Weights: convs2W, convs2Biases: convs2B,
                    alphas: alphas, betas: betas,
                    upFilters: upFilters, downFilters: downFilters
                )
                blocks.append(block)
            }
        }
        self.resblocks = blocks
        
        let postAlpha = try get("decoder.activation_post.act.alpha")
        let postBeta = try get("decoder.activation_post.act.beta")
        let postUpF = try get("decoder.activation_post.upsample.filter")
        let postDownF = try get("decoder.activation_post.downsample.lowpass.filter")
        self.activationPost = Activation1d(
            activation: SnakeBeta(alpha: postAlpha, beta: postBeta),
            upFilter: postUpF, downFilter: postDownF
        )
        
        let postW = try get("decoder.conv_post.weight")
        self.convPost = VaeConv1d(weight: postW.transposed(0, 2, 1), bias: nil, stride: 1, padding: 3)
        
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convPre(x)
        for i in 0 ..< numUpsamples {
            x = ups[i](x)
            var xs: MLXArray? = nil
            for j in 0 ..< numKernels {
                let block = resblocks[i * numKernels + j]
                let bx = block(x)
                if let s = xs {
                    xs = s + bx
                } else {
                    xs = bx
                }
            }
            x = xs! / Float(numKernels)
        }
        x = activationPost(x)
        let post = convPost(x)
        return minimum(maximum(post, -1.0), 1.0)
    }
}

package final class MiniMaxH3AudioVAE {
    package let url: URL
    package let latentsMean: MLXArray
    package let latentsStd: MLXArray
    package let decInProj: VaeConv1d
    package let decoder: BigVGAN

    package init(url: URL) throws {
        self.url = url
        let w = try MLX.loadArrays(url: url)
        
        func get(_ name: String) throws -> MLXArray {
            guard let a = w[name] else {
                throw H3Weights.Error.missing(name)
            }
            return a
        }

        self.latentsMean = try get("latents_mean")
        self.latentsStd = try get("latents_std")
        
        let inW = try get("dec_in_proj.weight")
        let inB = try get("dec_in_proj.bias")
        self.decInProj = VaeConv1d(weight: inW.transposed(0, 2, 1), bias: inB, stride: 1, padding: 0)
        
        self.decoder = try BigVGAN(numMels: 2048, upsampleInitialChannel: 1024, weights: w)
    }

    package func decode(_ z: MLXArray) -> MLXArray {
        let b = z.dim(0)
        let c = z.dim(1)
        let s = z.dim(2)
        let t = z.dim(3)
        
        let zPerm = z.transposed(0, 2, 1, 3)
        var flatZ = zPerm.reshaped([b * s, c, t])
        
        let mean = latentsMean.reshaped([1, c, 1])
        let std = latentsStd.reshaped([1, c, 1])
        flatZ = flatZ * std + mean
        
        let flatZT = flatZ.transposed(0, 2, 1)
        
        var x = decInProj(flatZT)
        x = decoder(x)
        
        let L = x.dim(1)
        var out = x.reshaped([b, s, L])
        
        let N = Float(s * L)
        let outMean = out.mean(axes: [1, 2], keepDims: true)
        let biasedVar = (out - outMean).square().mean(axes: [1, 2], keepDims: true)
        let unbiasedVar = biasedVar * (N / (N - 1.0))
        var outStd = sqrt(unbiasedVar) * 5.0
        
        outStd = maximum(outStd, 1.0)
        out = out / outStd
        return out
    }
}
