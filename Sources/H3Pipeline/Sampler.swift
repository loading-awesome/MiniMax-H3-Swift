// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import H3Foundation

final class Sampler {
    private var oldSigmaDown: Float? = nil
    private var oldDenoised: MLXArray? = nil

    init() {}

    /// Advance one step of the res_multistep ODE integration.
    func step(
        x: MLXArray,
        denoised: MLXArray,
        sigma: Float,
        sigmaNext: Float,
        prevSigma: Float?
    ) -> MLXArray {
        let sigmaDown = sigmaNext
        
        let tFn = { (s: Float) -> Float in -log(s) }
        let phi1Fn = { (t: Float) -> Float in
            if abs(t) < 1e-6 {
                return 1.0 + t / 2.0
            }
            return (exp(t) - 1.0) / t
        }
        let phi2Fn = { (t: Float) -> Float in
            if abs(t) < 1e-6 {
                return 0.5 + t / 6.0
            }
            return (phi1Fn(t) - 1.0) / t
        }

        var nextX = x
        if sigmaDown == 0 || oldDenoised == nil {
            // Euler step
            let d = (x - denoised) / sigma
            let dt = sigmaDown - sigma
            nextX = x + d * dt
        } else {
            // Second order multistep method
            let t = tFn(sigma)
            let tOld = tFn(oldSigmaDown!)
            let tNext = tFn(sigmaDown)
            let tPrev = tFn(prevSigma!)
            
            let h = tNext - t
            let c2 = (tPrev - tOld) / h
            
            let phi1Val = phi1Fn(-h)
            let phi2Val = phi2Fn(-h)
            
            var b1 = phi1Val - phi2Val / c2
            var b2 = phi2Val / c2
            
            if b1.isNaN { b1 = 0.0 }
            if b2.isNaN { b2 = 0.0 }
            
            let decay = sigmaDown / sigma
            nextX = MLXArray(decay) * x + MLXArray(h) * (MLXArray(b1) * denoised + MLXArray(b2) * oldDenoised!)
        }
        
        self.oldDenoised = denoised
        self.oldSigmaDown = sigmaDown
        
        return nextX
    }
}
