// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX

/// Timing support for opt-in production-shape benchmarks.
///
/// Long GPU measurements are sensitive to ordering, thermals and unrelated
/// system work. Alternating ABBA/BAAB prevents either arm from permanently
/// receiving the earlier or later position; medians reject isolated outliers.
enum BenchmarkSupport {
    static func interleaved(
        rounds: Int = 5, first: () -> MLXArray, second: () -> MLXArray
    ) -> (first: Double, second: Double) {
        MLX.eval(first(), second())
        var firstSamples: [Double] = []
        var secondSamples: [Double] = []

        func sample(_ body: () -> MLXArray, into samples: inout [Double]) {
            let t0 = Date()
            MLX.eval(body())
            samples.append(Date().timeIntervalSince(t0))
        }
        for round in 0 ..< rounds {
            if round.isMultiple(of: 2) {
                sample(first, into: &firstSamples)
                sample(second, into: &secondSamples)
                sample(second, into: &secondSamples)
                sample(first, into: &firstSamples)
            } else {
                sample(second, into: &secondSamples)
                sample(first, into: &firstSamples)
                sample(first, into: &firstSamples)
                sample(second, into: &secondSamples)
            }
        }
        return (median(firstSamples), median(secondSamples))
    }

    private static func median(_ values: [Double]) -> Double {
        precondition(!values.isEmpty)
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }
}
