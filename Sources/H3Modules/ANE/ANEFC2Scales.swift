// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Per-block operand scales that bring whole-`fc2` under the engine's 2^15
/// partial-sum cliff.
///
/// `fc2` was refused on saturation, and correctly: its worst partial-sum bound
/// is 4,573,078 at block 45, which at the shipping 1/16 scale is 9x past the
/// cliff, and the failure mode is silent zeros rather than an error. Splitting
/// the contraction does not rescue it — at block 45 it buys 1.05x, because the
/// magnitude is concentrated rather than spread across `k`.
///
/// A scale does, and `fc2` is the projection best able to afford one. The two
/// hazards pull in opposite directions: saturation wants a small scale, the
/// fp16 denormal floor at 6.1e-5 wants a large one. `fc2`'s products are
/// enormous — about 319 unscaled at block 45 — so it starts far from the floor.
///
/// The bounds vary 790x across blocks (5,781 at block 09 against 4,573,078 at
/// block 45), so this is per block. That is not only about saturation: because
/// each entry is the largest power of two holding `bound * scale` inside a
/// fixed window, every block's typical product lands in the same narrow range
/// instead of wherever its own magnitude falls. One global scale would be harsh
/// on the 14 blocks needing none, and would push them toward the floor.
///
/// The entries below carry the probe's own factor-of-two margin.
/// `H3_ANE_FC2_HALVINGS` adds more, and it is a real trade rather than free
/// safety: measured over every block and step, two extra halvings put the worst
/// block at 1/2048 and rel-RMS 1.81e-3, because underflow error then dominates.
/// Fewer halvings read better and sit closer to the cliff.
///
/// Generated from a faithful bound render — `H3_ANE_BOUND=... h3 render` with
/// the engine off, 50 blocks, every row, 20 steps. Do not hand-edit.
///
/// **These bounds are worst-case over one prompt at one shape.** They are
/// evidence, not proof, for another prompt, which is why `H3_ANE_FC2` is
/// opt-in and `H3_ANE_FC2_VERIFY` exists to re-audit.
enum ANEFC2Scales {
    /// Extra halvings beyond the probe's factor-of-two margin.
    static let halvings: Int =
        ProcessInfo.processInfo.environment["H3_ANE_FC2_HALVINGS"]
            .flatMap(Int.init).map { max(0, min(8, $0)) } ?? 2

    static let base: [Float] = [
        1.0 / 4,          // block 00  bound       43,676
        1.0 / 4,          // block 01  bound       61,935
        1.0 / 1,          // block 02  bound        6,288
        1.0 / 1,          // block 03  bound       10,591
        1.0 / 2,          // block 04  bound       17,444
        1.0 / 2,          // block 05  bound       20,945
        1.0 / 1,          // block 06  bound       13,882
        1.0 / 2,          // block 07  bound       21,890
        1.0 / 1,          // block 08  bound        8,083
        1.0 / 1,          // block 09  bound        5,781
        1.0 / 1,          // block 10  bound        6,367
        1.0 / 2,          // block 11  bound       32,558
        1.0 / 1,          // block 12  bound        8,758
        1.0 / 8,          // block 13  bound       94,909
        1.0 / 1,          // block 14  bound        5,849
        1.0 / 1,          // block 15  bound        8,196
        1.0 / 1,          // block 16  bound        8,371
        1.0 / 2,          // block 17  bound       17,311
        1.0 / 8,          // block 18  bound       92,776
        1.0 / 1,          // block 19  bound       10,452
        1.0 / 1,          // block 20  bound        9,565
        1.0 / 1,          // block 21  bound       11,553
        1.0 / 1,          // block 22  bound       13,269
        1.0 / 2,          // block 23  bound       17,654
        1.0 / 2,          // block 24  bound       19,637
        1.0 / 4,          // block 25  bound       35,342
        1.0 / 4,          // block 26  bound       33,399
        1.0 / 4,          // block 27  bound       47,590
        1.0 / 8,          // block 28  bound       66,144
        1.0 / 8,          // block 29  bound       79,816
        1.0 / 8,          // block 30  bound      115,822
        1.0 / 8,          // block 31  bound       85,534
        1.0 / 8,          // block 32  bound       89,622
        1.0 / 8,          // block 33  bound       94,926
        1.0 / 8,          // block 34  bound      109,506
        1.0 / 16,         // block 35  bound      149,350
        1.0 / 32,         // block 36  bound      454,961
        1.0 / 8,          // block 37  bound      117,157
        1.0 / 16,         // block 38  bound      131,542
        1.0 / 256,        // block 39  bound    3,033,599
        1.0 / 16,         // block 40  bound      228,453
        1.0 / 16,         // block 41  bound      220,318
        1.0 / 16,         // block 42  bound      189,070
        1.0 / 32,         // block 43  bound      514,636
        1.0 / 128,        // block 44  bound    1,053,233
        1.0 / 512,        // block 45  bound    4,573,078
        1.0 / 16,         // block 46  bound      195,601
        1.0 / 32,         // block 47  bound      346,103
        1.0 / 64,         // block 48  bound      827,710
        1.0 / 256,        // block 49  bound    2,576,862
    ]

    /// Nil for a block with no calibrated entry — the caller keeps `fc2` on the
    /// GPU rather than guessing, because guessing here is a silent zero.
    static func scale(forBlock block: Int) -> Float? {
        guard block >= 0, block < base.count else { return nil }
        var value = base[block]
        for _ in 0 ..< halvings where value > 1.0 / 65536 { value /= 2 }
        return value
    }
}
