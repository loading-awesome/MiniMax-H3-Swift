import Testing
import Foundation
@testable import H3Hardware

/// The planner is the component most likely to be quietly wrong, because it is
/// the one nobody checks until a render dies twenty minutes in. These run in
/// milliseconds with no GPU and no checkpoint.
@Suite("memory planner")
struct MemoryPlanTests {

    static let gb: UInt64 = 1_000_000_000

    @Test("a smaller checkpoint does not buy a bigger render")
    func activationsDominateAtLength() {
        // The single most important property. Activations scale with packed
        // sequence length and attention is quadratic in it, so swapping bf16
        // for a smaller variant must NOT be enough to make a long sequence fit
        // — a planner that only trades weights will happily produce a
        // configuration that still gets killed.
        let short = MemoryPlan.samplingActivationBytes(packedTokens: 15_750)
        let long = MemoryPlan.samplingActivationBytes(packedTokens: 31_500)
        // Doubling the length more than doubles the activations — superlinear,
        // because of the quadratic attention term. Measured 2.44x at these
        // lengths: the linear term still dominates at 15k-31k tokens, so the
        // growth is real but not yet dramatic. Asserting a higher multiple here
        // would be asserting a coefficient, not a property.
        #expect(Double(long) > Double(short) * 2.0)

        // And the weight saving between the largest and smallest non-approximate
        // variants is smaller than the activation growth over that same range.
        let weightSaving = MemoryPlan.Precision.bf16.residentDITBytes
                         - MemoryPlan.Precision.prunedInt8.residentDITBytes
        #expect(long - short > weightSaving)
    }

    @Test("int8 does not claim a memory saving it cannot deliver")
    func int8IsHonest() {
        // int8 checkpoints are stored I8 with per-channel scales and are
        // dequantised at load, so they save disk and nothing else until a
        // resident quantised matmul lands. The planner must not promise
        // otherwise; a plan that says 34 GB and uses 66 is worse than no plan.
        #expect(MemoryPlan.Precision.int8.isResidentQuantised == false)
        #expect(MemoryPlan.Precision.int8.residentDITBytes
                == MemoryPlan.Precision.bf16.residentDITBytes)
        #expect(MemoryPlan.Precision.int8.ditBytes < MemoryPlan.Precision.bf16.ditBytes)
    }

    @Test("approximate weights are never selected without permission")
    func approximateIsGated() {
        // 40 GB: too small for bf16, big enough for the pruned variants. The
        // planner must return nothing rather than silently substituting a model
        // whose AdaLN is a rank-64 curve.
        let denied = MemoryPlan.best(packedTokens: 15_750,
                                     availableBytes: 40 * Self.gb,
                                     allowApproximate: false)
        #expect(denied == nil)

        // 100 GB: still short of bf16 with its margin, comfortably enough for
        // the pruned variants at 73.8 GB peak.
        let allowed = MemoryPlan.best(packedTokens: 15_750,
                                      availableBytes: 100 * Self.gb,
                                      allowApproximate: true)
        #expect(allowed != nil)
    }

    @Test("the peak is the largest phase, not the sum")
    func phasesArePeakNotSum() {
        // Encoding conditions before loading the DiT is a memory contract. If
        // the planner ever sums phases instead of taking the maximum, it will
        // refuse configurations that run perfectly well.
        let plan = MemoryPlan.plan(precision: .bf16, packedTokens: 15_750,
                                   availableBytes: 275 * Self.gb)
        let sum = plan.phaseBytes.values.reduce(0, +)
        #expect(plan.peakBytes < sum)
        #expect(plan.peakBytes == plan.phaseBytes[.sampling])
    }

    @Test("small Macs are refused, large ones are not")
    func realWorldConfigurations() {
        // The honest table from the README, asserted.
        for gb in [8, 16, 24, 32] as [UInt64] {
            #expect(MemoryPlan.best(packedTokens: 15_750, availableBytes: gb * Self.gb,
                                    allowApproximate: true) == nil,
                    "\(gb) GB must be refused")
        }
        #expect(MemoryPlan.best(packedTokens: 15_750, availableBytes: 192 * Self.gb,
                                allowApproximate: false)?.precision == .bf16)
    }

    @Test("available memory is measured, not assumed from hw.memsize")
    func availableIsLessThanTotal() {
        let m = Machine.detect()
        let available = Machine.availableBytes()
        #expect(m.memoryBytes > 0)
        // A machine with anything running has less available than installed.
        // Planning against hw.memsize is how a render gets killed next to a
        // second process.
        #expect(available <= m.memoryBytes)
    }
}
