// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
@testable import H3Foundation

/// A profile and a checkpoint that disagree fail in a way nothing downstream
/// would notice: the shapes match, the render completes, and only the video is
/// worse. These pin both directions of that refusal.
@Suite("render profiles route to the right pipeline")
struct RenderProfileTests {

    @Test("turbo refuses a checkpoint with no ladder of its own")
    func turboRefusesBase() {
        #expect(throws: H3Error.self) {
            try H3RenderProfile.turbo.validate(declaresDistilledSteps: false,
                                               checkpoint: "MiniMax-H3-FL2VA_bf16.safetensors")
        }
    }

    @Test("standard refuses a distilled checkpoint")
    func standardRefusesDistill() {
        // The failure this catches is the quiet one: a distill overrides the
        // step count from its own metadata, so `--steps 20` yields four and the
        // receipt agrees with itself while disagreeing with the caller.
        #expect(throws: H3Error.self) {
            try H3RenderProfile.standard.validate(declaresDistilledSteps: true,
                                                  checkpoint: "FastH3-4step-Dense_bf16.safetensors")
        }
    }

    @Test("each profile accepts the checkpoint it is for")
    func matchesPass() throws {
        try H3RenderProfile.turbo.validate(declaresDistilledSteps: true, checkpoint: "d.safetensors")
        try H3RenderProfile.standard.validate(declaresDistilledSteps: false, checkpoint: "b.safetensors")
    }

    @Test("the refusal names the file and offers a way out")
    func refusalIsActionable() {
        // A refusal that does not say which file or what to do with it costs
        // more than the render it prevented.
        do {
            try H3RenderProfile.turbo.validate(declaresDistilledSteps: false,
                                               checkpoint: "base.safetensors")
            Issue.record("expected a refusal")
        } catch let error as H3Error {
            let text = error.errorDescription ?? ""
            #expect(text.contains("base.safetensors"))
            #expect(text.contains("standard"))
        } catch { Issue.record("wrong error type") }
    }

    @Test("the two profiles carry different step defaults")
    func stepDefaults() {
        #expect(H3RenderProfile.turbo.defaultSteps == 4)
        #expect(H3RenderProfile.standard.defaultSteps == 20)
        // The key prefix is what keeps a model out of the dtype namespace.
        #expect(H3RenderProfile.turbo.checkpointKey(precision: "bf16") == "turbo_bf16")
        #expect(H3RenderProfile.standard.checkpointKey(precision: "bf16") == "bf16")
    }
}
