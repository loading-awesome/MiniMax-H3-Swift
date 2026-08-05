// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

// The public surface. Everything a caller needs, and nothing that would make
// an internal change a breaking one.
//
// MLX types deliberately do not appear here. A caller passes URLs, structs and
// enums and gets back files, progress and typed errors — which is what lets the
// attention backend, the quantisation scheme and the compute backend change
// without a major version.
// The rule is enforced rather than merely stated: callers use RenderEngine,
// RenderJob, requests, events, results and receipts. The model layers and
// PipelineRuntime remain implementation-module details.
import Foundation

public enum MiniMaxH3 {
    /// Semantic version of the public API.
    public static let version = "0.1.0-dev"
}
