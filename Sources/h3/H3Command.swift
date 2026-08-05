// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import MiniMaxH3

/// A thin CLI over the public API. Every subcommand here is a few lines of
/// argument plumbing over a call into `MiniMaxH3` — if a command needs logic of
/// its own, that logic belongs in a library target where it can be tested
/// without spawning a process.
///
/// **`AsyncParsableCommand`, and not by preference.** `render` drives an actor,
/// so it is async, and ArgumentParser only awaits an async subcommand when the
/// root is async too. Declared synchronous it printed a warning and then never
/// called `run()` at all: `h3 render` parsed its arguments, rendered nothing,
/// and exited 0.
@main
struct H3: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "h3",
        abstract: "Generate video and audio together with MiniMax-H3, on Apple silicon.",
        discussion: """
            Three commands, in the order you will need them.

              h3 doctor          what this Mac can run, which checkpoints it found,
                                 and what it would choose. Takes about a second.
              h3 config init     write a config file to fill in with your model paths
              h3 render          make a video

            A first render, once `h3 doctor` reports no problems:

              h3 render --prompt "a red kite over a beach at sunset" --out kite.mp4

            That is 5 seconds at 1344x768, and on an M3 Ultra it takes about
            fifteen minutes. `h3 render --help` lists every option, and
            `h3 render --dry-run` prints the plan and the time estimate without
            loading a checkpoint.
            """,
        version: MiniMaxH3.version,
        subcommands: [Doctor.self, ConfigCommand.self, RenderCommand.self]
    )
}
