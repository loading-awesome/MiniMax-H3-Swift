import ArgumentParser
import MiniMaxH3

/// A thin CLI over the public API. Every subcommand here is a few lines of
/// argument plumbing over a call into `MiniMaxH3` — if a command needs logic of
/// its own, that logic belongs in a library target where it can be tested
/// without spawning a process.
@main
struct H3: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "h3",
        abstract: "MiniMax-H3 on Apple Silicon.",
        version: MiniMaxH3.version,
        subcommands: [Doctor.self, ConfigCommand.self]
    )
}
