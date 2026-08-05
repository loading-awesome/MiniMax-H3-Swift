// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import ArgumentParser
import MiniMaxH3
import H3Foundation
import H3Catalog

struct ConfigCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "config",
        abstract: "Create or check the configuration file.",
        subcommands: [Initialise.self, Validate.self]
    )

    struct Initialise: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "init",
            abstract: "Write a template configuration.")

        @Option(help: "where to write it")
        var output: String?

        @Flag(help: "overwrite an existing file")
        var force = false

        func run() throws {
            let url = output.map(URL.init(fileURLWithPath:)) ?? H3Configuration.defaultURL
            if FileManager.default.fileExists(atPath: url.path) && !force {
                throw H3Error.unreadable(path: url.path,
                                         reason: "already exists; pass --force to overwrite")
            }
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try H3Configuration.template.encoded().write(to: url)
            print("wrote \(url.path)")
            print("Edit `checkpoints.root`, then run `h3 doctor`.")
        }
    }

    struct Validate: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Resolve every configured path and identify what is there.")

        @Option(help: "configuration file")
        var config: String?

        func run() throws {
            let (cfg, url) = try H3Configuration.load(
                from: config.map(URL.init(fileURLWithPath:)))
            print(url?.path ?? "built-in defaults")
            var bad = 0
            for (role, path, result) in Catalog(config: cfg).inventory() {
                switch result {
                case .success(let id): print("  ok      \(role)  \(id.summary)")
                case .failure(let e):
                    bad += 1
                    print("  BAD     \(role)  \(path)")
                    print("          \(e)")
                }
            }
            if bad > 0 { throw ExitCode(1) }
        }
    }
}
