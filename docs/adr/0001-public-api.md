# ADR 0001: One supported public module

- Status: accepted
- Date: 2026-08-05

## Context

The Swift package has useful internal targets, but the umbrella used
`@_exported` imports and made their implementation surfaces look supported.
That turns changes to checkpoint parsing, hardware planning, or the MLX
pipeline into apparent source-breaking releases.

## Decision

`MiniMaxH3` is the only supported import. It exposes render-domain values and
an actor-owned engine. Internal targets remain separate for dependency and test
boundaries, but are not compatibility promises. MLX types never cross the
facade. The CLI consumes the same facade for rendering.

## Consequences

Public API compatibility can be measured at one module. Internal targets may
evolve without SemVer impact. Advanced backend experimentation remains an
internal or explicitly versioned SPI concern.
