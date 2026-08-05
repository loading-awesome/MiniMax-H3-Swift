# ADR 0006: Atomic outputs and durable receipts

- Status: accepted
- Date: 2026-08-05

## Context

A render may take thirty minutes. A mux failure or termination must not publish
a corrupt final file or erase the information needed to diagnose the run.
Plain callback strings are not a durable operational record.

## Decision

Each job writes into a same-volume temporary workspace. Completed artifacts are
promoted atomically. The engine emits typed events and writes a JSON receipt
containing job identity, resolved quality, hardware, checkpoint fingerprints,
timings, outputs, warnings, and terminal status. Failure receipts are retained;
temporary media is removed unless it is a deliberate recovery artifact.

## Consequences

Final paths contain complete artifacts only. Operators can diagnose a terminal
job without reproducing it, and logs can be integrated without parsing CLI
prose.
