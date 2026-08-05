# ADR 0002: Actor-owned, single-render runtime

- Status: accepted
- Date: 2026-08-05

## Context

H3 is batch-size-one and production renders hold tens of gigabytes for many
minutes. Two concurrent renderers caused a measured jetsam termination on a
275 GB machine. A static synchronous function cannot express ownership,
queueing, shutdown, or a stable event stream.

## Decision

`RenderEngine` is an actor and owns render admission. The initial queue policy
is `rejectWhenBusy`; it is explicit rather than an accidental side effect.
`RenderJob` exposes an `AsyncStream` of events, an async result, and cooperative
cancellation. Cancellation is observed at safe phase and sampler boundaries.

## Consequences

The runtime cannot silently oversubscribe one process. A future queued policy
can be added without changing the job model. AVFoundation callback APIs remain
contained inside their adapters and do not define the public concurrency model.
