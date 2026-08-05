# ADR 0003: Faithful output is the default

- Status: accepted
- Date: 2026-08-05

## Context

Cross-step reuse at threshold 0.10 is nearly twice as fast but loses a measured
16 percent of high-frequency detail. Approximate AdaLN weights also change the
model by construction. A default that changes output undermines parity claims.

## Decision

The default quality profile disables cross-step reuse and approximate weights.
Acceleration is selected explicitly and its measured trade-off is included in
events and the render receipt. Unknown profiles are refused, not downgraded.

## Consequences

Default results remain inside the verified implementation contract. Faster
runs are available, but cannot be mistaken for the faithful configuration.
