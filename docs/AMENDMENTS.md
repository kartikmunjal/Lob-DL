# Protocol Amendments

This file appends dated amendments to the original preregistration. The original text in `docs/PREREGISTRATION.md` remains unchanged.

## 2026-10-14 — LOBSTER replay and validation protocol

### Status and timing

This amendment was committed before any T1 run. At the time of amendment, no LOBSTER transfer model, prediction, label distribution, reconstruction match rate, or T1 evaluation result had been run or inspected. Inspection was limited to the first five AAPL messages and first two vendor snapshots to establish file semantics and the presence of pre-open liquidity.

### Reason and affected datasets

The LOBSTER message files begin at 09:30 while the first vendor snapshot contains orders entered before the supplied message stream. Therefore a book cannot be reconstructed from an empty state. This amendment affects only (1) LOBSTER order-book reconstruction validation and (2) the LOBSTER T1 transfer dataset. It does not change the FI-2010 protocol.

### Primary reconstruction protocol

- Initialize the aggregate book once from the first supplied vendor snapshot, which is the state after message 1.
- Replay messages 2 through the end in order, with no periodic or mismatch-triggered re-seeding. This is the primary pure-replay result.
- Any re-seeded reconstruction is auxiliary, must be labeled as such, and cannot feed the primary T1 analysis.
- For a cancel, delete, or visible execution whose order ID cannot be attributed because the initial snapshot is aggregated, apply the stated size change at the message price level. Clamp a negative result to zero and log every clamp. If the level is absent, log it separately and skip the change.
- Hidden executions (type 5) do not change the displayed book. Cross trades and trading-halt indicators likewise do not alter displayed depth.
- For every ticker, report the count and fraction of attributed events, unattributed applied events, zero clamps, absent-level skips, hidden executions, cross trades, and halt indicators. Fractions use all replayed messages (messages 2 through end) as the denominator.

### Fixed mismatch metrics

After every replayed message, compare reconstructed and vendor prices and sizes on both sides. Report per ticker:

- exact-match rate for level 1, all top 5 levels, and all top 10 levels;
- first mismatch timestamp and one-based message index;
- the same exact-match rates in fixed 30-minute bins since 09:30;
- total absolute size error separately for bid and ask at each level 1 through 10, summed over replayed messages.

The initialization row is reported separately and is excluded from replay match-rate denominators. No post-hoc match-rate cutoff or “good enough” classification is permitted; the full preregistered table is reported.

### T1 input choice

Vendor-provided LOBSTER book files are the primary T1 input. Reconstructed books are a sensitivity analysis only. A reconstructed 100-event window is eligible only when every bid/ask price and size cell at every top-10 level exactly matches the vendor book for every row in that entire window. No looser cutoff may be selected after observing results.

### Required wording

Project descriptions use: “order-book reconstruction from the first sample snapshot, validated against vendor book files.” Resume wording uses: “replayed LOBSTER messages with invariant checks and quantified the mismatch from unobserved pre-open liquidity.”
