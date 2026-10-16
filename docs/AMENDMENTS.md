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

## 2026-10-16 — Replicated hyperparameter selection and quality-first compute policy

### Status and reason

This amendment was committed before any FI-2010 hyperparameter search, test evaluation, or LOBSTER T1 run. It resolves the previously unspecified 12-trial grids and removes compute-driven fallbacks that could change the scientific scope. No validation or test result was used to choose these rules.

### Trial and horizon definition

- One trial is one candidate hyperparameter configuration trained on days 1–6 and ranked on day 7 by mean macro-F1.
- Seeds, epochs, checkpoints, horizons, final fits, and bootstrap replicates are not trials.
- Each of B1, B2, M1, and M2 receives exactly 12 candidate configurations independently at each primary horizon 10, 20, 50, and 100.
- B1 is deterministic. Each stochastic candidate for B2, M1, and M2 is replicated with dedicated selection seeds 5, 7, and 13. These replications remain one trial and are never evaluated on test.
- The selected configuration is retrained unchanged with final seeds 11, 23, 47, 91, and 137. Selection-seed fits are discarded. Configuration selection remains conditional on the three selection seeds and this limitation is reported.

### Fixed full grids

- B1: 12 base-10 log-spaced `C` values from `1e-4` through `1e2`, endpoints included.
- B2: leaves `{31, 63}` times learning rate `{0.03, 0.05, 0.1}` times minimum leaf size `{200, 400}`.
- M1: learning rate `{1e-4, 1e-3, 1e-2}` times Adam epsilon `{1e-8, 1}` times classifier dropout `{0, 0.2}`. Dropout is applied to the 64-dimensional final LSTM output immediately before the three-class linear classifier; it is not applied inside convolution, inception, or LSTM layers. Batch size is 64 and Adam weight decay is zero. The `lr=1e-2`, `epsilon=1`, `dropout=0` paper-optimizer anchor and the released-code learning rate are included.
- M2: learning rate `{3e-4, 1e-3, 3e-3}` times encoder layers `{2, 3}` times dropout `{0.1, 0.2}`. The original `lr=1e-3`, three-layer, dropout-0.1 configuration is included. Other M2 dimensions remain frozen.

### Selection and tie-breaking

- Rank stochastic candidates by mean unrounded validation macro-F1 over selection seeds 5, 7, and 13.
- Exact ties are broken by lower validation macro-F1 standard deviation, then fewer parameters, then lower learning rate, then canonical grid order.
- B1 uses the same rule without seed averaging or seed standard deviation.
- No day 8–10 observation, prediction, label distribution, or metric may be inspected before configuration and epoch selection are frozen. There is no primary refit on day 7.
- Training and final fits use the same maximum of 40 epochs and patience 5. There is no shortened tuning schedule.

### M1 reproduction anchor

At every horizon, run the M1 paper-optimizer anchor with all five final seeds even if it does not win validation. If the anchor wins, its selected final runs also serve as the anchor runs. Report it as the “DeepLOB paper-optimizer anchor under the study's leakage-controlled split,” not as an exact published-number reproduction. Report `selection_N_trials=12` for selected results and `reported_variant_N_trials=1` for the preregistered anchor.

### Compute and timing

- GPU timing is reported but cannot reduce horizons, trials, selection replications, final seeds, or epoch ceilings.
- Before search, run one unmeasured warmup epoch followed by three measured epochs for M1 and M2. Use the median measured epoch time and publish both the expected early-stopping projection and the no-early-stopping upper bound.
- The planning allowance is 200 GPU-hours and is not a hard stop. The earlier 60-hour cap, two-horizon M2 fallback, and proposed 30-epoch fallback are superseded.

### Reporting

Validation scores are explicitly labeled best-of-12 and potentially optimistic. Scientific claims use untouched test metrics only. Reports distinguish candidate configurations (`selection_N_trials`), seed replications, and preregistered reported variants. The search does not cover architecture width, and that limitation is stated.
