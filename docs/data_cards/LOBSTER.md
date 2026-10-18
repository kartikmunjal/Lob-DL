# LOBSTER sample data card

## Intended use

Free LOBSTER samples are the zero-shot transfer target. FI-2010-trained models are evaluated without target-label training. The experiment measures cross-market and cross-era degradation; it does not estimate an executable strategy.

## Provenance and coverage

The study uses official one-day sample message and vendor order-book files for AAPL, AMZN, GOOG, INTC, and MSFT on 2012-06-21, at top-10 depth. Data are not redistributed.

## Primary representation

Vendor book files are primary. The state after every tenth message is retained, aligned from the first row, so one target representation corresponds to ten underlying messages. A 100-representation input therefore spans 1,000 messages, matching FI-2010's documented granularity. Per-column decimal-precision powers are fitted on the first 20% of each ticker and frozen for the remaining 80%; this is scale-only and uses no centering or variance matching.

## Labels

The primary transfer label uses fixed `alpha=0.002`. A preregistered secondary threshold is chosen separately by ticker and horizon on the first 20% to match FI-2010 training class proportions by total-variation distance, then frozen. Both label variants evaluate the same predictions on the remaining 80%.

## Reconstruction validation

Messages are replayed from the first supplied aggregate snapshot with no reseeding. Unattributed visible reductions are applied at the message price level, negative sizes are clamped and logged, absent levels are logged and skipped, and hidden executions do not change displayed depth. Every replayed state is compared with the vendor top 10. Vendor books feed primary T1; reconstructed sensitivity windows require exact equality in every top-10 cell across all 100 input rows.

## Limitations

- The supplied stream begins after pre-open orders already exist, so initial order IDs are unobserved and pure replay error accumulates.
- A single US trading day per ticker cannot characterize general US-market transfer.
- FI-2010 and LOBSTER differ in venue, tick size, currency, era, sampling, and market structure.
- Fixed `alpha=0.002` is economically very large for these US samples and is expected to expose class collapse; the matched-threshold analysis is explicitly secondary.
