# Research roadmap — Study 2 and beyond

Study 1 (the preregistered FI-2010 benchmark and LOBSTER zero-shot transfer) is
complete and frozen. Its protocol, amendments, and generated results are not edited
by anything below. Every item here is either (a) an explicitly labeled *post-hoc
exploratory* re-analysis of Study 1 predictions, or (b) a new study governed by
[`STUDY2_PROTOCOL.md`](STUDY2_PROTOCOL.md), committed before its data are touched.

## Open questions Study 1 leaves

1. **Inference granularity.** The test set has 15 stock-day clusters (5 stocks x 3
   days). The registered moving-block intervals resample ~10^5 overlapping windows
   within each stock-day and therefore do not reflect between-day variation. With
   ~50 paired comparisons there is also no family-wise error control.
2. **Model versus representation.** B2 receives engineered order-flow features; M1
   and M2 receive the raw 40-column book. The B2 > M1/M2 result cannot separate
   architecture from input representation.
3. **Target relevance.** The FI-2010 three-class label at horizon 10 is matched to
   0.80 macro-F1 by sign persistence of the last representation move, so the label
   is dominated by autocorrelation induced by 10-event compression. Practitioners
   evaluate forecasts in return units (information coefficient, out-of-sample R^2,
   mark-outs relative to the half-spread).
4. **Transfer representation.** Decimal-precision scaling is market-specific; price
   distances in ticks and sizes normalized by depth are closer to market-invariant.
   The fixed-label transfer collapse may be partly a representation artifact.
5. **Scale and non-stationarity.** Ten days from 2010 cannot support claims about
   regime dependence, drift, or data-scaling behavior.

## Work plan

Each step lands as tested library code first, then a runner executed on the GPU host,
then generated report assets. No step edits Study 1 artifacts.

| Step | Question | Deliverable | Needs data run | Status |
|---|---|---|---|---|
| 1 | Q1 | Stock-day cluster inference: exact sign-flip randomization test and cluster bootstrap on paired differences, Holm family-wise adjustment | yes (reads existing predictions) | **done** (2026-11-03) |
| 2 | Q3 | Return-unit evaluation: Spearman IC by cluster, out-of-sample R^2, signed mark-out curves in bps versus half-spread | yes | **done** (2026-11-03) |
| 3 | Q4 | Market-invariant book representation (tick-distance prices, depth-share sizes) and a transfer re-run with it | yes | planned |
| 4 | Q2 | Representation x model factorial: {raw book, engineered, both} x {LightGBM, DeepLOB, transformer}; protocol in `STUDY2_PROTOCOL.md` Part C | yes (GPU) | **in progress** (protocol frozen 2026-11-03; B2 validation searches and neural final fits running on the GPU host since 2026-11-04) |
| 5 | Q5 | Multi-week crypto L2 pipeline (public exchange depth snapshots + trades), walk-forward splits with many day clusters | new data | planned |
| 6 | Q5 | Data-scaling curves (train days 1..N) for GBDT vs deep models; drift diagnostics (PSI, rolling IC) | new data | planned |
| 7 | Q2/Q4 | Self-supervised pretraining (masked book-event modeling) with linear-probe and fine-tune transfer | new data, GPU | planned |
| 8 | — | Event-driven execution simulator with queue position and latency, exposed as a gym-style environment for cost-aware decision evaluation | new data | planned |

Engineering items run alongside: CI (ruff + pytest), a pinned lockfile, and a
throughput benchmark (bf16 / `torch.compile`) for the neural trainers.

## What will not change

- No alpha, Sharpe, or profitability claim is made for FI-2010 or LOBSTER.
- Post-hoc analyses of Study 1 are labeled exploratory in every generated table.
- New-data studies freeze splits, targets, metrics, and comparison families before
  the data are downloaded.
