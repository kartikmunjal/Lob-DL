# Study 2 protocol — drafted 2026-11-03

## Status

Study 1 test results were inspected before this document was written. Therefore every
Study 2 analysis that reuses FI-2010 or LOBSTER Study 1 predictions is **post-hoc
exploratory**: it may qualify or contextualize Study 1 claims, but it cannot create a
new confirmatory claim on those datasets. Confirmatory status is reserved for new
data whose splits and analysis rules are committed here before download.

## Part A — post-hoc re-analysis of Study 1 predictions

### A1. Stock-day cluster inference

- Unit: one `(stock, day)` test cluster, pooled over final seeds by averaging the
  cluster-level metric across seeds. FI-2010 has 15 test clusters.
- Statistic per cluster: paired difference in macro-F1 (system A minus system B)
  computed within the cluster. Macro-F1 is used because it is the Study 1 primary
  metric; accuracy differences are reported alongside.
- Tests: exact two-sided sign-flip randomization test over all `2^C` sign
  assignments of the cluster differences (C = 15 gives 32,768 assignments; no Monte
  Carlo error), and a percentile cluster bootstrap (resampling clusters with
  replacement, 10,000 replicates, seed 20261103) for the mean cluster difference.
- Family: all paired comparisons published by Study 1 at the primary horizons. The
  Holm step-down procedure controls family-wise error at 0.05 over that whole family.
- Reported next to the registered moving-block interval, never in place of it.

### A2. Return-unit evaluation

- Signal: `P(up) - P(down)` from the frozen predicted probabilities.
- Target: forward-mean relative mid-price move at the registered horizon, in basis
  points, as already computed by `fi_economic_arrays`.
- Metrics: Spearman information coefficient within each cluster, summarized by the
  mean over clusters with the A1 cluster bootstrap; out-of-sample R^2 of a
  no-intercept linear map fitted on validation predictions only; mark-out curve of
  signed move conditional on signal deciles, compared with the contemporaneous
  half-spread.

### A3. Market-invariant transfer representation

- Prices become signed distances from the mid in units of the instrument's minimum
  observed positive top-of-book spread on the training (or calibration) prefix; sizes
  become the share of displayed depth on the same side within the top 10 levels.
- The FI-2010 train/validation/test split and LOBSTER calibration/evaluation split are
  unchanged from Study 1. Models are refit with the Study 1 selected configuration;
  no new hyperparameter search.

## Part B — new data (confirmatory)

Part B is frozen in a dated amendment to this document before any download. That
amendment must fix: venues and instruments, date range, walk-forward split rule,
purge length, targets and horizons, the comparison family, and the cluster unit.
