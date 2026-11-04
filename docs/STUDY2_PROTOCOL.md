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
  mean over clusters with the A1 cluster bootstrap; out-of-sample R^2 (zero-forecast
  benchmark) of a no-intercept linear map from signal to bps; mark-out curve of
  signed move conditional on signal deciles, compared with the contemporaneous
  half-spread.
- Study 1 did not retain validation predictions, so the R^2 slope and decile edges
  are cross-fitted: leave-one-test-day-out on FI-2010 (3 folds) and
  leave-one-ticker-out on LOBSTER (5 folds). No row is scored by a map or edge fitted
  on its own fold. Metrics are computed per fitted seed and averaged over seeds.
  (Clarification committed 2026-11-03, before any A2 computation.)

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

## Part C — representation x model factorial (registered 2026-11-03, before any C fit)

### Status and timing

This section was committed before any Part C model was fitted, trained, or scored. The
FI-2010 test days (8-10) were seen in Study 1, but none of the Part C cells below has
ever produced a prediction on them. Part C is therefore a **pre-registered new-fit
analysis on previously seen test days**: stronger than the post-hoc Part A, weaker than
a study on untouched data. Part B (new data) remains the confirmatory test.

### Question

Study 1 gave LightGBM engineered order-flow features and the deep models the raw book,
so "LightGBM beats deep learning" confounds model class with input representation. Part C
crosses three representations with three model classes on FI-2010, all other Study 1
rules unchanged (split, purge, labels, horizons, seeds, metrics, train-only statistics).

### Representations

- **raw:** the 40 decimal-precision book columns. Neural models receive the 100x40
  window exactly as in Study 1. LightGBM receives the raw columns of the window at row
  lags {0, 1, 2, 5, 10, 20, 50} before the prediction row (7 x 40 = 280 columns, no
  arithmetic). A single snapshot would deny the tree model all history that the deep
  models see; the fixed lag set is the minimal history-bearing raw input and is not tuned.
- **eng:** the 29 locked causal features of `configs/protocol.yaml`, computed per
  (stock, day) segment before windowing. LightGBM receives the prediction-row vector
  (identical to Study 1). Neural models receive the 100x29 window, standardized per
  feature with mean and standard deviation fitted on training days 1-6 only, then clipped
  to [-5, 5].
- **both:** raw and eng together. LightGBM: 280 + 29 = 309 columns. Transformer: 69
  channels per timestep (40 raw, then 29 standardized eng). DeepLOB: the unchanged
  convolution/inception trunk on the raw 40 columns; the 29 standardized engineered
  features of the last 82 timesteps (the trunk shortens 100 steps to 82) are concatenated
  to the 192-channel trunk output before the LSTM (LSTM input 192 + 29 = 221).

### Models and cells

DeepLOB's convolutions are tied to the price/size layout of the raw book and are
undefined on engineered features. For `eng` the M1 row is therefore a **different model,
labeled "LSTM-64 head"**: the same LSTM(64) + dropout + linear classifier on the 100x29
window, with no convolutional trunk. It is reported separately and is never called DeepLOB.

| Model | raw | eng | both |
|---|---|---|---|
| B2 LightGBM | new | Study 1 | new |
| M1 DeepLOB / LSTM head | Study 1 | new (LSTM head) | new (fused DeepLOB) |
| M2 transformer | Study 1 | new | new |

Study 1 cells are reused as-is, not refitted. Six cells x four horizons are new.

### Hyperparameters and budgets

- B2 new cells: the Study 1 12-configuration grid, selection seeds 5, 7, 13, validation
  macro-F1 on day 7, identical exact-tie rule; the 12 selected configurations are
  hash-locked and committed before any day 8-10 data are constructed.
- Neural new cells: **no new search.** Each cell reuses the Study 1 frozen selected
  configuration of the same model and horizon (M1 for the LSTM head and fused DeepLOB; M2
  for both M2 cells). This is cheap and keeps every neural cell on a matched training
  budget, but those configurations were tuned on `raw`, so any `raw` advantage for neural
  models is biased in favor of `raw`. This limitation is reported next to every Part C
  neural result. An exploratory, clearly labeled validation-only learning-rate sensitivity
  (frozen value x {1/3, 1, 3}, selection seed 5, horizon 100) may be reported alongside.
- Final fits: seeds 11, 23, 47, 91, 137; max 40 epochs, patience 5, batch 64 for M1 cells
  and the Study 1 M2 settings for M2 cells; mixed precision as in Study 1.

### Primary hypotheses

Let gap_R(m) = macro-F1(B2, R) - macro-F1(m, R) for representation R, computed per
stock-day cluster and averaged over seeds.

- **H1 (interaction, M2):** gap_eng(M2) < gap_raw(M2): giving the transformer the same
  engineered features narrows LightGBM's advantage. Tested at each horizon.
- **H2 (interaction, M1):** gap_both(M1) < gap_raw(M1) at each horizon (DeepLOB cannot
  consume `eng` alone, so its interaction uses `both`).

Test statistic: per-cluster difference of gap differences; exact sign-flip test over the
15 stock-day clusters; Holm correction over the 8 contrasts (2 hypotheses x 4 horizons).
Two-sided p-values; the direction above is stated for interpretation only.

### Secondary analyses (labeled exploratory in every table)

- All within-model representation contrasts (both-raw, eng-raw, both-eng) and within-
  representation model contrasts (B2-M1, B2-M2, M1-M2): 72 comparisons, Holm over the
  72, cluster sign-flip and cluster bootstrap exactly as in A1.
- MCC, and the A2 return-unit metrics (cluster IC, cross-fitted OOS R^2, mark-outs), using
  the stored forward returns.
- Parameter counts and measured GPU-hours for every new cell.

### Execution order and scope

All four horizons are registered for every cell. Execution order is horizons 100 and 10
first (Phase A), then 50 and 20 (Phase B), solely to surface problems early. No cell,
seed, or horizon may be dropped silently: if compute does not allow completion, the
report lists exactly which cells are missing and their primary contrasts are withheld.
GPU time and projected hours are reported and cannot change scope.

### Reporting

Every table and figure produced from Part C is generated by script from run artifacts,
carries the status string `preregistered_new_fit_study2_C`, and states the neural
configuration-reuse limitation. Disagreement with the hypotheses is reported as found.

### Part C amendment 1 — neural cells are not gated on the B2 lock (2026-11-04)

Committed before any Part C final fit. No Part C model has produced a prediction on
days 8-10, and no validation result from any Part C B2 search has been inspected.

Measured LightGBM fits on the 280- and 309-column inputs take 22-166 s each, so the 8
registered B2 validation searches (36 fits each) need about 6 hours of CPU. The rule that
"selections are frozen before any day 8-10 data are constructed" exists so test outcomes
cannot influence selection. It binds only cells that *have* a selection step:

- **B2 final fits** still require the committed, hash-verified B2 manifest. Unchanged.
- **Neural final fits** use the Study 1 frozen configurations and involve no Part C
  selection, so there is nothing a test outcome could influence. They are therefore
  gated only on the Study 1 lock and may run while B2 searches are in progress.
- B2 selection remains a deterministic validation-only rule with no discretion; the
  selection code, grids, seeds and tie-break are unchanged from Study 1.

This changes execution order only. No cell, horizon, seed, hypothesis, metric or
correction is altered.
