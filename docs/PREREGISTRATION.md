# Preregistration — frozen 2026-10-13

## Scope

This is a benchmark, leakage-audit, and transfer study. It is not a trading strategy, does not claim alpha, and does not report Sharpe or an equity curve. FI-2010 is the primary benchmark and free LOBSTER samples are the transfer target. Crypto L2 and a state-space model are optional stretch work.

## Labels and horizons

The primary FI-2010 ground truth is the dataset's shipped label: the percentage change between the current mid-price and the single mid-price at the future horizon, thresholded at inclusive +/-0.002. The available label columns represent 10, 20, 30, 50, and 100 book events. Primary reporting uses 10, 20, 50, and 100; 30 is intentionally excluded to match the original project scope.

The proposed forward-smoothed mean label is retained only as an explicitly named ablation and cannot replace the primary labels. Agreement between labels reconstructed from decimal-precision inputs and shipped labels is measured and reported, not assumed.

Every 100-event model window is contained within one `(stock, day)` segment. Labels and windows cannot cross a segment or split boundary.

For LOBSTER, both fixed alpha 0.002 and a distribution-matched alpha are reported. The latter is selected once on the first 20% of the target day by deterministic grid search to minimize total-variation distance from FI-2010 training class proportions. The remaining 80% is evaluation data.

## Temporal protocol

Days 1–6 train, day 7 validates, and days 8–10 test. A 200-event purge exceeds the 100-event input plus the largest primary horizon. Hyperparameters and epoch selection use validation macro-F1. The primary fit is never refit on validation. A separately dated amendment variant may refit days 1–7 for the validation-selected fixed epoch count.

## Model ladder and selection

Models run in order: B0 majority/persistence, B1 multinomial L2 logistic regression, B2 LightGBM, M1 DeepLOB, M2 transformer. M3 is optional. B1/B2 use the locked causal feature family in the machine-readable config, with the shipped 144-feature representation as an ablation.

B1, B2, M1, and M2 each receive exactly 12 hyperparameter trials. They use unweighted cross-entropy or its multinomial equivalent. B0 and B1 are deterministic. Stochastic models use seeds 11, 23, 47, 91, and 137.

M1 reproduces the convolution/inception/LSTM(64) architecture. The primary implementation follows the official PyTorch code's batch size 64. Both the paper-era Adam `(lr=0.01, epsilon=1)` and released-code Adam `(lr=0.0001)` settings are recorded as a protocol difference; selection is confined to the 12-trial budget.

M2 uses `d_model=64`, 4 heads, 3 pre-LN layers, FFN 128, dropout 0.1, learned positions, and last-token pooling. AdamW uses lr 0.001, weight decay 0.01, 5% warmup, cosine decay, and gradient clipping 1.0. Both neural models stop by 40 epochs with patience 5.

## Evaluation

Report class precision/recall/F1, macro-F1, MCC, reliability curves, and ECE. Confidence intervals use a 2,000-replicate 95% moving-block bootstrap over contiguous test windows within stock-day segments with minimum block length 100, pooled across seeds; seed SD is separate. Paired claims reuse identical sampled blocks on per-window model differences. Every reported result carries `N_trials` and a 95% CI. Parameter count and measured GPU-hours accompany each model.

## Leakage controls

Train-only normalization is asserted in code. The deliberately leaky full-dataset normalization variant runs once and is labeled. Features use data no later than prediction time. No resampling touches validation or test. Per-split class balance is always generated. The combined cross-stock test portfolio is the primary unit of truth; stock-level metrics are diagnostics.

## Economic-significance boundary

The analysis compares correctly predicted conditional mid-price movement with contemporaneous spread plus a conservative declared fee. It contains no execution strategy, equity curve, return series, or Sharpe statistic.

## Compute

GPU work runs through `ssh norgate` on its RTX 3070 with mixed precision. The cap is 60 measured GPU-hours. If measurement projects an overrun, M2 may be limited to horizons 10 and 100 only through a dated amendment; no silent scope change is allowed.

## Sources checked before freeze

- Ntakaris et al., *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods*, arXiv:1705.03233.
- Zhang, Zohren, and Roberts, *DeepLOB*, arXiv:1808.03668.
- Authors' released TensorFlow and PyTorch notebooks, repository `zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books`.

