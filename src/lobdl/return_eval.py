"""Return-unit evaluation of frozen classifier outputs (Study 2, A2).

Classification scores say little about whether a forecast is large enough to
matter after costs. These helpers turn the three-class probabilities into a
continuous directional signal and evaluate it against realized forward moves in
basis points:

* Spearman information coefficient (IC) per cluster;
* out-of-sample R^2 against the zero forecast, with the signal-to-bps scale fitted
  by leave-one-fold-out cross-fitting so no row is scored by a map fitted on it;
* mark-out curves: mean signed forward move by cross-fitted signal decile,
  compared with the contemporaneous half-spread a liquidity taker would pay.

Nothing here simulates fills, positions, or P&L.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

CLASS_DOWN, CLASS_UP = 0, 2


def directional_signal(probabilities: np.ndarray) -> np.ndarray:
    """``P(up) - P(down)`` in [-1, 1] using the Study 1 class order (down, flat, up)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (rows, 3)")
    return probabilities[:, CLASS_UP] - probabilities[:, CLASS_DOWN]


def spearman_ic(signal: np.ndarray, target: np.ndarray) -> float:
    """Tie-aware Spearman correlation; NaN when either side is constant."""
    signal, target = np.asarray(signal, float), np.asarray(target, float)
    if signal.shape != target.shape or signal.ndim != 1:
        raise ValueError("signal and target must be aligned 1-D arrays")
    if len(signal) < 3:
        return float("nan")
    a, b = rankdata(signal), rankdata(target)
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def cluster_ic(
    signal: np.ndarray, target: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Spearman IC within each cluster; returns ``(cluster_ids, ics)``."""
    clusters = np.asarray(clusters).reshape(len(signal), -1)
    ids, index = np.unique(clusters, axis=0, return_inverse=True)
    index = index.reshape(-1)
    ics = np.array(
        [spearman_ic(signal[index == k], target[index == k]) for k in range(len(ids))]
    )
    return ids, ics


def _fold_ids(folds: np.ndarray) -> np.ndarray:
    folds = np.asarray(folds).reshape(-1)
    unique = np.unique(folds)
    if len(unique) < 2:
        raise ValueError("cross-fitting needs at least two folds")
    return unique


def crossfit_oos_r2(signal: np.ndarray, target: np.ndarray, folds: np.ndarray) -> dict:
    """Out-of-sample R^2 of ``beta * signal`` versus the zero forecast.

    For each fold, ``beta`` is the no-intercept least-squares slope fitted on the
    other folds. ``R^2 = 1 - SSE / sum(target^2)``, the convention used for return
    forecasts where the natural benchmark is zero rather than the sample mean.
    """
    signal, target = np.asarray(signal, float), np.asarray(target, float)
    folds = np.asarray(folds).reshape(-1)
    forecast = np.empty_like(target)
    betas = {}
    for fold in _fold_ids(folds):
        train, test = folds != fold, folds == fold
        denominator = float(np.dot(signal[train], signal[train]))
        beta = float(np.dot(signal[train], target[train]) / denominator) if denominator else 0.0
        betas[str(fold)] = beta
        forecast[test] = beta * signal[test]
    total = float(np.dot(target, target))
    per_fold = {}
    for fold in _fold_ids(folds):
        mask = folds == fold
        fold_total = float(np.dot(target[mask], target[mask]))
        per_fold[str(fold)] = (
            1.0 - float(((target[mask] - forecast[mask]) ** 2).sum()) / fold_total
            if fold_total
            else float("nan")
        )
    return {
        "oos_r2": 1.0 - float(((target - forecast) ** 2).sum()) / total if total else float("nan"),
        "per_fold_oos_r2": per_fold,
        "per_fold_beta": betas,
    }


def crossfit_markout(
    signal: np.ndarray,
    target_bps: np.ndarray,
    half_spread_bps: np.ndarray,
    folds: np.ndarray,
    bins: int = 10,
) -> list[dict]:
    """Mean signed forward move by signal quantile bin, with cross-fitted edges.

    Bin edges for each fold come from the signal quantiles of the other folds, so
    bin membership never uses the scored rows. ``taker_edge_bps`` is the move in
    the direction of the bin's mean signal minus the half-spread: positive values
    mean the average forecast move would exceed the cost of crossing to the mid.
    """
    signal = np.asarray(signal, float)
    target_bps = np.asarray(target_bps, float)
    half_spread_bps = np.asarray(half_spread_bps, float)
    folds = np.asarray(folds).reshape(-1)
    assignment = np.empty(len(signal), dtype=np.int64)
    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    for fold in _fold_ids(folds):
        train, test = folds != fold, folds == fold
        edges = np.quantile(signal[train], quantiles)
        assignment[test] = np.searchsorted(edges, signal[test], side="right")
    output = []
    for index in range(bins):
        mask = assignment == index
        if not mask.any():
            output.append({"bin": index + 1, "count": 0})
            continue
        mean_signal = float(signal[mask].mean())
        direction = float(np.sign(mean_signal))
        mean_move = float(target_bps[mask].mean())
        mean_half = float(half_spread_bps[mask].mean())
        output.append(
            {
                "bin": index + 1,
                "count": int(mask.sum()),
                "mean_signal": mean_signal,
                "mean_forward_move_bps": mean_move,
                "mean_half_spread_bps": mean_half,
                "taker_edge_bps": direction * mean_move - mean_half if direction else float("nan"),
                "fraction_move_beyond_half_spread": (
                    float(np.mean(direction * target_bps[mask] > half_spread_bps[mask]))
                    if direction
                    else float("nan")
                ),
            }
        )
    return output
