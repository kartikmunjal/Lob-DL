"""Cluster-level inference for paired model comparisons (Study 2, A1).

The registered Study 1 intervals resample contiguous windows *within* each
stock-day. They quantify sampling noise inside a day but not variation between
days or instruments. Here the unit of inference is the cluster itself (a
``(day, stock)`` segment on FI-2010, a ticker on LOBSTER): each cluster yields one
paired metric difference, averaged over fitted seeds, and inference uses only
those few cluster-level numbers.
"""

from __future__ import annotations

import itertools

import numpy as np

from .metrics import classification_row_statistics, metrics_from_sufficient_sums

EXACT_SIGN_FLIP_MAX_CLUSTERS = 20


def cluster_keys(groups: np.ndarray, seed_column: int | None = -1) -> np.ndarray:
    """Return per-row cluster keys with the seed column removed."""
    groups = np.asarray(groups)
    if groups.ndim != 2:
        raise ValueError("groups must be a 2-D array")
    if seed_column is None:
        return groups
    keep = np.ones(groups.shape[1], dtype=bool)
    keep[seed_column] = False
    if not keep.any():
        raise ValueError("groups must contain a cluster column besides the seed column")
    return groups[:, keep]


def cluster_metric_differences(
    y: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    groups: np.ndarray,
    metric: str = "macro_f1",
    seed_column: int | None = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired metric(A) - metric(B) per cluster, averaged over seeds.

    ``groups`` uses the Study 1 layout: cluster columns followed by a one-based seed
    index column. Each (cluster, seed) cell gets its own metric difference; cells are
    then averaged within a cluster so every cluster carries equal weight.
    Returns ``(cluster_ids, differences)`` sorted by cluster id.
    """
    if probabilities_a.shape != probabilities_b.shape:
        raise ValueError("paired probability arrays must have identical shape")
    if len(y) != len(groups) or len(y) != len(probabilities_a):
        raise ValueError("labels, probabilities, and groups must be row-aligned")
    groups = np.asarray(groups)
    clusters = cluster_keys(groups, seed_column)
    seeds = (
        np.zeros(len(groups), dtype=np.int64)
        if seed_column is None
        else groups[:, seed_column].astype(np.int64)
    )
    stats_a = classification_row_statistics(y, probabilities_a)
    stats_b = classification_row_statistics(y, probabilities_b)
    cluster_ids, cluster_index = np.unique(clusters, axis=0, return_inverse=True)
    cluster_index = cluster_index.reshape(-1)
    seed_ids, seed_index = np.unique(seeds, return_inverse=True)
    cell = cluster_index * len(seed_ids) + seed_index.reshape(-1)
    cells = len(cluster_ids) * len(seed_ids)
    sums_a = np.zeros((cells, stats_a.shape[1]))
    sums_b = np.zeros_like(sums_a)
    np.add.at(sums_a, cell, stats_a)
    np.add.at(sums_b, cell, stats_b)
    present = np.zeros(cells, dtype=bool)
    present[cell] = True
    differences = np.full(cells, np.nan)
    differences[present] = (
        metrics_from_sufficient_sums(sums_a[present])[metric]
        - metrics_from_sufficient_sums(sums_b[present])[metric]
    )
    per_cluster = np.nanmean(differences.reshape(len(cluster_ids), len(seed_ids)), axis=1)
    return cluster_ids, per_cluster


def sign_flip_pvalue(
    differences: np.ndarray,
    monte_carlo_replicates: int = 100_000,
    seed: int = 20261103,
) -> dict:
    """Two-sided randomization test of mean cluster difference = 0.

    Under the null that A and B are exchangeable within each cluster, each
    cluster difference is symmetric about zero. For up to
    ``EXACT_SIGN_FLIP_MAX_CLUSTERS`` clusters all ``2^C`` sign assignments are
    enumerated (no Monte Carlo error); otherwise random sign flips are drawn and
    the p-value uses the ``(b + 1) / (B + 1)`` correction.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite 1-D array")
    observed = abs(values.mean())
    tolerance = 1e-12 * max(1.0, observed)
    count = len(values)
    if count <= EXACT_SIGN_FLIP_MAX_CLUSTERS:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=count)))
        null = np.abs(signs @ values) / count
        extreme = int(np.sum(null >= observed - tolerance))
        return {
            "p_value": extreme / len(signs),
            "method": "exact_sign_flip",
            "assignments": len(signs),
            "minimum_attainable_p": 2.0 / len(signs),
            "clusters": count,
        }
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(monte_carlo_replicates, count))
    null = np.abs(signs @ values) / count
    extreme = int(np.sum(null >= observed - tolerance))
    return {
        "p_value": (extreme + 1) / (monte_carlo_replicates + 1),
        "method": "monte_carlo_sign_flip",
        "assignments": int(monte_carlo_replicates),
        "minimum_attainable_p": 1.0 / (monte_carlo_replicates + 1),
        "clusters": count,
    }


def cluster_bootstrap_interval(
    differences: np.ndarray,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20261103,
) -> dict:
    """Percentile bootstrap of the mean, resampling whole clusters."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.quantile(means, tail)),
        "ci_upper": float(np.quantile(means, 1.0 - tail)),
        "between_cluster_sd": float(values.std(ddof=1)),
        "clusters_favoring_a": int(np.sum(values > 0)),
        "clusters": len(values),
        "replicates": int(replicates),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down family-wise adjusted p-values, in the input order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p-values must be 1-D")
    if len(values) == 0:
        return values.copy()
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    scaled = values[order] * (len(values) - np.arange(len(values)))
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_cluster_inference(
    y: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    groups: np.ndarray,
    metrics: tuple[str, ...] = ("macro_f1", "mcc"),
    seed_column: int | None = -1,
    bootstrap_replicates: int = 10_000,
) -> dict:
    """Cluster-level summary of one paired comparison across several metrics."""
    output: dict = {}
    for metric in metrics:
        ids, differences = cluster_metric_differences(
            y, probabilities_a, probabilities_b, groups, metric, seed_column
        )
        output[metric] = {
            "cluster_ids": ids.tolist(),
            "cluster_differences": differences.tolist(),
            "sign_flip": sign_flip_pvalue(differences),
            "cluster_bootstrap": cluster_bootstrap_interval(
                differences, replicates=bootstrap_replicates
            ),
        }
    return output
