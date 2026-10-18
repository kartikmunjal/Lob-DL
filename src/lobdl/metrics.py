from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_fscore_support


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in itertools.pairwise(edges):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def reliability_curve(y: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> list[dict]:
    """Top-label reliability bins used for the registered calibration plots."""
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = []
    for index, (low, high) in enumerate(itertools.pairwise(edges)):
        mask = (confidence > low) & (confidence <= high)
        output.append(
            {
                "bin": index + 1,
                "lower_exclusive": float(low),
                "upper_inclusive": float(high),
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return output


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=[0, 1, 2], zero_division=0)
    return {
        "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
        "support": support.tolist(), "macro_f1": float(f1.mean()),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "ece": expected_calibration_error(y, probabilities),
        "reliability_curve": reliability_curve(y, probabilities),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
    }


def moving_block_indices(groups: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for group in np.unique(groups, axis=0):
        positions = np.flatnonzero(np.all(groups == group, axis=1))
        if not len(positions):
            continue
        blocks = []
        while sum(len(block) for block in blocks) < len(positions):
            start = int(rng.integers(0, max(1, len(positions) - block_length + 1)))
            blocks.append(positions[start : start + block_length])
        sampled.append(np.concatenate(blocks)[: len(positions)])
    return np.concatenate(sampled)


def bootstrap_sufficient_sums(
    row_statistics: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Moving-block bootstrap sums without materializing every sampled window.

    Each statistic must be additive by row. Contiguous block sums are obtained
    from cumulative sums, preserving the registered within-group resampling.
    """
    if row_statistics.ndim != 2 or len(row_statistics) != len(groups):
        raise ValueError("row statistics and groups must be row-aligned 2-D arrays")
    if replicates <= 0 or block_length <= 0:
        raise ValueError("replicates and block length must be positive")
    rng = np.random.default_rng(seed)
    output = np.zeros((replicates, row_statistics.shape[1]), dtype=np.float64)
    for group in np.unique(groups, axis=0):
        positions = np.flatnonzero(np.all(groups == group, axis=1))
        if len(positions) > 1 and not np.all(np.diff(positions) == 1):
            raise ValueError("each bootstrap group must occupy one contiguous row segment")
        values = np.asarray(row_statistics[positions], dtype=np.float64)
        cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
        blocks = int(np.ceil(len(values) / block_length))
        final_length = len(values) - (blocks - 1) * block_length
        high = max(1, len(values) - block_length + 1)
        starts = rng.integers(0, high, size=(replicates, blocks))
        for block in range(blocks):
            length = block_length if block < blocks - 1 else final_length
            start = starts[:, block]
            output += cumulative[start + length] - cumulative[start]
    return output


def classification_row_statistics(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Additive sufficient statistics for registered classification metrics."""
    if probabilities.shape != (len(y), 3):
        raise ValueError("classification probabilities must have shape (rows, 3)")
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == y
    statistics = np.zeros((len(y), 9 + 15 * 3), dtype=np.float64)
    statistics[np.arange(len(y)), y * 3 + prediction] = 1.0
    edges = np.linspace(0.0, 1.0, 16)
    bins = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, 14)
    offset = 9 + bins * 3
    statistics[np.arange(len(y)), offset] = 1.0
    statistics[np.arange(len(y)), offset + 1] = correct
    statistics[np.arange(len(y)), offset + 2] = confidence
    return statistics


def metrics_from_sufficient_sums(statistics: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized metrics for one or more rows of additive sufficient statistics."""
    values = np.atleast_2d(statistics).astype(np.float64, copy=False)
    confusion = values[:, :9].reshape(-1, 3, 3)
    true_count = confusion.sum(axis=2)
    predicted_count = confusion.sum(axis=1)
    true_positive = np.diagonal(confusion, axis1=1, axis2=2)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros_like(true_positive),
        where=true_count != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    total = confusion.sum(axis=(1, 2))
    correct = true_positive.sum(axis=1)
    numerator = correct * total - (true_count * predicted_count).sum(axis=1)
    denominator = np.sqrt(
        (total**2 - (predicted_count**2).sum(axis=1))
        * (total**2 - (true_count**2).sum(axis=1))
    )
    mcc = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    calibration = values[:, 9:].reshape(-1, 15, 3)
    ece = np.abs(calibration[:, :, 1] - calibration[:, :, 2]).sum(axis=1) / total
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": f1.mean(axis=1),
        "mcc": mcc,
        "ece": ece,
    }


def bootstrap_metrics(y: np.ndarray, probabilities: np.ndarray, groups: np.ndarray, replicates: int = 2000, block_length: int = 100, seed: int = 20261013) -> dict:
    statistics = classification_row_statistics(y, probabilities)
    sums = bootstrap_sufficient_sums(statistics, groups, replicates, block_length, seed)
    values = metrics_from_sufficient_sums(sums)
    output = {}
    for name, value in values.items():
        samples = np.asarray(value)
        lower = np.quantile(samples, 0.025, axis=0)
        upper = np.quantile(samples, 0.975, axis=0)
        output[name] = {
            "lower": lower.tolist() if lower.ndim else float(lower),
            "upper": upper.tolist() if upper.ndim else float(upper),
        }
    return output


def paired_bootstrap_metric_differences(
    y: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    groups: np.ndarray,
    replicates: int = 2000,
    block_length: int = 100,
    seed: int = 20261013,
) -> dict:
    """Paired moving-block intervals for metric(A) minus metric(B)."""
    if probabilities_a.shape != probabilities_b.shape:
        raise ValueError("paired probability arrays must have identical shape")
    names = ("precision", "recall", "f1", "macro_f1", "mcc", "ece")
    row_statistics = np.column_stack(
        (
            classification_row_statistics(y, probabilities_a),
            classification_row_statistics(y, probabilities_b),
        )
    )
    sums = bootstrap_sufficient_sums(
        row_statistics, groups, replicates, block_length, seed
    )
    width = row_statistics.shape[1] // 2
    metrics_a = metrics_from_sufficient_sums(sums[:, :width])
    metrics_b = metrics_from_sufficient_sums(sums[:, width:])
    values = {name: metrics_a[name] - metrics_b[name] for name in names}
    output = {}
    for name, value in values.items():
        samples = np.asarray(value)
        point_a = classification_metrics(y, probabilities_a)[name]
        point_b = classification_metrics(y, probabilities_b)[name]
        point = np.asarray(point_a) - np.asarray(point_b)
        lower = np.quantile(samples, 0.025, axis=0)
        upper = np.quantile(samples, 0.975, axis=0)
        output[name] = {
            "difference": point.tolist() if point.ndim else float(point),
            "lower": lower.tolist() if lower.ndim else float(lower),
            "upper": upper.tolist() if upper.ndim else float(upper),
        }
    return output
