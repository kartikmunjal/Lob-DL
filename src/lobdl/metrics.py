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


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=[0, 1, 2], zero_division=0)
    return {
        "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
        "support": support.tolist(), "macro_f1": float(f1.mean()),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "ece": expected_calibration_error(y, probabilities),
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


def bootstrap_metrics(y: np.ndarray, probabilities: np.ndarray, groups: np.ndarray, replicates: int = 2000, block_length: int = 100, seed: int = 20261013) -> dict:
    rng = np.random.default_rng(seed)
    values = {
        "precision": [],
        "recall": [],
        "f1": [],
        "macro_f1": [],
        "mcc": [],
        "ece": [],
    }
    for _ in range(replicates):
        index = moving_block_indices(groups, block_length, rng)
        result = classification_metrics(y[index], probabilities[index])
        for name, samples in values.items():
            samples.append(result[name])
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
    rng = np.random.default_rng(seed)
    names = ("precision", "recall", "f1", "macro_f1", "mcc", "ece")
    values = {name: [] for name in names}
    for _ in range(replicates):
        index = moving_block_indices(groups, block_length, rng)
        a = classification_metrics(y[index], probabilities_a[index])
        b = classification_metrics(y[index], probabilities_b[index])
        for name in names:
            values[name].append(np.asarray(a[name]) - np.asarray(b[name]))
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
