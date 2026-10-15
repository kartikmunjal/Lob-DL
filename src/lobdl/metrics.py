from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_fscore_support


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
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
    values = {"macro_f1": [], "mcc": [], "ece": []}
    for _ in range(replicates):
        index = moving_block_indices(groups, block_length, rng)
        result = classification_metrics(y[index], probabilities[index])
        for name in values:
            values[name].append(result[name])
    return {name: {"lower": float(np.quantile(value, 0.025)), "upper": float(np.quantile(value, 0.975))} for name, value in values.items()}

