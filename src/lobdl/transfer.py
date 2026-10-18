from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def sample_message_blocks(book: np.ndarray, messages_per_representation: int = 10) -> np.ndarray:
    """Retain the post-message state at each complete fixed message block."""
    if book.ndim != 2 or book.shape[1] != 40:
        raise ValueError("expected a vendor book with shape (messages, 40)")
    if messages_per_representation <= 0:
        raise ValueError("messages_per_representation must be positive")
    return book[messages_per_representation - 1 :: messages_per_representation]


def calibration_boundary(rows: int, fraction: float = 0.2) -> int:
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must be strictly between zero and one")
    return int(np.floor(rows * fraction))


def decimal_precision_powers(calibration: np.ndarray) -> np.ndarray:
    """Smallest per-column nonnegative powers yielding strict absolute values below one."""
    if calibration.ndim != 2 or not len(calibration):
        raise ValueError("decimal scaling needs a nonempty 2-D calibration prefix")
    maximum = np.max(np.abs(calibration.astype(np.float64)), axis=0)
    powers = np.zeros(calibration.shape[1], dtype=np.int16)
    positive = maximum > 0
    powers[positive] = np.floor(np.log10(maximum[positive])).astype(np.int16) + 1
    # Exact powers of ten require one additional digit because the bound is strict.
    scaled = maximum[positive] / np.power(10.0, powers[positive])
    powers[positive] += (scaled >= 1.0).astype(np.int16)
    return powers


def apply_decimal_precision(values: np.ndarray, powers: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or powers.shape != (values.shape[1],):
        raise ValueError("values and decimal powers are not column-aligned")
    return (values.astype(np.float64) / np.power(10.0, powers)).astype(np.float32)


def mid_and_spread(book: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ask, bid = book[:, 0].astype(np.float64), book[:, 2].astype(np.float64)
    return (ask + bid) / 2.0, ask - bid


def forward_mean_returns(book: np.ndarray, horizon_rows: int) -> np.ndarray:
    """Forward-mean relative return, with NaN where the horizon is unavailable."""
    if horizon_rows <= 0:
        raise ValueError("horizon_rows must be positive")
    mid, _ = mid_and_spread(book)
    returns = np.full(len(mid), np.nan, dtype=np.float64)
    if len(mid) <= horizon_rows:
        return returns
    cumulative = np.cumsum(np.concatenate(([0.0], mid)))
    end = np.arange(len(mid) - horizon_rows)
    future = (cumulative[end + horizon_rows + 1] - cumulative[end + 1]) / horizon_rows
    returns[end] = (future - mid[end]) / np.maximum(np.abs(mid[end]), 1e-12)
    return returns


def label_returns(returns: np.ndarray, alpha: float) -> np.ndarray:
    labels = np.full(len(returns), -1, dtype=np.int8)
    finite = np.isfinite(returns)
    labels[finite] = np.where(
        returns[finite] >= alpha,
        2,
        np.where(returns[finite] <= -alpha, 0, 1),
    )
    return labels


def eligible_ends(
    rows: int,
    boundary: int,
    horizon_rows: int,
    window: int = 100,
    split: str = "evaluation",
) -> np.ndarray:
    if split == "calibration":
        first = window - 1
        stop = boundary - horizon_rows
    elif split == "evaluation":
        first = max(window - 1, boundary)
        stop = rows - horizon_rows
    else:
        raise ValueError("split must be calibration or evaluation")
    return np.arange(first, max(first, stop), dtype=np.int64)


def class_proportions(labels: np.ndarray) -> np.ndarray:
    if not len(labels) or np.any((labels < 0) | (labels > 2)):
        raise ValueError("class proportions require nonempty labels in {0,1,2}")
    return np.bincount(labels, minlength=3).astype(np.float64) / len(labels)


def select_distribution_matched_alpha(
    returns: np.ndarray,
    target_proportions: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Ascending grid makes the registered exact-tie rule select the smaller alpha."""
    if np.any(np.diff(grid) < 0):
        raise ValueError("alpha grid must be ascending")
    if target_proportions.shape != (3,) or not np.isclose(target_proportions.sum(), 1.0):
        raise ValueError("target proportions must be a three-class probability vector")
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in grid:
        labels = label_returns(returns, float(alpha))
        labels = labels[labels >= 0]
        proportions = class_proportions(labels)
        distance = float(0.5 * np.abs(proportions - target_proportions).sum())
        if best is None or distance < best[1]:
            best = float(alpha), distance, proportions
    if best is None:
        raise ValueError("alpha grid is empty")
    return best


@dataclass(frozen=True)
class TargetArrays:
    scaled_book: np.ndarray
    raw_book: np.ndarray
    ends: np.ndarray


class TargetWindowDataset(Dataset):
    def __init__(self, arrays: TargetArrays, window: int = 100):
        self.arrays = arrays
        self.window = window

    def __len__(self) -> int:
        return len(self.arrays.ends)

    def __getitem__(self, item: int) -> torch.Tensor:
        end = int(self.arrays.ends[item])
        values = self.arrays.scaled_book[end - self.window + 1 : end + 1]
        return torch.from_numpy(np.asarray(values, dtype=np.float32))
