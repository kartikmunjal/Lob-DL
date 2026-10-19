from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.fi2010 import Segment


@dataclass(frozen=True)
class TrainStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    fit_split: str = "train"

    @classmethod
    def fit(cls, x: np.ndarray, split: str) -> TrainStandardizer:
        if split != "train":
            raise AssertionError("normalization may be fitted on training data only")
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        mean = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(mean, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


def assert_no_window_seams(groups: np.ndarray, expected_rows: int) -> None:
    if groups.shape != (expected_rows, 2):
        raise AssertionError("every window needs exactly one stock/day group")
    if np.any(groups <= 0):
        raise AssertionError("invalid group identifier")


def assert_future_inside_segment(end: int, horizon_rows: int, segment: Segment) -> None:
    if end + horizon_rows >= segment.stop:
        raise AssertionError("label horizon crosses stock/day boundary")


def class_balance(y: np.ndarray) -> dict[str, dict[str, float | int]]:
    names = ("down", "stationary", "up")
    return {
        name: {"count": int((y == index).sum()), "fraction": float((y == index).mean())}
        for index, name in enumerate(names)
    }


def assert_split_days(train: set[int], validation: set[int], test: set[int]) -> None:
    if train & validation or train & test or validation & test:
        raise AssertionError("temporal split overlap")
    if max(train) >= min(validation) or max(validation) >= min(test):
        raise AssertionError("splits are not strictly temporal")
