"""Input representations for the Part C representation x model factorial.

Three representations of the same FI-2010 windows (see docs/STUDY2_PROTOCOL.md Part C):

* ``raw``  - the 40 decimal-precision book columns;
* ``eng``  - the 29 locked causal features of ``lobdl.features``;
* ``both`` - raw followed by eng.

Neural models consume (window, channels) sequences; tree models consume one flat
vector per prediction row. Engineered features are computed per (stock, day)
segment, so no feature mixes information across instruments or days, and
standardization statistics are fitted on training days only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data.fi2010 import HORIZONS, LABEL_TO_INDEX, iter_window_indices
from .datasets import load_segments
from .features import causal_features

REPRESENTATIONS = ("raw", "eng", "both")
RAW_COLUMNS = 40
ENG_COLUMNS = 29
RAW_LAGS = (0, 1, 2, 5, 10, 20, 50)
ENG_CLIP = 5.0
TRAIN_DAYS = tuple(range(1, 7))


def channel_count(representation: str) -> int:
    _check(representation)
    return {"raw": RAW_COLUMNS, "eng": ENG_COLUMNS, "both": RAW_COLUMNS + ENG_COLUMNS}[
        representation
    ]


def _check(representation: str) -> None:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")


def engineered_day(root: Path, day: int, cache_dir: Path | None = None) -> np.ndarray:
    """Unstandardized causal features for every row of one day, computed per segment."""
    root = Path(root)
    cache = None if cache_dir is None else Path(cache_dir) / f"eng_day_{day:02d}.npy"
    if cache is not None and cache.exists():
        return np.load(cache)
    matrix = np.load(root / f"day_{day:02d}.npy", mmap_mode="r")
    output = np.zeros((matrix.shape[0], ENG_COLUMNS), dtype=np.float32)
    for segment in load_segments(root)[day]:
        raw = np.asarray(matrix[segment.start : segment.stop, :RAW_COLUMNS])
        features, _ = causal_features(raw)
        output[segment.start : segment.stop] = features
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, output)
    return output


class EngineeredScaler:
    """Per-feature z-score fitted on training-day rows, clipped to +/- ``ENG_CLIP``."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        if mean.shape != (ENG_COLUMNS,) or std.shape != (ENG_COLUMNS,):
            raise ValueError(f"mean and std must have shape ({ENG_COLUMNS},)")
        self.mean = mean.astype(np.float64)
        self.std = np.where(std > 1e-12, std, 1.0).astype(np.float64)

    @classmethod
    def fit(
        cls,
        root: Path,
        days: tuple[int, ...] = TRAIN_DAYS,
        cache_dir: Path | None = None,
    ) -> EngineeredScaler:
        if any(day > 6 for day in days):
            raise ValueError("engineered scaling statistics may use training days 1-6 only")
        total = np.zeros(ENG_COLUMNS)
        squares = np.zeros(ENG_COLUMNS)
        count = 0
        for day in days:
            values = engineered_day(root, day, cache_dir).astype(np.float64)
            total += values.sum(axis=0)
            squares += (values**2).sum(axis=0)
            count += len(values)
        mean = total / count
        variance = np.maximum(squares / count - mean**2, 0.0)
        return cls(mean, np.sqrt(variance))

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(values, dtype=np.float64) - self.mean) / self.std
        return np.clip(scaled, -ENG_CLIP, ENG_CLIP).astype(np.float32)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": ENG_CLIP}

    @classmethod
    def from_json(cls, payload: dict) -> EngineeredScaler:
        return cls(np.asarray(payload["mean"]), np.asarray(payload["std"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")


class FactorialWindowDataset(Dataset):
    """Segment-safe 100-step windows of one representation (neural models)."""

    def __init__(
        self,
        root: str | Path,
        days: list[int],
        horizon: int,
        representation: str,
        scaler: EngineeredScaler | None,
        window: int = 100,
        purge: int = 0,
        cache_dir: Path | None = None,
    ) -> None:
        _check(representation)
        if representation != "raw" and scaler is None:
            raise ValueError("engineered representations require a fitted scaler")
        self.root = Path(root)
        self.representation = representation
        self.window = window
        self.label_column = HORIZONS.index(horizon)
        segments = load_segments(self.root)
        horizon_rows = horizon // 10
        self.raw = {
            day: np.load(self.root / f"day_{day:02d}.npy", mmap_mode="r") for day in days
        }
        self.eng = {}
        if representation != "raw":
            for day in days:
                self.eng[day] = scaler.transform(engineered_day(self.root, day, cache_dir))
        self.index: list[tuple[int, int, int]] = []
        for day in days:
            self.index.extend(
                (day, segment.stock, end)
                for segment, end in iter_window_indices(
                    segments[day], window, horizon_rows, purge
                )
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        day, stock, end = self.index[item]
        span = slice(end - self.window + 1, end + 1)
        parts = []
        if self.representation in ("raw", "both"):
            parts.append(np.asarray(self.raw[day][span, :RAW_COLUMNS], dtype=np.float32))
        if self.representation in ("eng", "both"):
            parts.append(self.eng[day][span])
        x = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        label = LABEL_TO_INDEX[int(self.raw[day][end, 144 + self.label_column])]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            label,
            torch.tensor((day, stock), dtype=torch.int16),
            end,
        )


def flat_feature_names(representation: str) -> list[str]:
    _check(representation)
    names: list[str] = []
    if representation in ("raw", "both"):
        names += [f"raw_c{column:02d}_lag{lag}" for lag in RAW_LAGS for column in range(RAW_COLUMNS)]
    if representation in ("eng", "both"):
        names += [f"eng_{index:02d}" for index in range(ENG_COLUMNS)]
    return names


def flat_matrix(
    root: str | Path,
    days: list[int],
    horizon: int,
    representation: str,
    purge: int = 0,
    window: int = 100,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One flat row per prediction window for tree models: ``(X, y, groups)``.

    ``raw`` stacks the book at row lags ``RAW_LAGS`` before the prediction row,
    ``eng`` is the prediction-row engineered vector (unscaled, as in Study 1), and
    ``both`` concatenates the two. Labels are the shipped labels in Study 1 encoding.
    """
    _check(representation)
    root = Path(root)
    segments = load_segments(root)
    horizon_rows = horizon // 10
    label_column = HORIZONS.index(horizon)
    xs, ys, groups = [], [], []
    for day in days:
        matrix = np.load(root / f"day_{day:02d}.npy", mmap_mode="r")
        engineered = engineered_day(root, day, cache_dir) if representation != "raw" else None
        for segment in segments[day]:
            ends = np.fromiter(
                (
                    end
                    for _, end in iter_window_indices((segment,), window, horizon_rows, purge)
                ),
                dtype=np.int64,
            )
            if not len(ends):
                continue
            blocks = []
            if representation in ("raw", "both"):
                book = np.asarray(matrix[segment.start : segment.stop, :RAW_COLUMNS], np.float32)
                local = ends - segment.start
                blocks.append(np.hstack([book[local - lag] for lag in RAW_LAGS]))
            if representation in ("eng", "both"):
                blocks.append(engineered[ends])
            xs.append(np.hstack(blocks).astype(np.float32))
            shipped = np.asarray(matrix[ends, 144 + label_column]).astype(np.int8)
            ys.append(np.vectorize(LABEL_TO_INDEX.__getitem__, otypes=[np.int8])(shipped))
            groups.append(np.tile((day, segment.stock), (len(ends), 1)))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)
