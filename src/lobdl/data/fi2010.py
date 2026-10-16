from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HORIZONS = (10, 20, 30, 50, 100)
LABEL_TO_INDEX = {1: 2, 2: 1, 3: 0}  # shipped up/flat/down -> down/flat/up


@dataclass(frozen=True)
class Segment:
    day: int
    stock: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FI2010Day:
    day: int
    features: np.ndarray
    labels: np.ndarray
    segments: tuple[Segment, ...]
    source: Path


def sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path: str | Path) -> np.ndarray:
    """Load FI-2010 text and normalize orientation to (events, 149)."""
    path = Path(path)
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"{path}: expected 2-D matrix, got {raw.shape}")
    if raw.shape[0] == 149:
        raw = raw.T
    if raw.shape[1] != 149:
        raise ValueError(f"{path}: expected 149 columns after orientation, got {raw.shape}")
    labels = raw[:, 144:149]
    if not np.isin(labels, (1.0, 2.0, 3.0)).all():
        bad = np.unique(labels[~np.isin(labels, (1.0, 2.0, 3.0))])
        raise ValueError(f"{path}: invalid shipped labels {bad[:10]}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: non-finite values")
    return raw


def split_stock_segments(features: np.ndarray, day: int) -> tuple[Segment, ...]:
    """Detect the four stock seams and reject ambiguous segmentation.

    FI-2010 fold files concatenate five instruments without a stock-id column.
    Decimal-precision prices make instrument changes orders of magnitude larger
    than valid within-instrument top-of-book moves. The manifest is accepted only
    when exactly four isolated jumps exceed both 5% and 100x the robust local scale.
    """
    if features.shape[1] < 4:
        raise ValueError("need raw LOB columns")
    mid = (features[:, 0].astype(np.float64) + features[:, 2]) / 2.0
    relative_jump = np.abs(np.diff(mid)) / np.maximum(np.abs(mid[:-1]), 1e-12)
    positive = relative_jump[relative_jump > 0]
    local_scale = np.median(positive) if positive.size else 0.0
    threshold = max(0.05, 100.0 * local_scale)
    candidates = np.flatnonzero(relative_jump > threshold) + 1
    if candidates.size != 4 or np.any(np.diff(candidates) < 1000):
        ranked = np.argsort(relative_jump)[-10:][::-1] + 1
        raise ValueError(
            f"day {day}: stock seams ambiguous; threshold={threshold:.6g}, "
            f"candidates={candidates.tolist()}, largest={ranked.tolist()}"
        )
    bounds = np.concatenate(([0], candidates, [len(features)]))
    return tuple(
        Segment(day=day, stock=i + 1, start=int(a), stop=int(b))
        for i, (a, b) in enumerate(itertools.pairwise(bounds))
    )


def load_day(path: str | Path, day: int, manifest: str | Path | None = None) -> FI2010Day:
    raw = load_matrix(path)
    if manifest is None:
        segments = split_stock_segments(raw[:, :40], day)
    else:
        entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
        segments = tuple(Segment(**item) for item in entries if item["day"] == day)
        validate_segments(segments, len(raw), day)
    return FI2010Day(day, raw[:, :144], raw[:, 144:149].astype(np.int8), segments, Path(path))


def validate_segments(segments: tuple[Segment, ...], n: int, day: int) -> None:
    if len(segments) != 5:
        raise ValueError(f"day {day}: expected five stock segments")
    if segments[0].start != 0 or segments[-1].stop != n:
        raise ValueError(f"day {day}: manifest does not cover the file")
    for left, right in itertools.pairwise(segments):
        if left.stop != right.start:
            raise ValueError(f"day {day}: manifest gap or overlap")


def decode_labels(labels: np.ndarray, horizon_events: int) -> np.ndarray:
    try:
        column = HORIZONS.index(horizon_events)
    except ValueError as exc:
        raise ValueError(f"unsupported horizon {horizon_events}; choose {HORIZONS}") from exc
    return np.vectorize(LABEL_TO_INDEX.__getitem__, otypes=[np.int8])(labels[:, column])


def iter_window_indices(
    segments: tuple[Segment, ...], window: int, horizon: int, purge: int = 0
) -> Iterator[tuple[Segment, int]]:
    """Yield window end indices whose inputs and future labels stay in one segment."""
    for segment in segments:
        first_end = segment.start + window - 1 + purge
        last_end = segment.stop - horizon - purge - 1
        for end in range(first_end, last_end + 1):
            yield segment, end


def build_windows(
    day: FI2010Day,
    horizon_events: int,
    window: int = 100,
    purge: int = 0,
    lob_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # FI-2010 is sampled once per book event. The five shipped label columns
    # correspond to 10, 20, 30, 50, and 100 future rows respectively.
    horizon_rows = horizon_events
    labels = decode_labels(day.labels, horizon_events)
    indices = list(iter_window_indices(day.segments, window, horizon_rows, purge))
    feature_count = 40 if lob_only else 144
    x = np.empty((len(indices), window, feature_count), dtype=np.float32)
    y = np.empty(len(indices), dtype=np.int64)
    group = np.empty((len(indices), 2), dtype=np.int16)
    for i, (segment, end) in enumerate(indices):
        x[i] = day.features[end - window + 1 : end + 1, :feature_count]
        y[i] = labels[end]
        group[i] = (segment.day, segment.stock)
    return x, y, group


def reconstruct_single_future_labels(features: np.ndarray, horizon_events: int, alpha: float) -> np.ndarray:
    rows = horizon_events
    mid = (features[:, 0].astype(np.float64) + features[:, 2]) / 2.0
    out = np.full(len(mid), -1, dtype=np.int8)
    returns = (mid[rows:] - mid[:-rows]) / np.maximum(np.abs(mid[:-rows]), 1e-12)
    tolerance = 8.0 * np.finfo(features.dtype).eps if np.issubdtype(features.dtype, np.floating) else 0.0
    out[:-rows] = np.where(
        returns >= alpha - tolerance, 2, np.where(returns <= -alpha + tolerance, 0, 1)
    )
    return out


def reconstruct_smoothed_ablation(features: np.ndarray, horizon_events: int, alpha: float) -> np.ndarray:
    rows = horizon_events
    mid = (features[:, 0].astype(np.float64) + features[:, 2]) / 2.0
    out = np.full(len(mid), -1, dtype=np.int8)
    sums = np.cumsum(np.concatenate(([0.0], mid)))
    for t in range(len(mid) - rows):
        forward_mean = (sums[t + rows + 1] - sums[t + 1]) / rows
        change = (forward_mean - mid[t]) / max(abs(mid[t]), 1e-12)
        out[t] = 2 if change > alpha else (0 if change < -alpha else 1)
    return out
