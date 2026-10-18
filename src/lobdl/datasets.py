from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data.fi2010 import HORIZONS, LABEL_TO_INDEX, Segment, iter_window_indices


def load_segments(root: Path) -> dict[int, tuple[Segment, ...]]:
    entries = json.loads((root / "segments.json").read_text(encoding="utf-8"))
    return {
        day: tuple(Segment(**item) for item in entries if item["day"] == day)
        for day in range(1, 11)
    }


class FIWindowDataset(Dataset):
    """Lazy, segment-safe windows over memory-mapped day arrays."""

    def __init__(
        self,
        root: str | Path,
        days: list[int],
        horizon: int,
        window: int = 100,
        purge: int = 0,
        levels: int = 10,
    ):
        self.root = Path(root)
        self.days = days
        self.horizon = horizon
        self.window = window
        if levels not in (1, 5, 10):
            raise ValueError("levels must be one of 1, 5, or 10")
        self.feature_count = levels * 4
        self.arrays = {day: np.load(self.root / f"day_{day:02d}.npy", mmap_mode="r") for day in days}
        segments = load_segments(self.root)
        horizon_rows = horizon // 10
        self.index: list[tuple[int, int, int]] = []
        for day in days:
            self.index.extend((day, segment.stock, end) for segment, end in iter_window_indices(segments[day], window, horizon_rows, purge))
        self.label_column = HORIZONS.index(horizon)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        day, stock, end = self.index[item]
        array = self.arrays[day]
        x = np.asarray(
            array[end - self.window + 1 : end + 1, : self.feature_count],
            dtype=np.float32,
        ).copy()
        shipped = int(array[end, 144 + self.label_column])
        y = LABEL_TO_INDEX[shipped]
        return torch.from_numpy(x), y, torch.tensor((day, stock), dtype=torch.int16), end


def collate_predictions(loader, model: torch.nn.Module, device: torch.device):
    probabilities, labels, groups, ends = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for x, y, group, end in loader:
            logits = model(x.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
            groups.append(group.numpy())
            ends.append(end.numpy())
    return tuple(np.concatenate(value) for value in (probabilities, labels, groups, ends))
