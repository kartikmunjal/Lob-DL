from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Message:
    time: float
    event_type: int
    order_id: int
    size: int
    price: int
    direction: int


class OrderBook:
    """LOBSTER message-driven book with executable invariants."""

    def __init__(self) -> None:
        self.orders: dict[int, tuple[int, int, int]] = {}
        self.bid: dict[int, int] = {}
        self.ask: dict[int, int] = {}

    def _levels(self, side: int) -> dict[int, int]:
        return self.bid if side == 1 else self.ask

    def _add_level(self, side: int, price: int, size: int) -> None:
        levels = self._levels(side)
        levels[price] = levels.get(price, 0) + size
        if levels[price] < 0:
            raise AssertionError("negative aggregate depth")
        if levels[price] == 0:
            del levels[price]

    def apply(self, message: Message) -> None:
        kind = message.event_type
        if kind == 1:
            if message.order_id in self.orders:
                raise AssertionError("duplicate order id")
            self.orders[message.order_id] = (message.direction, message.price, message.size)
            self._add_level(message.direction, message.price, message.size)
        elif kind in (2, 4):
            side, price, remaining = self.orders[message.order_id]
            removed = min(message.size, remaining)
            self._add_level(side, price, -removed)
            remaining -= removed
            if remaining:
                self.orders[message.order_id] = (side, price, remaining)
            else:
                del self.orders[message.order_id]
        elif kind == 3:
            side, price, remaining = self.orders.pop(message.order_id)
            self._add_level(side, price, -remaining)
        elif kind in (5, 6, 7):
            # Hidden executions, cross trades, and trading-halt indicators do not
            # alter displayed orders. LOBSTER repeats the previous snapshot.
            pass
        else:
            raise ValueError(f"unsupported LOBSTER event type {kind}")
        self.assert_invariants()

    def assert_invariants(self) -> None:
        if any(size <= 0 for size in self.bid.values()) or any(size <= 0 for size in self.ask.values()):
            raise AssertionError("nonpositive displayed depth")
        if self.bid and self.ask and max(self.bid) >= min(self.ask):
            raise AssertionError("crossed or locked reconstructed book")

    def snapshot(self, levels: int = 10) -> np.ndarray:
        bids = sorted(self.bid.items(), reverse=True)[:levels]
        asks = sorted(self.ask.items())[:levels]
        out = np.zeros(levels * 4, dtype=np.int64)
        for level in range(levels):
            if level < len(asks):
                out[4 * level : 4 * level + 2] = asks[level]
            if level < len(bids):
                out[4 * level + 2 : 4 * level + 4] = bids[level]
        return out


def read_messages(path: str | Path):
    frame = pd.read_csv(path, header=None, names=["time", "type", "order_id", "size", "price", "direction"])
    for row in frame.itertuples(index=False):
        yield Message(float(row.time), int(row.type), int(row.order_id), int(row.size), int(row.price), int(row.direction))


def reconstruct(message_path: str | Path, levels: int = 10) -> np.ndarray:
    book = OrderBook()
    snapshots = []
    for message in read_messages(message_path):
        book.apply(message)
        snapshots.append(book.snapshot(levels))
    return np.stack(snapshots)


def compare_snapshots(reconstructed: np.ndarray, book_path: str | Path) -> dict:
    published = pd.read_csv(book_path, header=None).to_numpy(dtype=np.int64)
    if reconstructed.shape != published.shape:
        raise AssertionError(f"shape mismatch: {reconstructed.shape} != {published.shape}")
    differences = np.count_nonzero(reconstructed != published)
    return {"rows": len(published), "cells": published.size, "different_cells": int(differences), "exact": differences == 0}
