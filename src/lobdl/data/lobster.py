from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
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
    """Order-ID book for streams that genuinely begin from empty."""

    def __init__(self) -> None:
        self.orders: dict[int, tuple[int, int, int]] = {}
        self.bid: dict[int, int] = {}
        self.ask: dict[int, int] = {}

    def _levels(self, direction: int) -> dict[int, int]:
        return self.bid if direction == 1 else self.ask

    def _add_level(self, direction: int, price: int, size: int) -> None:
        levels = self._levels(direction)
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
            direction, price, remaining = self.orders[message.order_id]
            removed = min(message.size, remaining)
            self._add_level(direction, price, -removed)
            remaining -= removed
            if remaining:
                self.orders[message.order_id] = (direction, price, remaining)
            else:
                del self.orders[message.order_id]
        elif kind == 3:
            direction, price, remaining = self.orders.pop(message.order_id)
            self._add_level(direction, price, -remaining)
        elif kind not in (5, 6, 7):
            raise ValueError(f"unsupported LOBSTER event type {kind}")
        self.assert_invariants()

    def assert_invariants(self) -> None:
        _assert_book_invariants(self.bid, self.ask)

    def snapshot(self, levels: int = 10) -> np.ndarray:
        return _snapshot(self.bid, self.ask, levels)


class SnapshotInitializedBook:
    """Pure aggregate replay initialized once from a vendor snapshot."""

    def __init__(self, snapshot: np.ndarray) -> None:
        if snapshot.ndim != 1 or len(snapshot) % 4:
            raise ValueError("invalid snapshot shape")
        self.bid: dict[int, int] = {}
        self.ask: dict[int, int] = {}
        for offset in range(0, len(snapshot), 4):
            ask_price, ask_size, bid_price, bid_size = map(int, snapshot[offset : offset + 4])
            if ask_size > 0:
                self.ask[ask_price] = ask_size
            if bid_size > 0:
                self.bid[bid_price] = bid_size
        self.orders: dict[int, tuple[int, int, int]] = {}
        self.counts: Counter[str] = Counter()
        _assert_book_invariants(self.bid, self.ask)
        self.first_crossed_or_locked: dict[str, int | float] | None = None

    def _levels(self, direction: int) -> dict[int, int]:
        if direction not in (-1, 1):
            raise ValueError(f"invalid displayed-order direction {direction}")
        return self.bid if direction == 1 else self.ask

    def _subtract(self, direction: int, price: int, size: int) -> bool:
        levels = self._levels(direction)
        if price not in levels:
            self.counts["absent_level_skips"] += 1
            return False
        updated = levels[price] - size
        if updated < 0:
            self.counts["zero_clamps"] += 1
            updated = 0
        if updated == 0:
            del levels[price]
        else:
            levels[price] = updated
        return True

    def apply(self, message: Message) -> None:
        kind = message.event_type
        self.counts["replayed_messages"] += 1
        if kind == 1:
            levels = self._levels(message.direction)
            levels[message.price] = levels.get(message.price, 0) + message.size
            self.orders[message.order_id] = (message.direction, message.price, message.size)
            self.counts["additions"] += 1
        elif kind in (2, 3, 4):
            if message.order_id in self.orders:
                direction, price, remaining = self.orders[message.order_id]
                removed = remaining if kind == 3 else min(message.size, remaining)
                self._subtract(direction, price, removed)
                remaining -= removed
                if remaining > 0:
                    self.orders[message.order_id] = (direction, price, remaining)
                else:
                    del self.orders[message.order_id]
                self.counts["attributed_events"] += 1
            else:
                if self._subtract(message.direction, message.price, message.size):
                    self.counts["unattributed_applied_events"] += 1
        elif kind == 5:
            self.counts["hidden_executions"] += 1
        elif kind == 6:
            self.counts["cross_trades"] += 1
        elif kind == 7:
            self.counts["halt_indicators"] += 1
        else:
            raise ValueError(f"unsupported LOBSTER event type {kind}")
        if self.bid and self.ask and max(self.bid) >= min(self.ask):
            self.counts["crossed_or_locked_states"] += 1
            if self.first_crossed_or_locked is None:
                self.first_crossed_or_locked = {
                    "message_index": self.counts["replayed_messages"] + 1,
                    "seconds_since_midnight": message.time,
                }
        if any(size <= 0 for size in self.bid.values()) or any(
            size <= 0 for size in self.ask.values()
        ):
            raise AssertionError("nonpositive displayed depth after clamp")

    def snapshot(self, levels: int = 10) -> np.ndarray:
        return _snapshot(self.bid, self.ask, levels)


def _assert_book_invariants(bid: dict[int, int], ask: dict[int, int]) -> None:
    if any(size <= 0 for size in bid.values()) or any(size <= 0 for size in ask.values()):
        raise AssertionError("nonpositive displayed depth")
    if bid and ask and max(bid) >= min(ask):
        raise AssertionError("crossed or locked reconstructed book")


def _snapshot(bid: dict[int, int], ask: dict[int, int], levels: int) -> np.ndarray:
    bids = sorted(bid.items(), reverse=True)[:levels]
    asks = sorted(ask.items())[:levels]
    out = np.zeros(levels * 4, dtype=np.int64)
    for level in range(levels):
        if level < len(asks):
            out[4 * level : 4 * level + 2] = asks[level]
        if level < len(bids):
            out[4 * level + 2 : 4 * level + 4] = bids[level]
    return out


def read_messages(path: str | Path) -> Iterator[Message]:
    names = ["time", "type", "order_id", "size", "price", "direction"]
    for row in pd.read_csv(path, header=None, names=names).itertuples(index=False):
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


def validate_pure_replay(
    message_path: str | Path,
    book_path: str | Path,
    levels: int = 10,
    time_bin_minutes: int = 30,
) -> tuple[dict, np.ndarray]:
    """Validate snapshot-initialized replay and return the per-row top-10 mask."""
    names = ["time", "type", "order_id", "size", "price", "direction"]
    messages = pd.read_csv(message_path, header=None, names=names)
    vendor = pd.read_csv(book_path, header=None).to_numpy(dtype=np.int64)
    if len(messages) != len(vendor):
        raise AssertionError(f"row mismatch: {len(messages)} != {len(vendor)}")
    if vendor.shape[1] != levels * 4:
        raise AssertionError(f"expected {levels * 4} book cells, found {vendor.shape[1]}")

    book = SnapshotInitializedBook(vendor[0])
    replay_rows = max(0, len(vendor) - 1)
    exact = np.zeros((replay_rows, 3), dtype=bool)
    size_error = np.zeros((levels, 2), dtype=np.int64)
    bins: dict[int, Counter[str]] = defaultdict(Counter)
    first_mismatch: dict | None = None

    for output_index, row in enumerate(messages.iloc[1:].itertuples(index=False)):
        message_index = output_index + 2
        message = Message(float(row.time), int(row.type), int(row.order_id), int(row.size), int(row.price), int(row.direction))
        book.apply(message)
        reconstructed = book.snapshot(levels)
        observed = vendor[message_index - 1]
        cells_equal = reconstructed == observed
        exact[output_index] = (cells_equal[:4].all(), cells_equal[:20].all(), cells_equal.all())
        for level in range(levels):
            size_error[level, 0] += abs(int(reconstructed[4 * level + 3]) - int(observed[4 * level + 3]))
            size_error[level, 1] += abs(int(reconstructed[4 * level + 1]) - int(observed[4 * level + 1]))
        if first_mismatch is None and not exact[output_index, 2]:
            first_mismatch = {"message_index": message_index, "seconds_since_midnight": message.time}
        bin_index = max(0, int((message.time - 9.5 * 3600) // (time_bin_minutes * 60)))
        counter = bins[bin_index]
        counter["rows"] += 1
        counter["level_1_exact"] += int(exact[output_index, 0])
        counter["top_5_exact"] += int(exact[output_index, 1])
        counter["top_10_exact"] += int(exact[output_index, 2])

    denominator = max(1, replay_rows)
    event_counts = dict(sorted(book.counts.items()))
    categories = (
        "attributed_events", "unattributed_applied_events", "zero_clamps", "absent_level_skips",
        "hidden_executions", "cross_trades", "halt_indicators",
    )
    for key in categories:
        event_counts.setdefault(key, 0)
    time_bins = []
    for bin_index, counter in sorted(bins.items()):
        rows = counter["rows"]
        start = bin_index * time_bin_minutes
        time_bins.append({
            "minutes_since_open_start": start,
            "minutes_since_open_end": start + time_bin_minutes,
            "rows": rows,
            "level_1_exact_rate": counter["level_1_exact"] / rows,
            "top_5_exact_rate": counter["top_5_exact"] / rows,
            "top_10_exact_rate": counter["top_10_exact"] / rows,
        })
    report = {
        "initialization": "first_vendor_snapshot_after_message_1_once",
        "replayed_message_range": "2_through_end",
        "reseeded": False,
        "initialization_row_exact_by_definition": True,
        "replayed_messages": replay_rows,
        "exact_match_rates": {
            "level_1": float(exact[:, 0].mean()) if replay_rows else None,
            "top_5": float(exact[:, 1].mean()) if replay_rows else None,
            "top_10": float(exact[:, 2].mean()) if replay_rows else None,
        },
        "first_top_10_mismatch": first_mismatch,
        "time_bins": time_bins,
        "total_absolute_size_error_by_level": [
            {"level": i + 1, "bid": int(size_error[i, 0]), "ask": int(size_error[i, 1])} for i in range(levels)
        ],
        "event_counts": event_counts,
        "event_fractions_of_replayed_messages": {key: event_counts[key] / denominator for key in categories},
        "invariant_diagnostics": {
            "crossed_or_locked_states": event_counts.get("crossed_or_locked_states", 0),
            "fraction_of_replayed_messages": event_counts.get("crossed_or_locked_states", 0)
            / denominator,
            "first_crossed_or_locked": book.first_crossed_or_locked,
            "repair_or_reseed_applied": False,
        },
    }
    return report, exact[:, 2]
