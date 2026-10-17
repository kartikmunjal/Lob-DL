from __future__ import annotations

import numpy as np


def _book(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.ndim != 2 or x.shape[1] < 40:
        raise ValueError("expected (events, >=40) FI-2010 matrix")
    ask_p = x[:, 0:40:4].astype(np.float64)
    ask_v = x[:, 1:40:4].astype(np.float64)
    bid_p = x[:, 2:40:4].astype(np.float64)
    bid_v = x[:, 3:40:4].astype(np.float64)
    return ask_p, ask_v, bid_p, bid_v


def _lag(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    out[lag:] = values[lag:] - values[:-lag]
    return out


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    sums = np.cumsum(np.concatenate(([0.0], values)))
    out = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        out[index] = sums[index + 1] - sums[start]
    return out


def causal_features(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Locked handcrafted feature family; every column is measurable at time t."""
    ask_p, ask_v, bid_p, bid_v = _book(raw)
    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2.0
    spread = ask_p[:, 0] - bid_p[:, 0]
    safe_mid = np.maximum(np.abs(mid), 1e-12)
    columns: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        columns.append(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    add("relative_spread", spread / safe_mid)
    for level in (5, 10):
        index = level - 1
        add(f"ask_distance_l{level}", (ask_p[:, index] - ask_p[:, 0]) / safe_mid)
        add(f"bid_distance_l{level}", (bid_p[:, 0] - bid_p[:, index]) / safe_mid)
    for depth in (1, 3, 5, 10):
        bid_total = bid_v[:, :depth].sum(axis=1)
        ask_total = ask_v[:, :depth].sum(axis=1)
        add(f"volume_imbalance_{depth}", (bid_total - ask_total) / np.maximum(bid_total + ask_total, 1e-12))
    for depth in (5, 10):
        add(f"log_bid_depth_{depth}", np.log1p(bid_v[:, :depth].sum(axis=1)))
        add(f"log_ask_depth_{depth}", np.log1p(ask_v[:, :depth].sum(axis=1)))
    microprice = (ask_p[:, 0] * bid_v[:, 0] + bid_p[:, 0] * ask_v[:, 0]) / np.maximum(
        bid_v[:, 0] + ask_v[:, 0], 1e-12
    )
    add("microprice_minus_mid", (microprice - mid) / safe_mid)
    ask_slope = (ask_p[:, -1] - ask_p[:, 0]) / np.maximum(ask_v.sum(axis=1), 1e-12)
    bid_slope = (bid_p[:, 0] - bid_p[:, -1]) / np.maximum(bid_v.sum(axis=1), 1e-12)
    add("book_slope", ask_slope - bid_slope)
    log_mid = np.log(np.maximum(mid, 1e-12))
    for lag in (1, 2, 5, 10, 20):
        add(f"mid_log_return_{lag}", _lag(log_mid, lag))
    one_step = _lag(log_mid, 1)
    for window in (10, 50):
        variance = _rolling_sum(one_step**2, window) / np.minimum(np.arange(len(raw)) + 1, window)
        add(f"realized_volatility_{window}", np.sqrt(np.maximum(variance, 0.0)))
    top_imbalance = (bid_v[:, 0] - ask_v[:, 0]) / np.maximum(bid_v[:, 0] + ask_v[:, 0], 1e-12)
    for lag in (1, 5):
        add(f"imbalance_delta_{lag}", _lag(top_imbalance, lag))
        add(f"spread_delta_{lag}", _lag(spread / safe_mid, lag))
    bid_price_delta = _lag(bid_p[:, 0], 1)
    ask_price_delta = _lag(ask_p[:, 0], 1)
    bid_flow = np.where(bid_price_delta > 0, bid_v[:, 0], np.where(bid_price_delta < 0, -np.roll(bid_v[:, 0], 1), _lag(bid_v[:, 0], 1)))
    ask_flow = np.where(ask_price_delta < 0, ask_v[:, 0], np.where(ask_price_delta > 0, -np.roll(ask_v[:, 0], 1), _lag(ask_v[:, 0], 1)))
    ofi = bid_flow - ask_flow
    ofi[0] = 0.0
    for window in (1, 5, 10):
        add(f"top_ofi_{window}", _rolling_sum(ofi, window))
    return np.column_stack(columns).astype(np.float32), names
