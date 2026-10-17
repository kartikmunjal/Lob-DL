from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.data.fi2010 import sha256
from lobdl.data.lobster import validate_pure_replay

TICKERS = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")


def eligible_window_count(exact_mask: np.ndarray, window: int) -> int:
    if len(exact_mask) < window:
        return 0
    running = np.convolve(
        exact_mask.astype(np.int16), np.ones(window, dtype=np.int16), "valid"
    )
    return int((running == window).sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preregistered pure replay for all samples"
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--time-bin-minutes", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    combined = {
        "protocol": "pure_replay_from_first_vendor_snapshot_no_reseeding",
        "primary_t1_input": "vendor_book",
        "reconstructed_sensitivity_rule": "all_top10_cells_exact_for_entire_window",
        "window": args.window,
        "tickers": {},
    }
    for ticker in TICKERS:
        messages = next(args.raw.glob(f"{ticker}_*_message_10.csv"))
        vendor = next(args.raw.glob(f"{ticker}_*_orderbook_10.csv"))
        report, exact_mask = validate_pure_replay(
            messages,
            vendor,
            levels=args.levels,
            time_bin_minutes=args.time_bin_minutes,
        )
        report.update(
            {
                "ticker": ticker,
                "message_sha256": sha256(messages),
                "vendor_book_sha256": sha256(vendor),
                "exact_mask_rows_including_initialization": len(exact_mask),
                "eligible_exact_windows": eligible_window_count(exact_mask, args.window),
            }
        )
        (args.output / f"lobster_{ticker}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        np.save(args.output / f"lobster_{ticker}_exact_top10.npy", exact_mask)
        combined["tickers"][ticker] = report
        print(
            ticker,
            json.dumps(
                {
                    "replayed_messages": report["replayed_messages"],
                    "exact_match_rates": report["exact_match_rates"],
                    "eligible_exact_windows": report["eligible_exact_windows"],
                }
            ),
            flush=True,
        )
    (args.output / "lobster_all.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
