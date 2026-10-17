from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.audit import assert_no_window_seams, assert_split_days, class_balance
from lobdl.config import load_protocol
from lobdl.data.fi2010 import (
    FI2010Day,
    Segment,
    build_windows,
    decode_labels,
    reconstruct_forward_mean_labels,
    sha256,
)


def load_npy_day(root: Path, day: int) -> FI2010Day:
    matrix_path = root / f"day_{day:02d}.npy"
    matrix = np.load(matrix_path, mmap_mode="r")
    entries = json.loads((root / "segments.json").read_text(encoding="utf-8"))
    segments = tuple(Segment(**item) for item in entries if item["day"] == day)
    return FI2010Day(day, matrix[:, :144], matrix[:, 144:149].astype(np.int8), segments, matrix_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/fi2010.json"))
    args = parser.parse_args()
    protocol = load_protocol(args.config)
    split = protocol["splits"]
    assert_split_days(set(split["train_days"]), set(split["validation_days"]), set(split["test_days"]))
    report: dict = {
        "files": {},
        "splits": {},
        "documented_formula_agreement": {
            "role": "data_integrity_diagnostic_only",
            "changes_primary_shipped_targets": False,
            "by_horizon": {},
        },
    }
    for day in range(1, 11):
        path = args.data / f"day_{day:02d}.npy"
        report["files"][str(day)] = {
            "events": int(np.load(path, mmap_mode="r").shape[0]),
            "sha256": sha256(path),
        }
    for split_name, key in (
        ("train", "train_days"),
        ("validation", "validation_days"),
        ("test", "test_days"),
    ):
        report["splits"][split_name] = {}
        for horizon in protocol["data"]["fi2010"]["primary_horizons_events"]:
            labels = []
            window_count = 0
            for day_number in split[key]:
                day = load_npy_day(args.data, day_number)
                _, y, groups = build_windows(day, horizon)
                assert_no_window_seams(groups, len(y))
                labels.append(y)
                window_count += len(y)
            combined = np.concatenate(labels)
            report["splits"][split_name][str(horizon)] = {
                "windows": window_count,
                "class_balance": class_balance(combined),
            }
    alpha = float(protocol["data"]["fi2010"]["threshold_alpha"])
    for horizon in protocol["data"]["fi2010"]["primary_horizons_events"]:
        agree = total = 0
        by_day = {}
        for day_number in range(1, 11):
            day = load_npy_day(args.data, day_number)
            shipped = decode_labels(day.labels, horizon)
            rebuilt = reconstruct_forward_mean_labels(day.features, horizon, alpha)
            valid = np.zeros(len(shipped), dtype=bool)
            rows = horizon // 10
            for segment in day.segments:
                valid[segment.start : segment.stop - rows] = True
            matches = int((shipped[valid] == rebuilt[valid]).sum())
            count = int(valid.sum())
            agree += matches
            total += count
            by_day[str(day_number)] = {"agree": matches, "n": count, "rate": matches / count}
        report["documented_formula_agreement"]["by_horizon"][str(horizon)] = {
            "agree": agree,
            "n": total,
            "rate": agree / total,
            "by_day": by_day,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
