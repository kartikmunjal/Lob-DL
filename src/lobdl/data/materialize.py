from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fi2010 import load_matrix, sha256, split_stock_segments


def materialize_days(source: Path, output: Path) -> dict:
    """Materialize day 1 from Train CF1 and days 2-10 from Test CF1-9."""
    train_dir = source / "NoAuction_DecPre_Training"
    test_dir = source / "NoAuction_DecPre_Testing"
    files = [train_dir / "Train_Dst_NoAuction_DecPre_CF_1.txt"] + [
        test_dir / f"Test_Dst_NoAuction_DecPre_CF_{fold}.txt" for fold in range(1, 10)
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("missing FI-2010 fold files:\n" + "\n".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    manifest: list[dict] = []
    counts: dict[str, int] = {}
    for day, path in enumerate(files, start=1):
        matrix = load_matrix(path)
        destination = output / f"day_{day:02d}.npy"
        np.save(destination, matrix, allow_pickle=False)
        segments = split_stock_segments(matrix[:, :40], day)
        manifest.extend(segment.__dict__ for segment in segments)
        checksums[str(path)] = sha256(path)
        checksums[str(destination)] = sha256(destination)
        counts[str(day)] = len(matrix)
    (output / "segments.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {"source_sha256": checksums, "event_counts": counts}
    (output / "materialization.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

