"""Part C analysis: the registered representation x model contrasts.

Reads Study 1 predictions (read-only) for the three reused cells and Part C predictions
for the six new cells. Writes cell metrics, the 8 primary interaction tests and the 72
secondary comparisons. Contrasts whose cells are missing are withheld and recorded;
Holm correction always uses the registered family sizes (8 and 72).

    python scripts/analyze_factorial.py --study1 S --partc P --output reports/factorial
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lobdl.factorial import HORIZONS, REPRESENTATIONS, STATUS
from lobdl.inference import (
    cluster_bootstrap_interval,
    cluster_metric_by_seed,
    holm_adjust,
    sign_flip_pvalue,
)
from lobdl.metrics import classification_metrics
from lobdl.search import FINAL_SEEDS

MODELS = ("B2", "M1", "M2")
STUDY1_DIR = {("B2", "eng"): "B2", ("M1", "raw"): "M1", ("M2", "raw"): "M2"}
PRIMARY_FAMILY, SECONDARY_FAMILY = 8, 72


def cell_label(model: str, representation: str) -> str:
    if (model, representation) == ("M1", "eng"):
        return "M1 eng (LSTM-64 head, not DeepLOB)"
    return f"{model} {representation}"


def prediction_path(study1: Path, partc: Path, model, representation, horizon, seed) -> Path:
    if (model, representation) in STUDY1_DIR:
        base = study1 / STUDY1_DIR[(model, representation)]
    else:
        base = partc / f"F_{model}_{representation}"
    return base / f"h{horizon}" / f"seed{seed}" / "predictions.npz"


def disk_loader(study1: Path, partc: Path):
    def load(model, representation, horizon, seed):
        path = prediction_path(study1, partc, model, representation, horizon, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        return data["labels"], data["probabilities"], np.asarray(data["groups"])

    return load


def load_cell(loader, model, representation, horizon):
    runs = [loader(model, representation, horizon, seed) for seed in FINAL_SEEDS]
    return [r[0] for r in runs], [r[1] for r in runs], [r[2] for r in runs]


def cell_metrics(cell) -> dict:
    ys, probabilities, _ = cell
    per_seed = [classification_metrics(y, p) for y, p in zip(ys, probabilities, strict=True)]
    macro = np.array([m["macro_f1"] for m in per_seed])
    mcc = np.array([m["mcc"] for m in per_seed])
    return {
        "macro_f1": float(macro.mean()),
        "macro_f1_seed_sd": float(macro.std()),
        "mcc": float(mcc.mean()),
        "N_seeds": len(per_seed),
    }


def contrast_definitions(horizon: int) -> list[dict]:
    """Registered contrasts as signed combinations of (model, representation) cells."""
    out = []
    out.append({
        "family": "primary", "name": "H1 M2 interaction: gap_raw - gap_eng",
        "terms": {("B2", "raw"): 1, ("M2", "raw"): -1, ("B2", "eng"): -1, ("M2", "eng"): 1},
    })
    out.append({
        "family": "primary", "name": "H2 M1 interaction: gap_raw - gap_both",
        "terms": {("B2", "raw"): 1, ("M1", "raw"): -1, ("B2", "both"): -1, ("M1", "both"): 1},
    })
    for model in MODELS:
        for a, b in (("both", "raw"), ("eng", "raw"), ("both", "eng")):
            out.append({
                "family": "secondary",
                "name": f"representation: {cell_label(model, a)} - {cell_label(model, b)}",
                "terms": {(model, a): 1, (model, b): -1},
            })
    for representation in REPRESENTATIONS:
        for a, b in (("B2", "M1"), ("B2", "M2"), ("M1", "M2")):
            out.append({
                "family": "secondary",
                "name": f"model: {cell_label(a, representation)} - {cell_label(b, representation)}",
                "terms": {(a, representation): 1, (b, representation): -1},
            })
    for item in out:
        item["horizon"] = horizon
    return out


def analyze(loader, horizons=HORIZONS, replicates: int = 10_000) -> dict:
    cells_table, contrasts = [], []
    for horizon in horizons:
        vectors, missing = {}, set()
        for model in MODELS:
            for representation in REPRESENTATIONS:
                try:
                    cell = load_cell(loader, model, representation, horizon)
                except FileNotFoundError:
                    missing.add((model, representation))
                    continue
                ids, vector = cluster_metric_by_seed(*cell)
                vectors[(model, representation)] = (ids, vector)
                cells_table.append({
                    "status": STATUS, "model": cell_label(model, representation),
                    "representation": representation, "horizon": horizon,
                    **cell_metrics(cell),
                })
        for definition in contrast_definitions(horizon):
            needed = set(definition["terms"])
            row = {
                "status": STATUS, "family": definition["family"], "horizon": horizon,
                "contrast": definition["name"],
            }
            if needed & missing:
                row.update({
                    "withheld": "missing cells: "
                    + ", ".join(cell_label(*c) for c in sorted(needed & missing))
                })
            else:
                ids = [vectors[c][0] for c in needed]
                if any(not np.array_equal(ids[0], other) for other in ids[1:]):
                    raise AssertionError("cells do not share identical test clusters")
                vector = sum(w * vectors[c][1] for c, w in definition["terms"].items())
                interval = cluster_bootstrap_interval(vector, replicates=replicates)
                test = sign_flip_pvalue(vector)
                row.update({
                    "clusters": interval["clusters"],
                    "mean_cluster_difference": interval["mean"],
                    "ci95_lower": interval["ci_lower"], "ci95_upper": interval["ci_upper"],
                    "clusters_positive": interval["clusters_favoring_a"],
                    "sign_flip_p": test["p_value"],
                })
            contrasts.append(row)
    for family, size in (("primary", PRIMARY_FAMILY), ("secondary", SECONDARY_FAMILY)):
        members = [r for r in contrasts if r["family"] == family]
        if len(horizons) == len(HORIZONS) and len(members) != size:
            raise AssertionError(f"{family} family has {len(members)} contrasts, expected {size}")
        # Withheld contrasts count as p = 1 so the correction keeps the registered size.
        raw_p = np.array([r.get("sign_flip_p", 1.0) for r in members])
        adjusted = holm_adjust(raw_p)
        for row, value in zip(members, adjusted, strict=True):
            if "withheld" not in row:
                row["holm_p"] = float(value)
                row["holm_family_size"] = len(members)
    return {"status": STATUS, "cells": cells_table, "contrasts": contrasts}


def write_outputs(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, name in (("cells", "cells.csv"), ("contrasts", "contrasts.csv")):
        rows = result[key]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study1", type=Path, required=True)
    parser.add_argument("--partc", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/factorial"))
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(disk_loader(args.study1, args.partc), replicates=args.replicates)
    write_outputs(result, args.output)
    done = sum("withheld" not in r for r in result["contrasts"])
    print(f"cells: {len(result['cells'])}/36, contrasts computed: {done}/{len(result['contrasts'])}")


if __name__ == "__main__":
    main()
