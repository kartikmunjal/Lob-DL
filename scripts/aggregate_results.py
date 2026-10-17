from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.metrics import bootstrap_metrics, classification_metrics

SEEDS = (11, 23, 47, 91, 137)
M1_ANCHOR = {"learning_rate": 0.01, "epsilon": 1.0, "dropout": 0.0}


def locate_runs(root: Path, model: str, horizon: int, variant: str | None):
    base = root / model / f"h{horizon}"
    if model == "M1_anchor" and not base.exists():
        selection = json.loads(
            (root / "selections" / "M1" / f"h{horizon}.json").read_text(
                encoding="utf-8"
            )
        )
        if selection["selected"]["parameters"] != M1_ANCHOR:
            raise FileNotFoundError("M1 anchor runs are absent and the anchor did not win")
        base = root / "M1" / f"h{horizon}"
    if model in ("M1", "M1_anchor", "M2", "B2", "B2_shipped144"):
        return [base / f"seed{seed}" for seed in SEEDS]
    return [base / (variant or "deterministic")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--model",
        choices=(
            "B0",
            "B1",
            "B1_shipped144",
            "B2",
            "B2_shipped144",
            "M1",
            "M1_anchor",
            "M2",
        ),
        required=True,
    )
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--variant", choices=("majority", "persistence"))
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    runs = locate_runs(args.artifacts, args.model, args.horizon, args.variant)
    missing = [str(run) for run in runs if not (run / "predictions.npz").exists()]
    if missing:
        raise FileNotFoundError("missing registered runs:\n" + "\n".join(missing))
    pooled_probabilities, pooled_labels, pooled_groups, seed_scores = [], [], [], []
    reports = []
    for index, run in enumerate(runs):
        prediction = np.load(run / "predictions.npz")
        probabilities, labels, groups = prediction["probabilities"], prediction["labels"], prediction["groups"]
        if pooled_labels and not np.array_equal(labels, pooled_labels[0]):
            raise AssertionError("seed prediction files are not aligned")
        metrics = classification_metrics(labels, probabilities)
        seed_scores.append(metrics["macro_f1"])
        pooled_probabilities.append(probabilities)
        pooled_labels.append(labels)
        pooled_groups.append(np.column_stack((groups, np.full(len(groups), index + 1))))
        reports.append(json.loads((run / "report.json").read_text(encoding="utf-8")))
    probabilities = np.concatenate(pooled_probabilities)
    labels = np.concatenate(pooled_labels)
    groups = np.concatenate(pooled_groups)
    point = classification_metrics(labels, probabilities)
    ci = bootstrap_metrics(labels, probabilities, groups, replicates=args.replicates)
    result = {
        "model": args.model, "variant": args.variant, "horizon": args.horizon,
        "selection_N_trials": int(reports[0].get("selection_N_trials", 0)),
        "reported_variant_N_trials": (
            1
            if args.model == "M1_anchor"
            else int(reports[0].get("reported_variant_N_trials", 0))
        ),
        "N_seeds": len(runs), "seed_macro_f1_sd": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
        "metrics": point, "ci95": ci,
        "compute_hours": float(sum(report.get("gpu_hours", report.get("compute_hours", 0.0)) for report in reports)),
        "reportable": True,
    }
    destination = args.artifacts / "reports" / args.model / f"h{args.horizon}"
    destination.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.variant}" if args.variant else ""
    path = destination / f"summary{suffix}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
