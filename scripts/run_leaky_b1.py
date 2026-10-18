from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from train_baselines import assemble, fit_model

from lobdl.metrics import classification_metrics
from lobdl.selection_lock import selection_path, verify_lock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deliberately invalid full-dataset-normalization B1 diagnostic"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    verify_lock(args.artifacts)
    selection = json.loads(
        selection_path(args.artifacts, "B1", args.horizon).read_text(encoding="utf-8")
    )
    primary = args.artifacts / "B1" / f"h{args.horizon}" / "deterministic" / "report.json"
    if not primary.exists():
        raise FileNotFoundError("primary B1 final result must exist before the leaky diagnostic")

    train_x, train_y, _, _ = assemble(args.data, list(range(1, 7)), args.horizon)
    validation_x, _, _, _ = assemble(args.data, [7], args.horizon, purge=200)
    test_x, test_y, test_groups, _ = assemble(
        args.data, [8, 9, 10], args.horizon, purge=200
    )
    started = time.perf_counter()
    scaler = StandardScaler().fit(np.concatenate((train_x, validation_x, test_x)))
    model = fit_model("B1", selection["selected"]["parameters"])
    model.fit(scaler.transform(train_x), train_y)
    probabilities = model.predict_proba(scaler.transform(test_x))
    run = args.artifacts / "B1_leaky" / f"h{args.horizon}" / "deterministic"
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run / "predictions.npz",
        probabilities=probabilities,
        labels=test_y,
        groups=test_groups,
    )
    report = {
        "model": "B1",
        "variant": "full_dataset_normalization_invalid_diagnostic",
        "horizon": args.horizon,
        "selection_N_trials": 12,
        "reported_variant_N_trials": 1,
        "selected": selection["selected"]["parameters"],
        "normalization_fit_scope": "train_validation_test",
        "valid_for_model_claims": False,
        "compute_hours": (time.perf_counter() - started) / 3600.0,
        "component_metrics_not_for_reporting": classification_metrics(
            test_y, probabilities
        ),
        "reporting_status": "awaiting_block_bootstrap",
    }
    (run / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
