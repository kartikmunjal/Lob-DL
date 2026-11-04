"""Part C LightGBM cells (B2 on raw and both representations).

``select`` runs the Study 1 12-configuration grid on validation day 7 only; ``final``
refits the frozen selection with one final seed and only then builds days 8-10.

    python scripts/factorial_gbm.py --data DATA --representation raw --phase select --horizon 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from train_baselines import fit_model, fit_with_validation, tree_parameter_count

from lobdl.datasets import fi_economic_arrays
from lobdl.factorial import (
    HORIZONS,
    NEW_B2_REPRESENTATIONS,
    STATUS,
    family,
    lock_file,
    selection_file,
    verify_lock,
)
from lobdl.metrics import classification_metrics
from lobdl.representations import flat_matrix
from lobdl.search import (
    FINAL_SEEDS,
    SELECTION_SEEDS,
    candidate_grid,
    load_progress,
    save_progress,
    select_candidate,
    summarize_candidate,
)


def run_selection(args) -> None:
    if lock_file(args.output).exists():
        raise RuntimeError("Part C selections are frozen; refusing to alter validation searches")
    path = selection_file(args.output, "B2", args.representation, args.horizon)
    if path.exists():
        print(f"selection already complete: {path}")
        return
    train_x, train_y, _ = flat_matrix(
        args.data, list(range(1, 7)), args.horizon, args.representation, cache_dir=args.cache
    )
    validation_x, validation_y, _ = flat_matrix(
        args.data, [7], args.horizon, args.representation, purge=200, cache_dir=args.cache
    )
    progress_path = (
        args.output
        / "factorial_selection_runs"
        / family("B2", args.representation)
        / f"h{args.horizon}.json"
    )
    progress = load_progress(progress_path, "B2", args.horizon)
    candidates = []
    for trial, parameters in enumerate(candidate_grid("B2"), start=1):
        scores, iterations, counts = [], [], []
        for seed in SELECTION_SEEDS:
            key = f"trial{trial:02d}_seed{seed}"
            if key in progress["completed_fits"]:
                done = progress["completed_fits"][key]
                if done["parameters"] != parameters:
                    raise ValueError(f"registered parameters changed for {key}")
            else:
                model = fit_model("B2", parameters, seed)
                fit_with_validation(model, train_x, train_y, validation_x, validation_y)
                done = {
                    "parameters": parameters,
                    "validation_macro_f1": float(
                        f1_score(validation_y, model.predict(validation_x), average="macro")
                    ),
                    "best_iteration": int(model.best_iteration_),
                    "parameter_count": tree_parameter_count(model),
                }
                progress["completed_fits"][key] = done
                save_progress(progress_path, progress)
            scores.append(done["validation_macro_f1"])
            iterations.append(done["best_iteration"])
            counts.append(done["parameter_count"])
        candidates.append(
            summarize_candidate(trial, parameters, scores, max(counts), best_epochs=iterations)
        )
    result = {
        "model": "B2",
        "representation": args.representation,
        "horizon": args.horizon,
        "status": STATUS,
        "selection_N_trials": 12,
        "selection_seeds": list(SELECTION_SEEDS),
        "feature_count": int(train_x.shape[1]),
        "ranking": "mean_validation_macro_f1_exact_tie_rule",
        "selected": select_candidate(candidates),
        "candidates": candidates,
        "test_data_loaded": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


def run_final(args) -> None:
    verify_lock(args.output)
    if args.seed not in FINAL_SEEDS:
        raise SystemExit(f"final seed must be one of {FINAL_SEEDS}")
    run = args.output / family("B2", args.representation) / f"h{args.horizon}" / f"seed{args.seed}"
    if (run / "predictions.npz").exists() and (run / "report.json").exists():
        print(f"already complete: {run}")
        return
    selection = json.loads(
        selection_file(args.output, "B2", args.representation, args.horizon).read_text("utf-8")
    )
    parameters = selection["selected"]["parameters"]
    train_x, train_y, _ = flat_matrix(
        args.data, list(range(1, 7)), args.horizon, args.representation, cache_dir=args.cache
    )
    validation_x, validation_y, _ = flat_matrix(
        args.data, [7], args.horizon, args.representation, purge=200, cache_dir=args.cache
    )
    started = time.perf_counter()
    model = fit_model("B2", parameters, args.seed)
    fit_with_validation(model, train_x, train_y, validation_x, validation_y)
    # Test windows are constructed only after the selected fit is complete.
    test_x, test_y, test_groups = flat_matrix(
        args.data, [8, 9, 10], args.horizon, args.representation, purge=200, cache_dir=args.cache
    )
    probabilities = model.predict_proba(test_x)
    forward_returns, relative_spreads = fi_economic_arrays(
        args.data, [8, 9, 10], args.horizon, purge=200
    )
    if len(forward_returns) != len(test_y):
        raise AssertionError("FI economic arrays are not aligned with test predictions")
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run / "predictions.npz",
        probabilities=probabilities,
        labels=test_y,
        groups=test_groups,
        forward_returns=forward_returns,
        relative_spreads=relative_spreads,
    )
    report = {
        "model": "B2",
        "representation": args.representation,
        "horizon": args.horizon,
        "seed": args.seed,
        "status": STATUS,
        "selection_N_trials": selection["selection_N_trials"],
        "selected": parameters,
        "feature_count": int(train_x.shape[1]),
        "trainable_parameters": tree_parameter_count(model),
        "compute_hours": (time.perf_counter() - started) / 3600.0,
        "component_metrics_not_for_reporting": classification_metrics(test_y, probabilities),
    }
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "selected"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--representation", choices=NEW_B2_REPRESENTATIONS, required=True)
    parser.add_argument("--phase", choices=("select", "final"), required=True)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.phase == "final" and args.seed is None:
        parser.error("final phase requires --seed")
    (run_selection if args.phase == "select" else run_final)(args)


if __name__ == "__main__":
    main()
