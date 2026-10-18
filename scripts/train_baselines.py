from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from lobdl.data.fi2010 import HORIZONS, LABEL_TO_INDEX, Segment, iter_window_indices
from lobdl.datasets import load_segments
from lobdl.features import causal_features
from lobdl.metrics import classification_metrics
from lobdl.search import (
    FINAL_SEEDS,
    SELECTION_SEEDS,
    candidate_grid,
    load_progress,
    save_progress,
    select_candidate,
    summarize_candidate,
)
from lobdl.selection_lock import (
    assert_not_frozen,
    completed_selection,
    selection_path,
    verify_lock,
    verify_presearch_timing,
)


def assemble(
    root: Path,
    days: list[int],
    horizon: int,
    purge: int = 0,
    feature_set: str = "primary",
):
    segment_map = load_segments(root)
    xs, ys, groups = [], [], []
    feature_names = None
    label_column = HORIZONS.index(horizon)
    horizon_rows = horizon // 10
    for day in days:
        matrix = np.load(root / f"day_{day:02d}.npy", mmap_mode="r")
        for segment in segment_map[day]:
            raw = np.asarray(matrix[segment.start : segment.stop, :40])
            if feature_set == "primary":
                engineered, names = causal_features(raw)
                feature_names = names
            elif feature_set == "shipped144":
                engineered = np.asarray(
                    matrix[segment.start : segment.stop, :144], dtype=np.float32
                )
                feature_names = [f"shipped_{index:03d}" for index in range(144)]
            else:
                raise ValueError(f"unsupported feature set {feature_set}")
            local_segment = Segment(day, segment.stock, 0, len(raw))
            ends = np.fromiter(
                (
                    end
                    for _, end in iter_window_indices(
                        (local_segment,), 100, horizon_rows, purge
                    )
                ),
                dtype=np.int64,
            )
            shipped = matrix[segment.start + ends, 144 + label_column].astype(np.int8)
            y = np.vectorize(LABEL_TO_INDEX.__getitem__, otypes=[np.int8])(shipped)
            xs.append(engineered[ends])
            ys.append(y)
            groups.append(np.tile((day, segment.stock), (len(ends), 1)))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups), feature_names


def b0_probabilities(
    train_y: np.ndarray, test_x: np.ndarray, feature_names: list[str]
) -> dict[str, np.ndarray]:
    majority = int(np.bincount(train_y, minlength=3).argmax())
    majority_prob = np.zeros((len(test_x), 3))
    majority_prob[:, majority] = 1.0
    last_move = test_x[:, feature_names.index("mid_log_return_1")]
    persistent_class = np.where(last_move > 0, 2, np.where(last_move < 0, 0, 1))
    persistence_prob = np.zeros((len(test_x), 3))
    persistence_prob[np.arange(len(test_x)), persistent_class] = 1.0
    return {"majority": majority_prob, "persistence": persistence_prob}


def fit_model(model_name: str, parameters: dict, seed: int | None = None):
    if model_name == "B1":
        return LogisticRegression(
            C=parameters["C"],
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            n_jobs=-1,
        )
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=2000,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        **parameters,
    )


def fit_with_validation(model, train_x, train_y, validation_x, validation_y) -> None:
    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        model.fit(train_x, train_y)


def tree_parameter_count(model: lgb.LGBMClassifier) -> int:
    """Count fitted terminal values, the free prediction values in the ensemble."""
    trees = model.booster_.dump_model()["tree_info"]
    return int(sum(tree["num_leaves"] for tree in trees))


def run_selection(args) -> None:
    if args.model == "B0":
        raise SystemExit("B0 has no selection phase")
    if args.feature_set != "primary":
        raise SystemExit("shipped144 is a fixed-config final ablation, not a search")
    verify_presearch_timing(args.timing_report)
    assert_not_frozen(args.output)
    existing = completed_selection(args.output, args.model, args.horizon)
    if existing is not None:
        print(json.dumps({**existing, "candidates": "already_complete"}, indent=2))
        return
    train_x, train_y, _, _ = assemble(args.data, list(range(1, 7)), args.horizon)
    validation_x, validation_y, _, _ = assemble(args.data, [7], args.horizon, purge=200)
    if args.model == "B1":
        scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x)
        validation_x = scaler.transform(validation_x)
    candidates = []
    progress_path = (
        args.output / "selection_runs" / args.model / f"h{args.horizon}" / "progress.json"
    )
    progress = load_progress(progress_path, args.model, args.horizon)
    for trial, parameters in enumerate(candidate_grid(args.model), start=1):
        seeds = (None,) if args.model == "B1" else SELECTION_SEEDS
        scores = []
        best_iterations = []
        parameter_counts = []
        for seed in seeds:
            seed_name = "deterministic" if seed is None else str(seed)
            key = f"trial{trial:02d}_seed{seed_name}"
            if key in progress["completed_fits"]:
                completed = progress["completed_fits"][key]
                if completed["parameters"] != parameters:
                    raise ValueError(f"registered parameters changed for {key}")
                scores.append(completed["validation_macro_f1"])
                if completed["best_iteration"] is not None:
                    best_iterations.append(completed["best_iteration"])
                    parameter_counts.append(completed["parameter_count"])
                continue
            model = fit_model(args.model, parameters, seed)
            fit_with_validation(model, train_x, train_y, validation_x, validation_y)
            scores.append(
                float(f1_score(validation_y, model.predict(validation_x), average="macro"))
            )
            if isinstance(model, lgb.LGBMClassifier):
                best_iterations.append(int(model.best_iteration_))
                parameter_counts.append(tree_parameter_count(model))
            progress["completed_fits"][key] = {
                "parameters": parameters,
                "validation_macro_f1": scores[-1],
                "best_iteration": best_iterations[-1] if best_iterations else None,
                "parameter_count": parameter_counts[-1] if parameter_counts else None,
            }
            save_progress(progress_path, progress)
        candidates.append(
            summarize_candidate(
                trial,
                parameters,
                scores,
                parameter_count=(
                    max(parameter_counts)
                    if parameter_counts
                    else train_x.shape[1] * 3 + 3
                ),
                best_epochs=best_iterations,
            )
        )
    selected = select_candidate(candidates)
    result = {
        "model": args.model,
        "horizon": args.horizon,
        "selection_N_trials": 12,
        "selection_seeds": [] if args.model == "B1" else list(SELECTION_SEEDS),
        "ranking": "mean_validation_macro_f1_exact_tie_rule",
        "selected": selected,
        "candidates": candidates,
        "test_data_loaded": False,
    }
    path = selection_path(args.output, args.model, args.horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "candidates"}, indent=2))


def run_final(args) -> None:
    verify_lock(args.output)
    if args.model == "B0" and args.feature_set != "primary":
        raise SystemExit("B0 has no shipped144 feature ablation")
    train_x, train_y, _, feature_names = assemble(
        args.data, list(range(1, 7)), args.horizon, feature_set=args.feature_set
    )
    validation_x, validation_y, _, _ = assemble(
        args.data, [7], args.horizon, purge=200, feature_set=args.feature_set
    )
    started = time.perf_counter()
    if args.model == "B0":
        test_x, test_y, test_groups, _ = assemble(
            args.data,
            [8, 9, 10],
            args.horizon,
            purge=200,
            feature_set=args.feature_set,
        )
        probabilities = b0_probabilities(train_y, test_x, feature_names)[args.rule]
        selected = {"rule": args.rule}
        selection_trials = 0
        run_name = args.rule
        seed = None
    else:
        selection = json.loads(
            selection_path(args.output, args.model, args.horizon).read_text(encoding="utf-8")
        )
        selected = selection["selected"]["parameters"]
        selection_trials = selection["selection_N_trials"]
        scaler = None
        if args.model == "B1":
            scaler = StandardScaler().fit(train_x)
            train_x = scaler.transform(train_x)
            validation_x = scaler.transform(validation_x)
        seed = None if args.model == "B1" else args.seed
        model = fit_model(args.model, selected, seed)
        fit_with_validation(model, train_x, train_y, validation_x, validation_y)
        # Test arrays become reachable only after the selected fit and its
        # validation-driven stopping point are complete.
        test_x, test_y, test_groups, _ = assemble(
            args.data,
            [8, 9, 10],
            args.horizon,
            purge=200,
            feature_set=args.feature_set,
        )
        if scaler is not None:
            test_x = scaler.transform(test_x)
        probabilities = model.predict_proba(test_x)
        run_name = "deterministic" if args.model == "B1" else f"seed{args.seed}"
    family = args.model if args.feature_set == "primary" else f"{args.model}_shipped144"
    run = args.output / family / f"h{args.horizon}" / run_name
    run.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(test_y, probabilities)
    report = {
        "model": args.model,
        "horizon": args.horizon,
        "feature_set": args.feature_set,
        "seed": seed,
        "selection_N_trials": selection_trials,
        "reported_variant_N_trials": 1 if args.feature_set == "shipped144" else 0,
        "selected": selected,
        "compute_hours": (time.perf_counter() - started) / 3600.0,
        "component_metrics_not_for_reporting": metrics,
        "reporting_status": "awaiting_pooled_seed_block_bootstrap",
    }
    np.savez_compressed(
        run / "predictions.npz",
        probabilities=probabilities,
        labels=test_y,
        groups=test_groups,
    )
    (run / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=("B0", "B1", "B2"), required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--phase", choices=("select", "final"), required=True)
    parser.add_argument("--seed", type=int, choices=FINAL_SEEDS)
    parser.add_argument("--rule", choices=("majority", "persistence"), default="majority")
    parser.add_argument(
        "--feature-set", choices=("primary", "shipped144"), default="primary"
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--timing-report", type=Path, default=Path("reports/phase1_timing.json")
    )
    args = parser.parse_args()
    if args.model == "B2" and args.phase == "final" and args.seed is None:
        parser.error("B2 final phase requires --seed")
    if args.phase == "select":
        run_selection(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
