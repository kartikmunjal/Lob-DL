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


def assemble(root: Path, days: list[int], horizon: int, purge: int = 0):
    segment_map = load_segments(root)
    xs, ys, groups = [], [], []
    feature_names = None
    label_column = HORIZONS.index(horizon)
    horizon_rows = horizon // 10
    for day in days:
        matrix = np.load(root / f"day_{day:02d}.npy", mmap_mode="r")
        for segment in segment_map[day]:
            raw = np.asarray(matrix[segment.start : segment.stop, :40])
            engineered, names = causal_features(raw)
            feature_names = names
            local_segment = Segment(day, segment.stock, 0, len(raw))
            ends = np.fromiter((end for _, end in iter_window_indices((local_segment,), 100, horizon_rows, purge)), dtype=np.int64)
            shipped = matrix[segment.start + ends, 144 + label_column].astype(np.int8)
            y = np.vectorize(LABEL_TO_INDEX.__getitem__, otypes=[np.int8])(shipped)
            xs.append(engineered[ends])
            ys.append(y)
            groups.append(np.tile((day, segment.stock), (len(ends), 1)))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups), feature_names


def b0_probabilities(train_y: np.ndarray, test_x: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    majority = int(np.bincount(train_y, minlength=3).argmax())
    majority_prob = np.zeros((len(test_x), 3)); majority_prob[:, majority] = 1.0
    last_move = test_x[:, feature_names.index("mid_log_return_1")]
    persistent_class = np.where(last_move > 0, 2, np.where(last_move < 0, 0, 1))
    persistence_prob = np.zeros((len(test_x), 3))
    persistence_prob[np.arange(len(test_x)), persistent_class] = 1.0
    return {"majority": majority_prob, "persistence": persistence_prob}


def model_trials(name: str):
    if name == "B1":
        return [{"C": float(value)} for value in np.logspace(-4, 2, 12)]
    leaves = (31, 47, 63)
    min_leaf = (200, 500)
    estimators = (300, 700)
    return [{"num_leaves": leaf, "min_child_samples": minimum, "n_estimators": trees} for leaf in leaves for minimum in min_leaf for trees in estimators]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=("B0", "B1", "B2"), required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rule", choices=("majority", "persistence"), default="majority")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train_x, train_y, _, feature_names = assemble(args.data, list(range(1, 7)), args.horizon)
    validation_x, validation_y, _, _ = assemble(args.data, [7], args.horizon, purge=200)
    test_x, test_y, test_groups, _ = assemble(args.data, [8, 9, 10], args.horizon, purge=200)
    run_name = args.rule if args.model == "B0" else ("deterministic" if args.model != "B2" else f"seed{args.seed}")
    run = args.output / args.model / f"h{args.horizon}" / run_name
    run.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trial_log = []
    if args.model == "B0":
        probabilities = b0_probabilities(train_y, test_x, feature_names)[args.rule]
        selected = {"rule": args.rule}
        n_trials = 1
    else:
        scaler = StandardScaler().fit(train_x)
        train_scaled, validation_scaled, test_scaled = (scaler.transform(value) for value in (train_x, validation_x, test_x))
        best_score, best_model, selected = -np.inf, None, None
        for trial, parameters in enumerate(model_trials(args.model), start=1):
            if args.model == "B1":
                model = LogisticRegression(C=parameters["C"], penalty="l2", solver="lbfgs", max_iter=300, n_jobs=-1)
            else:
                model = lgb.LGBMClassifier(objective="multiclass", learning_rate=0.05, verbosity=-1, random_state=args.seed, **parameters)
            model.fit(train_scaled, train_y)
            score = float(f1_score(validation_y, model.predict(validation_scaled), average="macro"))
            trial_log.append({"trial": trial, "parameters": parameters, "validation_macro_f1": score})
            if score > best_score:
                best_score, best_model, selected = score, model, parameters
        probabilities = best_model.predict_proba(test_scaled)
        n_trials = len(trial_log)
    elapsed = (time.perf_counter() - started) / 3600.0
    metrics = classification_metrics(test_y, probabilities)
    report = {"model": args.model, "horizon": args.horizon, "seed": None if args.model != "B2" else args.seed, "N_trials": n_trials, "selected": selected, "compute_hours": elapsed, "component_metrics_not_for_reporting": metrics, "reporting_status": "awaiting_pooled_seed_block_bootstrap"}
    np.savez_compressed(run / "predictions.npz", probabilities=probabilities, labels=test_y, groups=test_groups)
    (run / "trials.json").write_text(json.dumps(trial_log, indent=2) + "\n", encoding="utf-8")
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
