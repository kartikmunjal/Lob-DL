from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.datasets import fi_state_arrays
from lobdl.metrics import moving_block_indices
from lobdl.search import FINAL_SEEDS

COMPARISONS = (("B2", "B1"), ("M1", "B2"), ("M2", "M1"))
STOCHASTIC = {"B2", "M1", "M2"}
CLASS_NAMES = ("down", "stationary", "up")


def prediction_path(root: Path, model: str, horizon: int, seed: int) -> Path:
    run = f"seed{seed}" if model in STOCHASTIC else "deterministic"
    return root / model / f"h{horizon}" / run / "predictions.npz"


def paired_correctness(
    artifacts: Path, model_a: str, model_b: str, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences, labels_all, groups_all = [], [], []
    for seed_index, seed in enumerate(FINAL_SEEDS, start=1):
        a = np.load(prediction_path(artifacts, model_a, horizon, seed))
        b = np.load(prediction_path(artifacts, model_b, horizon, seed))
        if not np.array_equal(a["labels"], b["labels"]):
            raise AssertionError("mechanism comparison labels are not aligned")
        if not np.array_equal(a["groups"], b["groups"]):
            raise AssertionError("mechanism comparison groups are not aligned")
        labels = a["labels"]
        differences.append(
            (a["probabilities"].argmax(1) == labels).astype(np.int8)
            - (b["probabilities"].argmax(1) == labels).astype(np.int8)
        )
        labels_all.append(labels)
        groups_all.append(
            np.column_stack(
                (a["groups"], np.full(len(labels), seed_index, dtype=np.int16))
            )
        )
    return np.concatenate(differences), np.concatenate(labels_all), np.concatenate(groups_all)


def interval_for_mask(
    difference: np.ndarray,
    mask: np.ndarray,
    sampled_indices: list[np.ndarray],
) -> dict:
    if not mask.any():
        return {"windows": 0, "mean_correctness_difference": None, "ci95": None}
    values = []
    for index in sampled_indices:
        selected = index[mask[index]]
        values.append(float(difference[selected].mean()) if len(selected) else np.nan)
    return {
        "windows": int(mask.sum()),
        "mean_correctness_difference": float(difference[mask].mean()),
        "ci95": {
            "lower": float(np.nanquantile(values, 0.025)),
            "upper": float(np.nanquantile(values, 0.975)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Registered FI mechanical explanation strata")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/mechanisms"))
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    for horizon in (10, 20, 50, 100):
        train_states = fi_state_arrays(args.data, list(range(1, 7)), horizon)
        test_states = fi_state_arrays(args.data, [8, 9, 10], horizon, purge=200)
        thresholds = {
            name: np.quantile(values, (0.25, 0.5, 0.75)).tolist()
            for name, values in train_states.items()
        }
        for model_a, model_b in COMPARISONS:
            difference, labels, groups = paired_correctness(
                args.artifacts, model_a, model_b, horizon
            )
            repeats = len(FINAL_SEEDS)
            states = {name: np.tile(values, repeats) for name, values in test_states.items()}
            if any(len(values) != len(labels) for values in states.values()):
                raise AssertionError("mechanism states are not aligned with predictions")
            # Reset to the registered primary-analysis seed so each fixed comparison
            # at a horizon uses identical bootstrap blocks.
            rng = np.random.default_rng(20261013)
            sampled = [moving_block_indices(groups, 100, rng) for _ in range(args.replicates)]
            rows = []
            for name, values in states.items():
                bins = np.digitize(values, thresholds[name], right=True)
                for bin_index in range(4):
                    state_mask = bins == bin_index
                    row = {
                        "state": name,
                        "bin": bin_index + 1,
                        "train_quantile_boundaries": thresholds[name],
                        "true_class": "all",
                        **interval_for_mask(difference, state_mask, sampled),
                    }
                    rows.append(row)
                    for class_index, class_name in enumerate(CLASS_NAMES):
                        rows.append(
                            {
                                "state": name,
                                "bin": bin_index + 1,
                                "train_quantile_boundaries": thresholds[name],
                                "true_class": class_name,
                                **interval_for_mask(
                                    difference,
                                    state_mask & (labels == class_index),
                                    sampled,
                                ),
                            }
                        )
            result = {
                "system_a": model_a,
                "system_b": model_b,
                "difference_order": "system_a_minus_system_b",
                "system_a_N_trials": {"selection_N_trials": 12, "reported_variant_N_trials": 0},
                "system_b_N_trials": {"selection_N_trials": 12, "reported_variant_N_trials": 0},
                "horizon": horizon,
                "outcome": "paired_per_window_correctness_difference",
                "bin_fit_scope": "FI_days_1_to_6",
                "evaluation_scope": "FI_days_8_to_10",
                "bootstrap": {
                    "type": "paired_moving_block_within_stock_day_and_seed",
                    "block_length": 100,
                    "replicates": args.replicates,
                },
                "descriptive_not_primary_performance_metric": True,
                "rows": rows,
            }
            path = args.output / f"h{horizon}" / f"{model_a}_minus_{model_b}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
