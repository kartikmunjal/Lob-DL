from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.metrics import bootstrap_metrics, classification_metrics, moving_block_indices
from lobdl.search import FINAL_SEEDS

STOCHASTIC = {"B2", "M1", "M2"}


def run_paths(root: Path, model: str, horizon: int) -> list[Path]:
    base = root / "transfer" / model / f"h{horizon}"
    if model in STOCHASTIC:
        return [base / f"seed{seed}" for seed in FINAL_SEEDS]
    return [base / "deterministic"]


def economics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    forward_returns: np.ndarray,
    mid: np.ndarray,
    spread: np.ndarray,
    fee_bps: float,
) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    correct_direction = (prediction == labels) & np.isin(labels, (0, 2))
    move = np.abs(forward_returns) * mid
    cost = spread + fee_bps * mid / 10000.0
    selected_move = move[correct_direction]
    selected_cost = cost[correct_direction]
    if not len(selected_move):
        return {
            "correct_direction_windows": 0,
            "mean_absolute_forward_move": float("nan"),
            "mean_spread_plus_fee": float("nan"),
            "mean_move_minus_cost": float("nan"),
            "fraction_move_exceeds_cost": float("nan"),
        }
    return {
        "correct_direction_windows": len(selected_move),
        "mean_absolute_forward_move": float(selected_move.mean()),
        "mean_spread_plus_fee": float(selected_cost.mean()),
        "mean_move_minus_cost": float((selected_move - selected_cost).mean()),
        "fraction_move_exceeds_cost": float((selected_move > selected_cost).mean()),
    }


def bootstrap_economics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    forward_returns: np.ndarray,
    mid: np.ndarray,
    spread: np.ndarray,
    fee_bps: float,
    replicates: int,
    seed: int = 20261017,
) -> dict:
    rng = np.random.default_rng(seed)
    names = (
        "mean_absolute_forward_move",
        "mean_spread_plus_fee",
        "mean_move_minus_cost",
        "fraction_move_exceeds_cost",
    )
    samples = {name: [] for name in names}
    for _ in range(replicates):
        index = moving_block_indices(groups, 100, rng)
        result = economics(
            labels[index],
            probabilities[index],
            forward_returns[index],
            mid[index],
            spread[index],
            fee_bps,
        )
        for name in names:
            samples[name].append(result[name])
    return {
        name: {
            "lower": float(np.nanquantile(values, 0.025)),
            "upper": float(np.nanquantile(values, 0.975)),
        }
        for name, values in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Registered T1 block-bootstrap report")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--model",
        choices=("B0_majority", "B0_persistence", "B1", "B2", "M1", "M2"),
        required=True,
    )
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--labels", choices=("fixed", "matched"), required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()

    paths = run_paths(args.artifacts, args.model, args.horizon)
    missing = [str(path) for path in paths if not (path / "predictions.npz").exists()]
    if missing:
        raise FileNotFoundError("missing registered T1 runs:\n" + "\n".join(missing))
    pooled = {
        name: []
        for name in ("probabilities", "labels", "groups", "forward_returns", "mid", "spread")
    }
    seed_scores = []
    for seed_index, path in enumerate(paths, start=1):
        prediction = np.load(path / "predictions.npz")
        labels = prediction[f"labels_{args.labels}"]
        probabilities = prediction["probabilities"]
        seed_scores.append(classification_metrics(labels, probabilities)["macro_f1"])
        pooled["probabilities"].append(probabilities)
        pooled["labels"].append(labels)
        pooled["groups"].append(
            np.column_stack(
                (prediction["groups"], np.full(len(labels), seed_index, dtype=np.int16))
            )
        )
        for name in ("forward_returns", "mid", "spread"):
            pooled[name].append(prediction[name])
    values = {name: np.concatenate(parts) for name, parts in pooled.items()}
    point = classification_metrics(values["labels"], values["probabilities"])
    ci = bootstrap_metrics(
        values["labels"],
        values["probabilities"],
        values["groups"],
        replicates=args.replicates,
    )
    fee_reports = {}
    for fee_bps in (0.0, 0.5, 1.0, 2.0):
        point_economic = economics(
            values["labels"],
            values["probabilities"],
            values["forward_returns"],
            values["mid"],
            values["spread"],
            fee_bps,
        )
        point_economic["ci95"] = bootstrap_economics(
            values["labels"],
            values["probabilities"],
            values["groups"],
            values["forward_returns"],
            values["mid"],
            values["spread"],
            fee_bps,
            args.replicates,
        )
        fee_reports[str(fee_bps)] = point_economic
    result = {
        "model": args.model,
        "horizon": args.horizon,
        "label_variant": args.labels,
        "selection_N_trials": 0 if args.model.startswith("B0_") else 12,
        "reported_variant_N_trials": 1 if args.labels == "matched" else 0,
        "N_seeds": len(paths),
        "seed_macro_f1_sd": (
            float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0
        ),
        "metrics": point,
        "ci95": ci,
        "economics_no_strategy_or_return_claim": {
            "primary_round_trip_fee_bps": 1.0,
            "fee_scenarios_bps": fee_reports,
        },
        "bootstrap": {
            "type": "moving_block_within_ticker_and_seed",
            "block_length": 100,
            "replicates": args.replicates,
        },
    }
    destination = (
        args.artifacts
        / "reports"
        / "transfer"
        / args.model
        / f"h{args.horizon}"
        / f"summary_{args.labels}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
