from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.metrics import bootstrap_metrics, classification_metrics, moving_block_indices

SEEDS = (11, 23, 47, 91, 137)
M1_ANCHOR = {"learning_rate": 0.01, "epsilon": 1.0, "dropout": 0.0}


def fi_economics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    forward_returns: np.ndarray,
    relative_spreads: np.ndarray,
    fee_bps: float,
) -> dict:
    prediction = probabilities.argmax(axis=1)
    selected = (prediction == labels) & np.isin(labels, (0, 2))
    moves = np.abs(forward_returns[selected]) * 10000.0
    costs = relative_spreads[selected] * 10000.0 + fee_bps
    if not len(moves):
        return {
            "correct_direction_windows": 0,
            "mean_absolute_forward_move_bps": float("nan"),
            "mean_spread_plus_fee_bps": float("nan"),
            "mean_move_minus_cost_bps": float("nan"),
            "fraction_move_exceeds_cost": float("nan"),
        }
    return {
        "correct_direction_windows": len(moves),
        "mean_absolute_forward_move_bps": float(moves.mean()),
        "mean_spread_plus_fee_bps": float(costs.mean()),
        "mean_move_minus_cost_bps": float((moves - costs).mean()),
        "fraction_move_exceeds_cost": float((moves > costs).mean()),
    }


def bootstrap_fi_economics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    forward_returns: np.ndarray,
    relative_spreads: np.ndarray,
    fees: tuple[float, ...],
    replicates: int,
    seed: int = 20261018,
) -> dict:
    rng = np.random.default_rng(seed)
    names = (
        "mean_absolute_forward_move_bps",
        "mean_spread_plus_fee_bps",
        "mean_move_minus_cost_bps",
        "fraction_move_exceeds_cost",
    )
    samples = {fee: {name: [] for name in names} for fee in fees}
    for _ in range(replicates):
        index = moving_block_indices(groups, 100, rng)
        for fee in fees:
            values = fi_economics(
                labels[index],
                probabilities[index],
                forward_returns[index],
                relative_spreads[index],
                fee,
            )
            for name in names:
                samples[fee][name].append(values[name])
    return {
        str(fee): {
            name: {
                "lower": float(np.nanquantile(values, 0.025)),
                "upper": float(np.nanquantile(values, 0.975)),
            }
            for name, values in fee_samples.items()
        }
        for fee, fee_samples in samples.items()
    }


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
    if model in (
        "M1", "M1_anchor", "M1_refit", "M1_depth1", "M1_depth5",
        "M2", "M2_depth1", "M2_depth5", "B2", "B2_shipped144",
    ):
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
            "B1_leaky",
            "B2",
            "B2_shipped144",
            "M1",
            "M1_anchor",
            "M1_refit",
            "M1_depth1",
            "M1_depth5",
            "M2",
            "M2_depth1",
            "M2_depth5",
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
    pooled_returns, pooled_spreads = [], []
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
        if "forward_returns" not in prediction or "relative_spreads" not in prediction:
            raise KeyError(f"registered FI economic arrays are absent: {run / 'predictions.npz'}")
        pooled_returns.append(prediction["forward_returns"])
        pooled_spreads.append(prediction["relative_spreads"])
        reports.append(json.loads((run / "report.json").read_text(encoding="utf-8")))
    probabilities = np.concatenate(pooled_probabilities)
    labels = np.concatenate(pooled_labels)
    groups = np.concatenate(pooled_groups)
    forward_returns = np.concatenate(pooled_returns)
    relative_spreads = np.concatenate(pooled_spreads)
    point = classification_metrics(labels, probabilities)
    ci = bootstrap_metrics(labels, probabilities, groups, replicates=args.replicates)
    fees = (0.0, 0.5, 1.0, 2.0)
    economic_ci = bootstrap_fi_economics(
        labels,
        probabilities,
        groups,
        forward_returns,
        relative_spreads,
        fees,
        args.replicates,
    )
    fee_reports = {}
    for fee in fees:
        values = fi_economics(labels, probabilities, forward_returns, relative_spreads, fee)
        values["ci95"] = economic_ci[str(fee)]
        fee_reports[str(fee)] = values
    result = {
        "model": args.model, "variant": args.variant, "horizon": args.horizon,
        "class_order": ["down", "stationary", "up"],
        "selection_N_trials": int(reports[0].get("selection_N_trials", 0)),
        "reported_variant_N_trials": (
            1
            if args.model == "M1_anchor"
            else int(reports[0].get("reported_variant_N_trials", 0))
        ),
        "N_seeds": len(runs), "seed_macro_f1_sd": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
        "metrics": point, "ci95": ci,
        "economics_no_strategy_or_return_claim": {
            "primary_round_trip_fee_bps": 1.0,
            "fee_scenarios_bps": fee_reports,
            "price_basis": "FI_DecPre_relative_ratios",
        },
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
