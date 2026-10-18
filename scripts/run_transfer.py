from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lobdl.data.fi2010 import HORIZONS, LABEL_TO_INDEX
from lobdl.datasets import FIWindowDataset
from lobdl.features import causal_features
from lobdl.neural import build_model
from lobdl.search import FINAL_SEEDS
from lobdl.selection_lock import selection_path, verify_lock
from lobdl.transfer import (
    TargetArrays,
    TargetWindowDataset,
    apply_decimal_precision,
    calibration_boundary,
    class_proportions,
    decimal_precision_powers,
    eligible_ends,
    forward_mean_returns,
    label_returns,
    mid_and_spread,
    sample_message_blocks,
    select_distribution_matched_alpha,
)

TICKERS = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
STOCHASTIC = {"B2", "M1", "M2"}


def fi_training_proportions(root: Path, horizon: int) -> np.ndarray:
    dataset = FIWindowDataset(root, list(range(1, 7)), horizon)
    column = HORIZONS.index(horizon)
    labels = np.fromiter(
        (
            LABEL_TO_INDEX[int(dataset.arrays[day][end, 144 + column])]
            for day, _, end in dataset.index
        ),
        dtype=np.int8,
        count=len(dataset.index),
    )
    return class_proportions(labels)


def run_directory(artifacts: Path, model: str, horizon: int, seed: int | None) -> Path:
    if model in STOCHASTIC:
        return artifacts / model / f"h{horizon}" / f"seed{seed}"
    if model == "B1":
        return artifacts / model / f"h{horizon}" / "deterministic"
    rule = model.removeprefix("B0_")
    return artifacts / "B0" / f"h{horizon}" / rule


def load_estimator(artifacts: Path, model: str, horizon: int, seed: int | None):
    run = run_directory(artifacts, model, horizon, seed)
    if not (run / "report.json").exists():
        raise FileNotFoundError(f"primary FI-2010 final run is incomplete: {run}")
    if model in {"B1", "B2"}:
        with (run / "model.pkl").open("rb") as handle:
            return pickle.load(handle)
    if model in {"M1", "M2"}:
        selected = json.loads(
            selection_path(artifacts, model, horizon).read_text(encoding="utf-8")
        )["selected"]["parameters"]
        network = build_model(model, selected)
        saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        network.load_state_dict(saved["model"])
        return network
    return None


def baseline_probabilities(
    model: str,
    estimator,
    scaled_book: np.ndarray,
    ends: np.ndarray,
    majority: int,
) -> np.ndarray:
    if model == "B0_majority":
        probabilities = np.zeros((len(ends), 3), dtype=np.float64)
        probabilities[:, majority] = 1.0
        return probabilities
    features, names = causal_features(scaled_book)
    values = features[ends]
    if model == "B0_persistence":
        movement = values[:, names.index("mid_log_return_1")]
        prediction = np.where(movement > 0, 2, np.where(movement < 0, 0, 1))
        probabilities = np.zeros((len(ends), 3), dtype=np.float64)
        probabilities[np.arange(len(ends)), prediction] = 1.0
        return probabilities
    if estimator["feature_names"] != names or estimator["feature_set"] != "primary":
        raise AssertionError("serialized FI estimator does not use the locked primary features")
    if estimator["scaler"] is not None:
        values = estimator["scaler"].transform(values)
    return estimator["model"].predict_proba(values)


def neural_probabilities(
    network: torch.nn.Module,
    scaled_book: np.ndarray,
    raw_book: np.ndarray,
    ends: np.ndarray,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    dataset = TargetWindowDataset(TargetArrays(scaled_book, raw_book, ends))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    network.to(device).eval()
    output = []
    with torch.inference_mode():
        for values in loader:
            logits = network(values.to(device, non_blocking=True))
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Registered zero-shot FI-2010 to LOBSTER T1")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fi-data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--model",
        choices=("B0_majority", "B0_persistence", "B1", "B2", "M1", "M2"),
        required=True,
    )
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--seed", type=int, choices=FINAL_SEEDS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    stochastic = args.model in STOCHASTIC
    if stochastic != (args.seed is not None):
        parser.error("B2/M1/M2 require a final seed; deterministic systems prohibit one")

    verify_lock(args.artifacts)
    estimator = load_estimator(args.artifacts, args.model, args.horizon, args.seed)
    target = fi_training_proportions(args.fi_data, args.horizon)
    majority = int(target.argmax())
    horizon_rows = args.horizon // 10
    alpha_grid = np.linspace(0.0, 0.01, 10001, dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model in {"M1", "M2"} and device.type != "cuda":
        raise SystemExit("CUDA is required for registered neural transfer inference")

    pooled: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "probabilities",
            "labels_fixed",
            "labels_matched",
            "groups",
            "ends",
            "forward_returns",
            "mid",
            "spread",
        )
    }
    ticker_reports = {}
    for ticker_index, ticker in enumerate(TICKERS, start=1):
        path = next(args.raw.glob(f"{ticker}_*_orderbook_10.csv"))
        vendor = pd.read_csv(path, header=None).to_numpy(dtype=np.int64)
        raw_book = sample_message_blocks(vendor)
        boundary = calibration_boundary(len(raw_book))
        powers = decimal_precision_powers(raw_book[:boundary])
        scaled_book = apply_decimal_precision(raw_book, powers)
        returns = forward_mean_returns(raw_book, horizon_rows)
        calibration_ends = eligible_ends(
            len(raw_book), boundary, horizon_rows, split="calibration"
        )
        evaluation_ends = eligible_ends(
            len(raw_book), boundary, horizon_rows, split="evaluation"
        )
        alpha, distance, calibration_proportions = select_distribution_matched_alpha(
            returns[calibration_ends], target, alpha_grid
        )
        fixed = label_returns(returns[evaluation_ends], 0.002)
        matched = label_returns(returns[evaluation_ends], alpha)
        if np.any(fixed < 0) or np.any(matched < 0):
            raise AssertionError("evaluation labels crossed their registered split boundary")

        if args.model in {"M1", "M2"}:
            probabilities = neural_probabilities(
                estimator,
                scaled_book,
                raw_book,
                evaluation_ends,
                device,
                args.batch_size,
                args.workers,
            )
        else:
            probabilities = baseline_probabilities(
                args.model, estimator, scaled_book, evaluation_ends, majority
            )
        mid, spread = mid_and_spread(raw_book)
        pooled["probabilities"].append(probabilities)
        pooled["labels_fixed"].append(fixed)
        pooled["labels_matched"].append(matched)
        pooled["groups"].append(
            np.tile(np.array([[ticker_index, 1]], dtype=np.int16), (len(evaluation_ends), 1))
        )
        pooled["ends"].append(evaluation_ends)
        pooled["forward_returns"].append(returns[evaluation_ends])
        # LOBSTER encodes prices in 1/10,000 dollars; economics are reported in dollars.
        pooled["mid"].append(mid[evaluation_ends] / 10000.0)
        pooled["spread"].append(spread[evaluation_ends] / 10000.0)
        ticker_reports[ticker] = {
            "vendor_rows": len(vendor),
            "sampled_rows": len(raw_book),
            "calibration_boundary": boundary,
            "calibration_eligible_labels": len(calibration_ends),
            "evaluation_windows": len(evaluation_ends),
            "decimal_precision_powers": powers.tolist(),
            "matched_alpha": alpha,
            "matched_total_variation_distance": distance,
            "fi_training_class_proportions_down_flat_up": target.tolist(),
            "calibration_class_proportions_down_flat_up": calibration_proportions.tolist(),
            "evaluation_fixed_class_counts_down_flat_up": np.bincount(
                fixed, minlength=3
            ).tolist(),
            "evaluation_matched_class_counts_down_flat_up": np.bincount(
                matched, minlength=3
            ).tolist(),
        }

    run_name = "deterministic" if args.seed is None else f"seed{args.seed}"
    output = args.artifacts / "transfer" / args.model / f"h{args.horizon}" / run_name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz", **{key: np.concatenate(value) for key, value in pooled.items()}
    )
    report = {
        "model": args.model,
        "horizon": args.horizon,
        "seed": args.seed,
        "source_training": "FI-2010_days_1_to_6_selected_primary_fit",
        "target": "LOBSTER_vendor_book_zero_shot",
        "messages_per_representation": 10,
        "calibration_fraction": 0.2,
        "fixed_alpha": 0.002,
        "matched_alpha_grid": {"start": 0.0, "stop": 0.01, "points": 10001},
        "ticker_reports": ticker_reports,
        "reporting_status": "awaiting_pooled_seed_block_bootstrap",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
