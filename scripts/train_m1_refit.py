from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lobdl.datasets import FIWindowDataset, collate_predictions, fi_economic_arrays
from lobdl.metrics import classification_metrics
from lobdl.models import count_parameters
from lobdl.neural import build_model, build_optimizer_and_scheduler, make_test_dataset
from lobdl.search import FINAL_SEEDS
from lobdl.selection_lock import selection_path, verify_lock
from lobdl.training import fit_fixed_epochs, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Registered M1 days-1-through-7 refit amendment")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--seed", type=int, choices=FINAL_SEEDS, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for registered M1 refits")
    verify_lock(args.artifacts)
    selection = json.loads(
        selection_path(args.artifacts, "M1", args.horizon).read_text(encoding="utf-8")
    )
    primary_run = args.artifacts / "M1" / f"h{args.horizon}" / f"seed{args.seed}"
    primary_report_path = primary_run / "report.json"
    if not primary_report_path.exists():
        raise FileNotFoundError("primary M1 final run must precede its refit amendment")
    primary_report = json.loads(primary_report_path.read_text(encoding="utf-8"))
    fixed_epochs = int(primary_report["best_epoch"])
    parameters = selection["selected"]["parameters"]
    run = args.artifacts / "M1_refit" / f"h{args.horizon}" / f"seed{args.seed}"
    required = (run / "checkpoint.pt", run / "predictions.npz", run / "report.json")
    present = [path.exists() for path in required]
    if all(present):
        existing = json.loads((run / "report.json").read_text(encoding="utf-8"))
        if (
            existing.get("horizon") != args.horizon
            or existing.get("seed") != args.seed
            or existing.get("parameters") != parameters
            or existing.get("fixed_epochs_from_primary_seed_run") != fixed_epochs
        ):
            raise ValueError(f"completed refit metadata mismatch: {run}")
        print(json.dumps({**existing, "resume_status": "already_complete"}, indent=2))
        return
    if (run / "report.json").exists():
        raise FileExistsError(f"incomplete or corrupt refit artifact set: {run}")

    seed_everything(args.seed)
    device = torch.device("cuda")
    train = FIWindowDataset(args.data, list(range(1, 8)), args.horizon)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = build_model("M1", parameters).to(device)
    optimizer, scheduler, clip = build_optimizer_and_scheduler(
        "M1", model, parameters, fixed_epochs, len(train_loader)
    )
    checkpoint = run / "checkpoint.pt"
    fit = fit_fixed_epochs(
        model,
        train_loader,
        optimizer,
        device,
        checkpoint,
        fixed_epochs,
        clip,
        scheduler,
    )

    test = make_test_dataset(args.data, args.horizon)
    test_loader = DataLoader(
        test,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    probabilities, labels, groups, ends = collate_predictions(test_loader, model, device)
    forward_returns, relative_spreads = fi_economic_arrays(
        args.data, [8, 9, 10], args.horizon, purge=200
    )
    np.savez_compressed(
        run / "predictions.npz",
        probabilities=probabilities,
        labels=labels,
        groups=groups,
        ends=ends,
        forward_returns=forward_returns,
        relative_spreads=relative_spreads,
    )
    report = {
        "model": "M1",
        "variant": "refit_days_1_through_7_fixed_primary_epoch",
        "horizon": args.horizon,
        "seed": args.seed,
        "selection_N_trials": 12,
        "reported_variant_N_trials": 1,
        "parameters": parameters,
        "train_days": list(range(1, 8)),
        "fixed_epochs_from_primary_seed_run": fixed_epochs,
        "further_early_stopping": False,
        "trainable_parameters": count_parameters(model),
        "gpu_hours": fit.elapsed_hours,
        "component_metrics_not_for_reporting": classification_metrics(labels, probabilities),
        "reporting_status": "awaiting_pooled_seed_block_bootstrap",
    }
    (run / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
