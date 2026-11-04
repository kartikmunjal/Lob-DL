"""Part C neural cells (M1 / M2 on engineered and combined representations).

No hyperparameter search: each cell reuses the Study 1 frozen selection of the same
model and horizon (read-only). Test windows are constructed only after the fit.

    python scripts/factorial_neural.py --data DATA --study1 STUDY1_ARTIFACTS \
        --output OUT --model M2 --representation eng --horizon 100 --seed 11
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lobdl.datasets import collate_predictions, fi_economic_arrays
from lobdl.factorial import (
    HORIZONS,
    NEW_NEURAL_CELLS,
    STATUS,
    build_neural,
    family,
    study1_selection_file,
    verify_lock,
)
from lobdl.metrics import classification_metrics
from lobdl.models import count_parameters
from lobdl.neural import build_optimizer_and_scheduler, make_loaders
from lobdl.representations import EngineeredScaler, FactorialWindowDataset
from lobdl.search import FINAL_SEEDS
from lobdl.selection_lock import verify_lock as verify_study1_lock
from lobdl.training import fit_classifier, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--study1", type=Path, required=True, help="Study 1 artifacts (read-only)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--model", choices=("M1", "M2"), required=True)
    parser.add_argument("--representation", choices=("eng", "both"), required=True)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    parser.add_argument("--seed", type=int, choices=FINAL_SEEDS, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=0,
        help="validation-only timing check: fit this many epochs, never build test, write nothing",
    )
    parser.add_argument("--allow-cpu", action="store_true", help="tests and smoke checks only")
    args = parser.parse_args()
    if (args.model, args.representation) not in NEW_NEURAL_CELLS:
        parser.error("not a Part C neural cell")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is required for Part C neural runs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verify_study1_lock(args.study1)
    if not args.smoke_epochs:
        verify_lock(args.output)
    selection = json.loads(
        study1_selection_file(args.study1, args.model, args.horizon).read_text(encoding="utf-8")
    )
    parameters = selection["selected"]["parameters"]
    cell = family(args.model, args.representation)
    run = args.output / cell / f"h{args.horizon}" / f"seed{args.seed}"
    if not args.smoke_epochs and (run / "predictions.npz").exists() and (
        run / "report.json"
    ).exists():
        print(f"already complete: {run}")
        return

    seed_everything(args.seed)
    scaler = EngineeredScaler.fit(args.data, cache_dir=args.cache)
    scaler.save(args.output / "scalers" / "engineered_train_days_1_6.json")
    train = FactorialWindowDataset(
        args.data, list(range(1, 7)), args.horizon, args.representation, scaler,
        cache_dir=args.cache,
    )
    validation = FactorialWindowDataset(
        args.data, [7], args.horizon, args.representation, scaler, purge=200,
        cache_dir=args.cache,
    )
    train_loader, validation_loader, _ = make_loaders(
        train, validation, None, args.seed, args.batch_size, args.workers
    )
    model = build_neural(args.model, args.representation, parameters).to(device)
    epochs = args.smoke_epochs or args.max_epochs
    optimizer, scheduler, clip = build_optimizer_and_scheduler(
        args.model, model, parameters, epochs, len(train_loader)
    )
    checkpoint = (args.output if not args.smoke_epochs else Path(args.output) / "smoke") / (
        f"{cell}_h{args.horizon}_seed{args.seed}.pt"
        if args.smoke_epochs
        else f"{cell}/h{args.horizon}/seed{args.seed}/checkpoint.pt"
    )
    fit = fit_classifier(
        model, train_loader, validation_loader, optimizer, device, checkpoint,
        epochs, args.patience, clip, scheduler,
    )
    if args.smoke_epochs:
        print(
            json.dumps(
                {
                    "cell": cell,
                    "horizon": args.horizon,
                    "epochs": args.smoke_epochs,
                    "gpu_hours": fit.elapsed_hours,
                    "seconds_per_epoch": fit.elapsed_hours * 3600.0 / args.smoke_epochs,
                    "best_validation_macro_f1": fit.best_macro_f1,
                    "parameters": count_parameters(model),
                },
                indent=2,
            )
        )
        return
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])

    # Test windows exist only after the validation-selected checkpoint is fixed.
    test = FactorialWindowDataset(
        args.data, [8, 9, 10], args.horizon, args.representation, scaler, purge=200,
        cache_dir=args.cache,
    )
    _, _, test_loader = make_loaders(
        train, validation, test, args.seed, args.batch_size, args.workers
    )
    probabilities, labels, groups, ends = collate_predictions(test_loader, model, device)
    forward_returns, relative_spreads = fi_economic_arrays(
        args.data, [8, 9, 10], args.horizon, purge=200
    )
    if len(forward_returns) != len(labels):
        raise AssertionError("FI economic arrays are not aligned with neural predictions")
    run.mkdir(parents=True, exist_ok=True)
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
        "model": args.model,
        "representation": args.representation,
        "architecture": (
            "LSTM-64 head (no convolutional trunk)"
            if (args.model, args.representation) == ("M1", "eng")
            else "DeepLOB trunk + engineered side channel"
            if (args.model, args.representation) == ("M1", "both")
            else "transformer encoder"
        ),
        "horizon": args.horizon,
        "seed": args.seed,
        "status": STATUS,
        "hyperparameters_source": f"study1_frozen_selection_{args.model}_h{args.horizon}",
        "hyperparameters_tuned_on": "raw representation (limitation)",
        "parameters": parameters,
        "trainable_parameters": count_parameters(model),
        "gpu_hours": fit.elapsed_hours,
        "best_epoch": fit.best_epoch,
        "validation_macro_f1": fit.best_macro_f1,
        "component_metrics_not_for_reporting": classification_metrics(labels, probabilities),
    }
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
