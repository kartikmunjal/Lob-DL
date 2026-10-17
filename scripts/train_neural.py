from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lobdl.datasets import collate_predictions
from lobdl.metrics import classification_metrics
from lobdl.models import count_parameters
from lobdl.neural import (
    M1_ANCHOR,
    build_model,
    build_optimizer_and_scheduler,
    make_loaders,
    make_test_dataset,
    make_train_validation_datasets,
)
from lobdl.search import FINAL_SEEDS
from lobdl.selection_lock import selection_path, verify_lock
from lobdl.training import fit_classifier, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-selection neural final fit")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=("M1", "M2"), required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--seed", type=int, choices=FINAL_SEEDS, required=True)
    parser.add_argument("--variant", choices=("selected", "anchor"), default="selected")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if args.variant == "anchor" and args.model != "M1":
        parser.error("the preregistered paper-optimizer anchor applies only to M1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for registered neural runs")

    verify_lock(args.output)
    selection = json.loads(
        selection_path(args.output, args.model, args.horizon).read_text(encoding="utf-8")
    )
    selected_parameters = selection["selected"]["parameters"]
    parameters = M1_ANCHOR if args.variant == "anchor" else selected_parameters
    if args.variant == "anchor" and parameters == selected_parameters:
        raise SystemExit(
            "M1 anchor is the selected configuration; its selected runs serve as anchor runs"
        )

    seed_everything(args.seed)
    device = torch.device("cuda")
    train, validation = make_train_validation_datasets(args.data, args.horizon)
    train_loader, validation_loader, _ = make_loaders(
        train, validation, None, args.seed, args.batch_size, args.workers
    )
    model = build_model(args.model, parameters).to(device)
    family = args.model if args.variant == "selected" else "M1_anchor"
    run = args.output / family / f"h{args.horizon}" / f"seed{args.seed}"
    checkpoint = run / "checkpoint.pt"
    optimizer, scheduler, clip = build_optimizer_and_scheduler(
        args.model, model, parameters, args.max_epochs, len(train_loader)
    )
    fit = fit_classifier(
        model,
        train_loader,
        validation_loader,
        optimizer,
        device,
        checkpoint,
        args.max_epochs,
        args.patience,
        clip,
        scheduler,
    )
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])

    # Test construction happens only after the validation-selected checkpoint
    # is fixed and the global selection lock has been verified.
    test = make_test_dataset(args.data, args.horizon)
    _, _, test_loader = make_loaders(
        train, validation, test, args.seed, args.batch_size, args.workers
    )
    probabilities, labels, groups, ends = collate_predictions(test_loader, model, device)
    metrics = classification_metrics(labels, probabilities)
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run / "predictions.npz",
        probabilities=probabilities,
        labels=labels,
        groups=groups,
        ends=ends,
    )
    report = {
        "model": args.model,
        "variant": args.variant,
        "horizon": args.horizon,
        "seed": args.seed,
        "selection_N_trials": selection["selection_N_trials"],
        "reported_variant_N_trials": 1 if args.variant == "anchor" else 0,
        "selection_conditional_on_seeds": selection["selection_seeds"],
        "parameters": parameters,
        "trainable_parameters": count_parameters(model),
        "gpu_hours": fit.elapsed_hours,
        "best_epoch": fit.best_epoch,
        "validation_macro_f1": fit.best_macro_f1,
        "component_metrics_not_for_reporting": metrics,
        "reporting_status": "awaiting_pooled_seed_block_bootstrap",
    }
    (run / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
