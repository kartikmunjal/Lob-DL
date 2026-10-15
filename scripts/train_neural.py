from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lobdl.datasets import FIWindowDataset, collate_predictions
from lobdl.metrics import classification_metrics
from lobdl.models import DeepLOB, TransformerLOB, count_parameters
from lobdl.training import fit_classifier, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=("M1", "M2"), required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--seed", type=int, choices=(11, 23, 47, 91, 137), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for registered neural runs")
    seed_everything(args.seed)
    device = torch.device("cuda")
    train = FIWindowDataset(args.data, list(range(1, 7)), args.horizon)
    validation = FIWindowDataset(args.data, [7], args.horizon, purge=200)
    test = FIWindowDataset(args.data, [8, 9, 10], args.horizon, purge=200)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, generator=generator, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    validation_loader = DataLoader(validation, args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    test_loader = DataLoader(test, args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    model = (DeepLOB() if args.model == "M1" else TransformerLOB()).to(device)
    run = args.output / args.model / f"h{args.horizon}" / f"seed{args.seed}"
    checkpoint = run / "checkpoint.pt"
    if args.model == "M1":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = None
        clip = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        total_steps = args.max_epochs * math.ceil(len(train) / args.batch_size)
        warmup = int(0.05 * total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, total_steps - warmup))))
        clip = 1.0
    fit = fit_classifier(model, train_loader, validation_loader, optimizer, device, checkpoint, args.max_epochs, args.patience, clip, scheduler)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    probabilities, labels, groups, ends = collate_predictions(test_loader, model, device)
    metrics = classification_metrics(labels, probabilities)
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / "predictions.npz", probabilities=probabilities, labels=labels, groups=groups, ends=ends)
    report = {"model": args.model, "horizon": args.horizon, "seed": args.seed, "N_trials": 12, "parameters": count_parameters(model), "gpu_hours": fit.elapsed_hours, "best_epoch": fit.best_epoch, "validation_macro_f1": fit.best_macro_f1, "component_metrics_not_for_reporting": metrics, "reporting_status": "awaiting_pooled_seed_block_bootstrap"}
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
