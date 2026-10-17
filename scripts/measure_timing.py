from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn

from lobdl.neural import (
    M1_ANCHOR,
    build_model,
    build_optimizer_and_scheduler,
    make_loaders,
    make_train_validation_datasets,
)
from lobdl.training import seed_everything

TIMING_CONFIGS = {
    "M1": M1_ANCHOR,
    "M2": {"learning_rate": 1e-3, "layers": 3, "dropout": 0.1},
}
FIT_COUNTS = {"M1": 184, "M2": 164}


def epoch(
    model: nn.Module,
    train_loader,
    validation_loader,
    optimizer,
    scheduler,
    clip,
    device: torch.device,
) -> float:
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    started = time.perf_counter()
    model.train()
    for x, y, _, _ in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        if clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
    predictions, labels = [], []
    model.eval()
    with torch.inference_mode():
        for x, y, _, _ in validation_loader:
            predictions.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
            labels.append(y)
    f1_score(torch.cat(labels).numpy(), torch.cat(predictions).numpy(), average="macro")
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Registered pre-search GPU timing measurement")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/phase1_timing.json"))
    parser.add_argument("--horizon", type=int, default=10, choices=(10, 20, 50, 100))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for registered timing")

    train, validation = make_train_validation_datasets(args.data, args.horizon)
    device = torch.device("cuda")
    results = {}
    for model_name, parameters in TIMING_CONFIGS.items():
        seed_everything(5)
        train_loader, validation_loader, _ = make_loaders(
            train, validation, None, 5, args.batch_size, args.workers
        )
        model = build_model(model_name, parameters).to(device)
        optimizer, scheduler, clip = build_optimizer_and_scheduler(
            model_name, model, parameters, 40, len(train_loader)
        )
        warmup_seconds = epoch(
            model, train_loader, validation_loader, optimizer, scheduler, clip, device
        )
        measured = [
            epoch(model, train_loader, validation_loader, optimizer, scheduler, clip, device)
            for _ in range(3)
        ]
        median = statistics.median(measured)
        fits = FIT_COUNTS[model_name]
        results[model_name] = {
            "configuration": parameters,
            "fit_count_upper_bound": fits,
            "warmup_epoch_seconds_unmeasured": warmup_seconds,
            "measured_epoch_seconds": measured,
            "median_epoch_seconds": median,
            "expected_20_epoch_gpu_hours": median * fits * 20 / 3600,
            "no_early_stopping_40_epoch_gpu_hours": median * fits * 40 / 3600,
        }
    report = {
        "status": "measured_before_search",
        "device": torch.cuda.get_device_name(0),
        "horizon": args.horizon,
        "train_days": [1, 2, 3, 4, 5, 6],
        "validation_day": 7,
        "warmup_epochs": 1,
        "measured_epochs": 3,
        "statistic": "median",
        "expected_projection_epoch_scenario": 20,
        "upper_bound_epochs": 40,
        "results": results,
        "combined_expected_gpu_hours": sum(
            result["expected_20_epoch_gpu_hours"] for result in results.values()
        ),
        "combined_upper_bound_gpu_hours": sum(
            result["no_early_stopping_40_epoch_gpu_hours"] for result in results.values()
        ),
        "scope_changes_permitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
