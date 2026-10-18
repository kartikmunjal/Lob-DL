from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets import FIWindowDataset
from .models import DeepLOB, TransformerLOB

M1_ANCHOR = {"learning_rate": 1e-2, "epsilon": 1.0, "dropout": 0.0}


def build_model(model_name: str, parameters: dict, levels: int = 10) -> torch.nn.Module:
    if model_name == "M1":
        return DeepLOB(dropout=float(parameters["dropout"]), levels=levels)
    if model_name == "M2":
        return TransformerLOB(
            features=levels * 4,
            layers=int(parameters["layers"]),
            dropout=float(parameters["dropout"]),
        )
    raise ValueError(f"unsupported neural model {model_name}")


def build_optimizer_and_scheduler(
    model_name: str,
    model: torch.nn.Module,
    parameters: dict,
    max_epochs: int,
    steps_per_epoch: int,
):
    if model_name == "M1":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(parameters["learning_rate"]),
            eps=float(parameters["epsilon"]),
            weight_decay=0.0,
        )
        return optimizer, None, None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=0.01,
    )
    total_steps = max_epochs * steps_per_epoch
    warmup = int(0.05 * total_steps)

    def schedule(step: int) -> float:
        warmup_factor = min(1.0, step / max(1, warmup))
        cosine_step = max(0, step - warmup)
        cosine_span = max(1, total_steps - warmup)
        return warmup_factor * 0.5 * (1 + math.cos(math.pi * cosine_step / cosine_span))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler, 1.0


def make_train_validation_datasets(data: Path, horizon: int, levels: int = 10):
    """Construct only development data; test is deliberately unavailable here."""
    return (
        FIWindowDataset(data, list(range(1, 7)), horizon, levels=levels),
        FIWindowDataset(data, [7], horizon, purge=200, levels=levels),
    )


def make_test_dataset(data: Path, horizon: int, levels: int = 10):
    return FIWindowDataset(data, [8, 9, 10], horizon, purge=200, levels=levels)


def make_loaders(
    train,
    validation,
    test,
    seed: int,
    batch_size: int,
    workers: int,
):
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train,
        batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        validation, batch_size * 2, shuffle=False, **common
    )
    test_loader = None
    if test is not None:
        test_loader = DataLoader(test, batch_size * 2, shuffle=False, **common)
    return train_loader, validation_loader, test_loader
