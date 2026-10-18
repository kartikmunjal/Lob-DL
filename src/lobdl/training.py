from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


@dataclass
class FitResult:
    best_epoch: int
    best_macro_f1: float
    elapsed_hours: float
    history: list[dict]


def fit_classifier(
    model: nn.Module,
    train_loader,
    validation_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output: Path,
    max_epochs: int = 40,
    patience: int = 5,
    clip: float | None = None,
    scheduler=None,
) -> FitResult:
    output.parent.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -np.inf
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for x, y, _, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            loss_sum += float(loss.detach()) * len(y)
            examples += len(y)
        model.eval()
        prediction, truth = [], []
        with torch.inference_mode():
            for x, y, _, _ in validation_loader:
                logits = model(x.to(device, non_blocking=True))
                prediction.append(logits.argmax(1).cpu().numpy())
                truth.append(y.numpy())
        score = float(f1_score(np.concatenate(truth), np.concatenate(prediction), average="macro"))
        entry = {"epoch": epoch, "train_loss": loss_sum / examples, "validation_macro_f1": score}
        history.append(entry)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_macro_f1": score}, output)
        else:
            stale += 1
            if stale >= patience:
                break
    elapsed = (time.perf_counter() - started) / 3600.0
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return FitResult(best_epoch, best_score, elapsed, history)


def fit_fixed_epochs(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output: Path,
    epochs: int,
    clip: float | None = None,
    scheduler=None,
) -> FitResult:
    """Fit an amendment-locked epoch count without validation or further stopping."""
    if epochs <= 0:
        raise ValueError("fixed epoch count must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for x, y, _, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            loss_sum += float(loss.detach()) * len(y)
            examples += len(y)
        history.append({"epoch": epoch, "train_loss": loss_sum / examples})
    elapsed = (time.perf_counter() - started) / 3600.0
    torch.save({"model": model.state_dict(), "epoch": epochs}, output)
    output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return FitResult(epochs, float("nan"), elapsed, history)
