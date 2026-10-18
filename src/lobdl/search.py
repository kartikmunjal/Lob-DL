from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

SELECTION_SEEDS = (5, 7, 13)
FINAL_SEEDS = (11, 23, 47, 91, 137)


def load_progress(path: Path, model: str, horizon: int) -> dict:
    if not path.exists():
        return {"model": model, "horizon": horizon, "completed_fits": {}}
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("model") != model or progress.get("horizon") != horizon:
        raise ValueError(f"progress metadata mismatch in {path}")
    return progress


def save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_grid(model: str) -> list[dict[str, float | int]]:
    if model == "B1":
        return [{"C": float(value)} for value in np.logspace(-4, 2, 12)]
    if model == "B2":
        return [
            {"num_leaves": leaves, "learning_rate": rate, "min_child_samples": minimum}
            for leaves, rate, minimum in itertools.product(
                (31, 63), (0.03, 0.05, 0.1), (200, 400)
            )
        ]
    if model == "M1":
        return [
            {"learning_rate": rate, "epsilon": epsilon, "dropout": dropout}
            for rate, epsilon, dropout in itertools.product(
                (1e-4, 1e-3, 1e-2), (1e-8, 1.0), (0.0, 0.2)
            )
        ]
    if model == "M2":
        return [
            {"learning_rate": rate, "layers": layers, "dropout": dropout}
            for rate, layers, dropout in itertools.product(
                (3e-4, 1e-3, 3e-3), (2, 3), (0.1, 0.2)
            )
        ]
    raise ValueError(f"no registered grid for {model}")


def summarize_candidate(
    trial: int,
    parameters: dict[str, float | int],
    scores: Iterable[float],
    parameter_count: int,
    best_epochs: Iterable[int] = (),
) -> dict:
    values = np.asarray(tuple(scores), dtype=np.float64)
    if not len(values):
        raise ValueError("candidate needs at least one validation score")
    return {
        "trial": trial,
        "parameters": parameters,
        "validation_macro_f1_by_replication": values.tolist(),
        "mean_validation_macro_f1": float(values.mean()),
        "sd_validation_macro_f1": float(values.std(ddof=0)),
        "parameter_count": int(parameter_count),
        "best_epochs": list(best_epochs),
    }


def select_candidate(candidates: list[dict]) -> dict:
    """Apply the exact-tie rule in canonical trial order."""
    if not candidates:
        raise ValueError("cannot select from an empty candidate list")
    return min(
        candidates,
        key=lambda item: (
            -item["mean_validation_macro_f1"],
            item["sd_validation_macro_f1"],
            item["parameter_count"],
            item["parameters"].get("learning_rate", 0.0),
            item["trial"],
        ),
    )
