from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .search import candidate_grid

MODELS = ("B1", "B2", "M1", "M2")
HORIZONS = (10, 20, 50, 100)


def verify_presearch_timing(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "measured_before_search":
        raise ValueError("timing report is not a registered pre-search measurement")
    if report.get("warmup_epochs") != 1 or report.get("measured_epochs") != 3:
        raise ValueError("timing report does not use the registered 1+3 epoch protocol")
    return report


def assert_not_frozen(root: Path) -> None:
    if (root / "selections" / "FROZEN.json").exists():
        raise RuntimeError("selections are frozen; refusing to alter validation searches")


def completed_selection(root: Path, model: str, horizon: int) -> dict | None:
    path = selection_path(root, model, horizon)
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_parameters = candidate_grid(model)
    observed_parameters = [item["parameters"] for item in report.get("candidates", ())]
    if (
        report.get("model") != model
        or report.get("horizon") != horizon
        or report.get("selection_N_trials") != 12
        or report.get("test_data_loaded") is not False
        or observed_parameters != expected_parameters
    ):
        raise ValueError(f"completed selection does not match registered grid: {path}")
    return report


def selection_path(root: Path, model: str, horizon: int) -> Path:
    return root / "selections" / model / f"h{horizon}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lock(root: Path) -> dict:
    selections = {}
    for model in MODELS:
        for horizon in HORIZONS:
            path = selection_path(root, model, horizon)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("test_data_loaded") is not False:
                raise ValueError(f"selection does not attest test isolation: {path}")
            if report.get("selection_N_trials") != 12:
                raise ValueError(f"selection does not contain 12 trials: {path}")
            if len(report.get("candidates", ())) != 12:
                raise ValueError(f"selection candidate count is not 12: {path}")
            key = f"{model}/h{horizon}"
            selections[key] = {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
                "selected_trial": report["selected"]["trial"],
                "selected_parameters": report["selected"]["parameters"],
            }
    return {
        "status": "all_validation_selections_frozen_before_test",
        "models": list(MODELS),
        "horizons": list(HORIZONS),
        "selections": selections,
    }


def verify_lock(root: Path) -> dict:
    path = root / "selections" / "FROZEN.json"
    locked = json.loads(path.read_text(encoding="utf-8"))
    current = build_lock(root)
    if current != locked:
        raise RuntimeError("selection files differ from the frozen pre-test manifest")
    return locked
