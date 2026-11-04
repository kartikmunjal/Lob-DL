"""Part C factorial bookkeeping: cells, artifact paths, and the pre-test lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .models import DeepLOB, DeepLOBFused, LSTMHead, TransformerLOB
from .representations import ENG_COLUMNS, channel_count

HORIZONS = (10, 20, 50, 100)
REPRESENTATIONS = ("raw", "eng", "both")
STATUS = "preregistered_new_fit_study2_C"
# (model, representation) cells that Part C fits; Study 1 already covers the others.
NEW_B2_REPRESENTATIONS = ("raw", "both")
NEW_NEURAL_CELLS = (
    ("M1", "eng"),
    ("M1", "both"),
    ("M2", "eng"),
    ("M2", "both"),
)
PHASE_A = (100, 10)
PHASE_B = (50, 20)


def family(model: str, representation: str) -> str:
    return f"F_{model}_{representation}"


def selection_file(root: Path, model: str, representation: str, horizon: int) -> Path:
    return root / "factorial_selections" / family(model, representation) / f"h{horizon}.json"


def lock_file(root: Path) -> Path:
    return root / "factorial_selections" / "FROZEN.json"


def study1_selection_file(root: Path, model: str, horizon: int) -> Path:
    return root / "selections" / model / f"h{horizon}.json"


def build_neural(model: str, representation: str, parameters: dict) -> torch.nn.Module:
    """Neural model for one Part C cell, using Study 1 frozen hyperparameters."""
    dropout = float(parameters["dropout"])
    if model == "M1":
        if representation == "raw":
            return DeepLOB(dropout=dropout)
        if representation == "both":
            return DeepLOBFused(side_features=ENG_COLUMNS, dropout=dropout)
        return LSTMHead(features=channel_count("eng"), dropout=dropout)
    if model == "M2":
        return TransformerLOB(
            features=channel_count(representation),
            layers=int(parameters["layers"]),
            dropout=dropout,
        )
    raise ValueError(f"unsupported neural model {model}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lock(root: Path) -> dict:
    """Hash all eight new B2 selections; each must attest test isolation."""
    selections = {}
    for representation in NEW_B2_REPRESENTATIONS:
        for horizon in HORIZONS:
            path = selection_file(root, "B2", representation, horizon)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("test_data_loaded") is not False:
                raise ValueError(f"selection does not attest test isolation: {path}")
            if len(report.get("candidates", ())) != 12:
                raise ValueError(f"selection candidate count is not 12: {path}")
            selections[f"B2/{representation}/h{horizon}"] = {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "selected_parameters": report["selected"]["parameters"],
            }
    return {"status": "partC_b2_selections_frozen_before_test", "selections": selections}


def verify_lock(root: Path) -> dict:
    locked = json.loads(lock_file(root).read_text(encoding="utf-8"))
    if build_lock(root) != locked:
        raise RuntimeError("Part C selection files differ from the frozen manifest")
    return locked
