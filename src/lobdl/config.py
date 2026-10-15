from __future__ import annotations

from pathlib import Path

import yaml


def load_protocol(path: str | Path = "configs/protocol.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

