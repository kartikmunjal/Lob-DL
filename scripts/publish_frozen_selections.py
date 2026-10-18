from __future__ import annotations

import argparse
import json
from pathlib import Path

from lobdl.selection_lock import HORIZONS, MODELS, selection_path, sha256, verify_lock


def copy_immutable(source: Path, target: Path) -> None:
    payload = source.read_bytes()
    if target.exists() and target.read_bytes() != payload:
        raise FileExistsError(f"published frozen file differs and will not be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def publish(artifacts: Path, output: Path) -> dict:
    lock = verify_lock(artifacts)
    files = {}
    for model in MODELS:
        for horizon in HORIZONS:
            source = selection_path(artifacts, model, horizon)
            target = output / model / f"h{horizon}.json"
            copy_immutable(source, target)
            files[f"{model}/h{horizon}"] = {
                "path": str(target.relative_to(output)),
                "sha256": sha256(target),
            }
    source_lock = artifacts / "selections" / "FROZEN.json"
    target_lock = output / "FROZEN.json"
    copy_immutable(source_lock, target_lock)
    report = {
        "status": "validation_selections_published_before_test",
        "source_lock_sha256": sha256(source_lock),
        "frozen_status": lock["status"],
        "files": files,
    }
    manifest = output / "PUBLISHED.json"
    payload = json.dumps(report, indent=2) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != payload:
        raise FileExistsError(
            f"published selection manifest differs and will not be overwritten: {manifest}"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(payload, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and publish all frozen validation selections before test"
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/frozen_selections")
    )
    args = parser.parse_args()
    print(json.dumps(publish(args.artifacts, args.output), indent=2))


if __name__ == "__main__":
    main()
