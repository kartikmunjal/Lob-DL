from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

HORIZONS = (10, 20, 50, 100)
PRIMARY_SYSTEMS = (
    "B0:majority",
    "B0:persistence",
    "B1",
    "B1_shipped144",
    "B2",
    "B2_shipped144",
    "M1",
    "M1_anchor",
    "M2",
)
SECONDARY_SYSTEMS = (
    "B1_leaky",
    "M1_refit",
    "M1_depth1",
    "M1_depth5",
    "M2_depth1",
    "M2_depth5",
)
TRANSFER_SYSTEMS = ("B0_majority", "B0_persistence", "B1", "B2", "M1", "M2")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def expected_reports() -> tuple[Path, ...]:
    paths: list[Path] = []
    for horizon in HORIZONS:
        paths.extend(
            (
                Path("B0") / f"h{horizon}" / "summary_majority.json",
                Path("B0") / f"h{horizon}" / "summary_persistence.json",
            )
        )
        for model in PRIMARY_SYSTEMS[2:] + SECONDARY_SYSTEMS:
            paths.append(Path(model) / f"h{horizon}" / "summary.json")
        for system_a, system_b in itertools.combinations(PRIMARY_SYSTEMS, 2):
            name_a, name_b = system_a.replace(":", "_"), system_b.replace(":", "_")
            paths.append(Path("paired") / f"h{horizon}" / f"{name_a}_minus_{name_b}.json")
        for model, variant in (
            ("M1", "M1_refit"),
            ("M1", "M1_depth1"),
            ("M1", "M1_depth5"),
            ("M2", "M2_depth1"),
            ("M2", "M2_depth5"),
        ):
            paths.append(Path("paired") / f"h{horizon}" / f"{model}_minus_{variant}.json")
        for model_a, model_b in (("B2", "B1"), ("M1", "B2"), ("M2", "M1")):
            paths.append(
                Path("mechanisms") / f"h{horizon}" / f"{model_a}_minus_{model_b}.json"
            )
        for labels in ("fixed", "matched"):
            for model in TRANSFER_SYSTEMS:
                paths.append(
                    Path("transfer") / model / f"h{horizon}" / f"summary_{labels}.json"
                )
            for system_a, system_b in itertools.combinations(TRANSFER_SYSTEMS, 2):
                paths.append(
                    Path("transfer")
                    / "paired"
                    / f"h{horizon}"
                    / f"{system_a}_minus_{system_b}_{labels}.json"
                )
    return tuple(paths)


def publish(artifacts: Path, output: Path) -> dict:
    source_root = artifacts / "reports"
    expected = expected_reports()
    missing = [str(source_root / relative) for relative in expected if not (source_root / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing registered reports:\n" + "\n".join(missing))
    files = {}
    for relative in expected:
        source = source_root / relative
        target = output / relative
        payload = source.read_bytes()
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(f"published report differs and will not be overwritten: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        files[str(relative)] = {"sha256": digest(payload), "bytes": len(payload)}
    report = {
        "status": "all_registered_reports_published",
        "expected_report_count": len(expected),
        "files": files,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "MANIFEST.json"
    payload = json.dumps(report, indent=2) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"published manifest differs and will not be overwritten: {manifest}")
    manifest.write_text(payload, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the complete immutable registered report set"
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/results"))
    args = parser.parse_args()
    print(json.dumps(publish(args.artifacts, args.output), indent=2))


if __name__ == "__main__":
    main()
