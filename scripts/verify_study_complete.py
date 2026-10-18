from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from publish_registered_results import expected_reports

from lobdl.search import FINAL_SEEDS
from lobdl.selection_lock import HORIZONS, verify_lock

FIGURES = {
    "fi_inside_spread_primary_fee.png",
    "fi_mechanical_explanation.png",
    "fi_primary_macro_f1.png",
    "fi_reliability.png",
    "inside_spread_primary_fee.png",
    "lobster_fixed_macro_f1.png",
    "lobster_matched_macro_f1.png",
    "lobster_replay_exact_rate.png",
}
TABLES = {
    "compute_and_parameters.csv",
    "fi_ablations.csv",
    "fi_economic_significance.csv",
    "fi_label_formula_agreement.csv",
    "fi_mechanical_explanation.csv",
    "fi_primary_metrics.csv",
    "lobster_economic_significance.csv",
    "lobster_reconstruction_by_time.csv",
    "lobster_reconstruction_summary.csv",
    "lobster_transfer_metrics.csv",
    "paired_model_comparisons.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("required study artifacts are absent:\n" + "\n".join(missing))


def verify_published_manifest(root: Path, expected_count: int) -> None:
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "all_registered_reports_published":
        raise ValueError("registered-report publication is not complete")
    if manifest.get("expected_report_count") != expected_count:
        raise ValueError("registered-report manifest count differs from the frozen inventory")
    for relative, metadata in manifest["files"].items():
        path = root / relative
        if sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"published report digest mismatch: {path}")


def verify_summary(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    for key in ("selection_N_trials", "reported_variant_N_trials", "N_seeds"):
        if key not in report:
            raise KeyError(f"{key} absent from {path}")
    metrics, intervals = report.get("metrics", {}), report.get("ci95", {})
    for metric in ("precision", "recall", "f1", "macro_f1", "mcc", "ece"):
        if metric not in metrics or metric not in intervals:
            raise KeyError(f"metric or 95% interval for {metric} absent from {path}")
    economics = report.get("economics_no_strategy_or_return_claim", {})
    if set(economics.get("fee_scenarios_bps", {})) != {"0.0", "0.5", "1.0", "2.0"}:
        raise ValueError(f"economic-significance fee scenarios incomplete in {path}")
    for values in economics["fee_scenarios_bps"].values():
        if "ci95" not in values:
            raise KeyError(f"economic-significance 95% interval absent from {path}")


def primary_run_files(artifacts: Path) -> list[Path]:
    paths: list[Path] = []
    for horizon in HORIZONS:
        for rule in ("majority", "persistence"):
            run = artifacts / "B0" / f"h{horizon}" / rule
            paths.extend((run / "predictions.npz", run / "report.json"))
        for family in ("B1", "B1_shipped144"):
            run = artifacts / family / f"h{horizon}" / "deterministic"
            paths.extend((run / "predictions.npz", run / "report.json", run / "model.pkl"))
        for family in ("B2", "B2_shipped144"):
            for seed in FINAL_SEEDS:
                run = artifacts / family / f"h{horizon}" / f"seed{seed}"
                paths.extend((run / "predictions.npz", run / "report.json", run / "model.pkl"))
        for family in ("M1", "M2"):
            for seed in FINAL_SEEDS:
                run = artifacts / family / f"h{horizon}" / f"seed{seed}"
                paths.extend((run / "checkpoint.pt", run / "predictions.npz", run / "report.json"))
        selection = json.loads(
            (artifacts / "selections" / "M1" / f"h{horizon}.json").read_text(
                encoding="utf-8"
            )
        )
        anchor = {"learning_rate": 0.01, "epsilon": 1.0, "dropout": 0.0}
        if selection["selected"]["parameters"] != anchor:
            for seed in FINAL_SEEDS:
                run = artifacts / "M1_anchor" / f"h{horizon}" / f"seed{seed}"
                paths.extend((run / "checkpoint.pt", run / "predictions.npz", run / "report.json"))
    return paths


def secondary_run_files(artifacts: Path) -> list[Path]:
    paths: list[Path] = []
    for horizon in HORIZONS:
        leaky = artifacts / "B1_leaky" / f"h{horizon}" / "deterministic"
        paths.extend((leaky / "predictions.npz", leaky / "report.json"))
        for family in (
            "M1_refit",
            "M1_depth1",
            "M1_depth5",
            "M2_depth1",
            "M2_depth5",
        ):
            for seed in FINAL_SEEDS:
                run = artifacts / family / f"h{horizon}" / f"seed{seed}"
                paths.extend((run / "checkpoint.pt", run / "predictions.npz", run / "report.json"))
    return paths


def transfer_run_files(artifacts: Path) -> list[Path]:
    paths: list[Path] = []
    for horizon in HORIZONS:
        for family in ("B0_majority", "B0_persistence", "B1"):
            run = artifacts / "transfer" / family / f"h{horizon}" / "deterministic"
            paths.extend((run / "predictions.npz", run / "report.json"))
        for family in ("B2", "M1", "M2"):
            for seed in FINAL_SEEDS:
                run = artifacts / "transfer" / family / f"h{horizon}" / f"seed{seed}"
                paths.extend((run / "predictions.npz", run / "report.json"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the completed preregistered study")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    verify_lock(args.artifacts)
    frozen = args.reports / "frozen_selections"
    require([frozen / "FROZEN.json", frozen / "PUBLISHED.json"])
    if (frozen / "FROZEN.json").read_bytes() != (
        args.artifacts / "selections" / "FROZEN.json"
    ).read_bytes():
        raise ValueError("published and active frozen-selection locks differ")
    publication = json.loads((frozen / "PUBLISHED.json").read_text(encoding="utf-8"))
    if publication.get("status") != "validation_selections_published_before_test":
        raise ValueError("frozen selections do not carry the pre-test publication attestation")
    if publication.get("source_lock_sha256") != sha256(frozen / "FROZEN.json"):
        raise ValueError("published frozen-selection lock digest differs")
    for metadata in publication.get("files", {}).values():
        path = frozen / metadata["path"]
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"published selection digest mismatch: {path}")

    require(primary_run_files(args.artifacts))
    require(secondary_run_files(args.artifacts))
    require(transfer_run_files(args.artifacts))

    registered = args.reports / "results"
    relative_reports = expected_reports()
    require([registered / path for path in relative_reports])
    verify_published_manifest(registered, len(relative_reports))
    for relative in relative_reports:
        source = args.artifacts / "reports" / relative
        published = registered / relative
        if source.read_bytes() != published.read_bytes():
            raise ValueError(f"active and published reports differ: {relative}")
        if relative.name.startswith("summary"):
            verify_summary(published)

    generated = args.reports / "generated"
    require(
        [
            generated / "RESULTS.md",
            generated / "external_anchor" / "comparison.json",
            generated / "external_anchor" / "comparison.csv",
        ]
        + [generated / "figures" / name for name in FIGURES]
        + [generated / "tables" / name for name in TABLES]
        + [args.reports / "audits" / name for name in (
            "fi2010.json",
            "lobster_all.json",
            "reconstructed_sensitivity.json",
        )]
    )
    readme = Path("README.md").read_text(encoding="utf-8")
    if "No model has seen test data" in readme:
        raise ValueError("README still claims the study is pre-test")
    print(
        json.dumps(
            {
                "status": "complete_study_artifact_audit_passed",
                "registered_reports": len(relative_reports),
                "figures": len(FIGURES),
                "tables": len(TABLES),
                "primary_files_checked": len(primary_run_files(args.artifacts)),
                "secondary_files_checked": len(secondary_run_files(args.artifacts)),
                "transfer_files_checked": len(transfer_run_files(args.artifacts)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
