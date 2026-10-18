from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(audit_path: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("window") != 100:
        raise ValueError("reconstruction audit does not use the registered 100-row window")
    if audit.get("reconstructed_sensitivity_rule") != "all_top10_cells_exact_for_entire_window":
        raise ValueError("reconstruction audit rule differs from the registered exact rule")
    rows = [
        {
            "ticker": ticker,
            "eligible_exact_100_row_windows": values["eligible_exact_windows"],
            "status": (
                "eligible_for_registered_sensitivity"
                if values["eligible_exact_windows"] > 0
                else "not_estimable_zero_eligible_windows"
            ),
        }
        for ticker, values in audit["tickers"].items()
    ]
    total = sum(row["eligible_exact_100_row_windows"] for row in rows)
    return {
        "analysis": "LOBSTER_reconstructed_book_exact_window_sensitivity",
        "primary_t1_input": audit["primary_t1_input"],
        "eligibility_rule": audit["reconstructed_sensitivity_rule"],
        "window": audit["window"],
        "audit_sha256": sha256(audit_path),
        "tickers": rows,
        "total_eligible_windows": total,
        "result_status": (
            "requires_registered_model_evaluation"
            if total > 0
            else "not_estimable_without_relaxing_preregistered_rule"
        ),
        "metrics": None if total == 0 else "pending_registered_evaluation",
        "selection_N_trials": 0,
        "reported_variant_N_trials": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report eligibility for exact reconstructed-book T1 sensitivity"
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/audit/lobster/lobster_all.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/audits/reconstructed_sensitivity.json"),
    )
    args = parser.parse_args()
    result = build_report(args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
