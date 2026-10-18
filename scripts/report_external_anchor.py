from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the sourced DeepLOB anchor comparison")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/external_anchors.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/generated/external_anchor")
    )
    args = parser.parse_args()
    anchor = yaml.safe_load(args.config.read_text(encoding="utf-8"))["deeplob_setup_2"]
    rows = []
    for horizon, published_percent in anchor["horizons"].items():
        if published_percent is None:
            rows.append(
                {
                    "horizon": int(horizon),
                    "published_f1": None,
                    "study_macro_f1": None,
                    "study_ci95_lower": None,
                    "study_ci95_upper": None,
                    "difference_percentage_points": None,
                    "status": "paper_setup_2_not_reported",
                }
            )
            continue
        path = args.artifacts / "reports" / "M1_anchor" / f"h{horizon}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing registered anchor report: {path}")
        study = json.loads(path.read_text(encoding="utf-8"))
        point = float(study["metrics"]["macro_f1"])
        interval = study["ci95"]["macro_f1"]
        rows.append(
            {
                "horizon": int(horizon),
                "published_f1": float(published_percent) / 100.0,
                "study_macro_f1": point,
                "study_ci95_lower": interval["lower"],
                "study_ci95_upper": interval["upper"],
                "difference_percentage_points": 100.0 * point - float(published_percent),
                "status": "contextual_only_protocols_differ",
            }
        )
    result = {
        "source": anchor["source"],
        "metric_note": "The paper's reported F1 is contextual; the study reports macro-F1.",
        "comparability": anchor["comparability"],
        "protocol_differences": anchor["registered_protocol_differences"],
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
