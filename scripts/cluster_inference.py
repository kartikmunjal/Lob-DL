"""Study 2 A1: stock-day cluster inference over the published comparison family.

Post-hoc exploratory. Reads the frozen Study 1 predictions, recomputes every
published paired comparison at the cluster level, and applies Holm family-wise
adjustment separately to the FI-2010 family and to each LOBSTER label family.

Run on the host that holds ``artifacts/``:

    python scripts/cluster_inference.py --artifacts artifacts --output reports/generated/study2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lobdl.inference import holm_adjust, paired_cluster_inference

METRICS = ("macro_f1", "mcc")
STATUS = "post_hoc_exploratory_study2_A1"


def published_family(paired_root: Path) -> list[dict]:
    """Enumerate the registered paired reports so the family is not re-chosen."""
    family = []
    for path in sorted(paired_root.glob("h*/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        family.append(
            {
                "system_a": report["system_a"],
                "system_b": report["system_b"],
                "horizon": int(report["horizon"]),
                "label_variant": report.get("label_variant"),
                "registered_report": path.as_posix(),
                "registered_macro_f1": report["metric_differences"].get("macro_f1"),
            }
        )
    if not family:
        raise FileNotFoundError(f"no registered paired reports under {paired_root}")
    return family


def load_fi_pair(artifacts: Path, entry: dict):
    from paired_comparison import load_pair, parse_system

    return load_pair(
        artifacts,
        parse_system(entry["system_a"]),
        parse_system(entry["system_b"]),
        entry["horizon"],
    )


def load_transfer_pair(artifacts: Path, entry: dict):
    from paired_transfer import STOCHASTIC, prediction_path

    from lobdl.search import FINAL_SEEDS

    a_name, b_name = entry["system_a"], entry["system_b"]
    seeds = FINAL_SEEDS if a_name in STOCHASTIC or b_name in STOCHASTIC else (None,)
    parts: list[list[np.ndarray]] = [[], [], [], []]
    for seed_index, seed in enumerate(seeds, start=1):
        a = np.load(
            prediction_path(artifacts, a_name, entry["horizon"], seed if a_name in STOCHASTIC else None)
        )
        b = np.load(
            prediction_path(artifacts, b_name, entry["horizon"], seed if b_name in STOCHASTIC else None)
        )
        labels = a[f"labels_{entry['label_variant']}"]
        if not np.array_equal(labels, b[f"labels_{entry['label_variant']}"]):
            raise AssertionError("paired T1 labels are not aligned")
        if not np.array_equal(a["groups"], b["groups"]):
            raise AssertionError("paired T1 ticker groups are not aligned")
        groups = np.asarray(a["groups"]).reshape(len(labels), -1)
        parts[0].append(a["probabilities"])
        parts[1].append(b["probabilities"])
        parts[2].append(labels)
        parts[3].append(np.column_stack((groups, np.full(len(labels), seed_index))))
    return tuple(np.concatenate(part) for part in parts)


def analyse(family: list[dict], loader, artifacts: Path, replicates: int) -> list[dict]:
    rows = []
    for entry in family:
        probabilities_a, probabilities_b, labels, groups = loader(artifacts, entry)
        result = paired_cluster_inference(
            labels,
            probabilities_a,
            probabilities_b,
            groups,
            metrics=METRICS,
            bootstrap_replicates=replicates,
        )
        rows.append({**entry, "cluster_inference": result})
        print(
            f"{entry['label_variant'] or 'fi2010'} h{entry['horizon']} "
            f"{entry['system_a']} - {entry['system_b']}: "
            f"p={result['macro_f1']['sign_flip']['p_value']:.4g}",
            flush=True,
        )
    for metric in METRICS:
        adjusted = holm_adjust(
            np.array([row["cluster_inference"][metric]["sign_flip"]["p_value"] for row in rows])
        )
        for row, value in zip(rows, adjusted, strict=True):
            row["cluster_inference"][metric]["holm_p_value"] = float(value)
            row["cluster_inference"][metric]["holm_family_size"] = len(rows)
    return rows


def flatten(rows: list[dict], dataset: str) -> list[dict]:
    table = []
    for row in rows:
        for metric in METRICS:
            result = row["cluster_inference"][metric]
            boot = result["cluster_bootstrap"]
            registered = row["registered_macro_f1"] if metric == "macro_f1" else None
            table.append(
                {
                    "status": STATUS,
                    "dataset": dataset,
                    "label_variant": row["label_variant"] or "",
                    "horizon": row["horizon"],
                    "system_a": row["system_a"],
                    "system_b": row["system_b"],
                    "metric": metric,
                    "clusters": boot["clusters"],
                    "mean_cluster_difference": boot["mean"],
                    "cluster_ci95_lower": boot["ci_lower"],
                    "cluster_ci95_upper": boot["ci_upper"],
                    "between_cluster_sd": boot["between_cluster_sd"],
                    "clusters_favoring_a": boot["clusters_favoring_a"],
                    "sign_flip_p": result["sign_flip"]["p_value"],
                    "minimum_attainable_p": result["sign_flip"]["minimum_attainable_p"],
                    "holm_p": result["holm_p_value"],
                    "holm_family_size": result["holm_family_size"],
                    "registered_block_difference": (registered or {}).get("difference", ""),
                    "registered_block_ci95_lower": (registered or {}).get("lower", ""),
                    "registered_block_ci95_upper": (registered or {}).get("upper", ""),
                }
            )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/study2"))
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()

    reports = args.artifacts / "reports"
    fi_rows = analyse(
        published_family(reports / "paired"), load_fi_pair, args.artifacts, args.replicates
    )
    transfer_family = published_family(reports / "transfer" / "paired")
    transfer_rows = []
    for variant in sorted({entry["label_variant"] for entry in transfer_family}):
        subset = [entry for entry in transfer_family if entry["label_variant"] == variant]
        transfer_rows.extend(analyse(subset, load_transfer_pair, args.artifacts, args.replicates))

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": STATUS,
        "protocol": "docs/STUDY2_PROTOCOL.md#a1-stock-day-cluster-inference",
        "fi2010": fi_rows,
        "lobster": transfer_rows,
    }
    (args.output / "cluster_inference.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    table = flatten(fi_rows, "FI2010") + flatten(transfer_rows, "LOBSTER")
    with (args.output / "cluster_inference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
