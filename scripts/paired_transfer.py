from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.metrics import paired_bootstrap_metric_differences
from lobdl.search import FINAL_SEEDS

SYSTEMS = {"B0_majority", "B0_persistence", "B1", "B2", "M1", "M2"}
STOCHASTIC = {"B2", "M1", "M2"}


def prediction_path(
    root: Path, system: str, horizon: int, seed: int | None
) -> Path:
    run = "deterministic" if system not in STOCHASTIC else f"seed{seed}"
    return root / "transfer" / system / f"h{horizon}" / run / "predictions.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired T1 moving-block comparison")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--system-a", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--system-b", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--labels", choices=("fixed", "matched"), required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.system_a == args.system_b:
        parser.error("paired systems must differ")

    seeds = (
        FINAL_SEEDS
        if args.system_a in STOCHASTIC or args.system_b in STOCHASTIC
        else (None,)
    )
    all_a, all_b, all_labels, all_groups = [], [], [], []
    for seed_index, seed in enumerate(seeds, start=1):
        seed_a = seed if args.system_a in STOCHASTIC else None
        seed_b = seed if args.system_b in STOCHASTIC else None
        a = np.load(prediction_path(args.artifacts, args.system_a, args.horizon, seed_a))
        b = np.load(prediction_path(args.artifacts, args.system_b, args.horizon, seed_b))
        labels_a, labels_b = a[f"labels_{args.labels}"], b[f"labels_{args.labels}"]
        if not np.array_equal(labels_a, labels_b):
            raise AssertionError("paired T1 labels are not aligned")
        if not np.array_equal(a["groups"], b["groups"]):
            raise AssertionError("paired T1 ticker groups are not aligned")
        all_a.append(a["probabilities"])
        all_b.append(b["probabilities"])
        all_labels.append(labels_a)
        all_groups.append(
            np.column_stack(
                (a["groups"], np.full(len(labels_a), seed_index, dtype=np.int16))
            )
        )
    probabilities_a, probabilities_b, labels, groups = (
        np.concatenate(parts) for parts in (all_a, all_b, all_labels, all_groups)
    )
    result = {
        "system_a": args.system_a,
        "system_b": args.system_b,
        "horizon": args.horizon,
        "label_variant": args.labels,
        "bootstrap": {
            "type": "paired_moving_block_within_ticker_and_seed",
            "block_length": 100,
            "replicates": args.replicates,
            "difference_order": "system_a_minus_system_b",
        },
        "metric_differences": paired_bootstrap_metric_differences(
            labels,
            probabilities_a,
            probabilities_b,
            groups,
            replicates=args.replicates,
        ),
    }
    name = f"{args.system_a}_minus_{args.system_b}_{args.labels}.json"
    path = args.artifacts / "reports" / "transfer" / "paired" / f"h{args.horizon}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
