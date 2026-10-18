from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lobdl.metrics import paired_bootstrap_metric_differences

SEEDS = (11, 23, 47, 91, 137)
STOCHASTIC = {
    "B2", "B2_shipped144", "M1", "M1_anchor", "M1_refit", "M1_depth1",
    "M1_depth5", "M2", "M2_depth1", "M2_depth5",
}
M1_ANCHOR = {"learning_rate": 0.01, "epsilon": 1.0, "dropout": 0.0}


def parse_system(value: str) -> tuple[str, str | None]:
    model, separator, variant = value.partition(":")
    if model == "B0" and (not separator or variant not in {"majority", "persistence"}):
        raise ValueError("B0 systems must specify B0:majority or B0:persistence")
    if model not in {
        "B0",
        "B1",
        "B1_shipped144",
        "B2",
        "B2_shipped144",
        "M1",
        "M1_anchor",
        "M1_refit",
        "M1_depth1",
        "M1_depth5",
        "M2",
        "M2_depth1",
        "M2_depth5",
    }:
        raise ValueError(f"unsupported system {value}")
    return model, variant or None


def family_root(root: Path, model: str, horizon: int) -> Path:
    path = root / model / f"h{horizon}"
    if model == "M1_anchor" and not path.exists():
        selection = json.loads(
            (root / "selections" / "M1" / f"h{horizon}.json").read_text(
                encoding="utf-8"
            )
        )
        if selection["selected"]["parameters"] != M1_ANCHOR:
            raise FileNotFoundError("M1 anchor runs are absent and the anchor did not win")
        path = root / "M1" / f"h{horizon}"
    return path


def run_path(
    root: Path,
    model: str,
    variant: str | None,
    horizon: int,
    seed: int,
) -> Path:
    base = family_root(root, model, horizon)
    if model in STOCHASTIC:
        return base / f"seed{seed}"
    if model == "B0":
        return base / str(variant)
    return base / "deterministic"


def load_pair(
    root: Path,
    system_a: tuple[str, str | None],
    system_b: tuple[str, str | None],
    horizon: int,
):
    use_seeds = SEEDS if system_a[0] in STOCHASTIC or system_b[0] in STOCHASTIC else (11,)
    probabilities_a, probabilities_b, labels_all, groups_all = [], [], [], []
    for seed_index, seed in enumerate(use_seeds):
        a = np.load(run_path(root, *system_a, horizon, seed) / "predictions.npz")
        b = np.load(run_path(root, *system_b, horizon, seed) / "predictions.npz")
        if not np.array_equal(a["labels"], b["labels"]):
            raise AssertionError("paired labels are not aligned")
        if not np.array_equal(a["groups"], b["groups"]):
            raise AssertionError("paired stock-day groups are not aligned")
        probabilities_a.append(a["probabilities"])
        probabilities_b.append(b["probabilities"])
        labels_all.append(a["labels"])
        groups_all.append(
            np.column_stack((a["groups"], np.full(len(a["groups"]), seed_index + 1)))
        )
    return tuple(
        np.concatenate(value)
        for value in (probabilities_a, probabilities_b, labels_all, groups_all)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired moving-block model comparison")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--system-a", required=True)
    parser.add_argument("--system-b", required=True)
    parser.add_argument("--horizon", type=int, choices=(10, 20, 50, 100), required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    system_a, system_b = parse_system(args.system_a), parse_system(args.system_b)
    probabilities_a, probabilities_b, labels, groups = load_pair(
        args.artifacts, system_a, system_b, args.horizon
    )
    result = {
        "system_a": args.system_a,
        "system_b": args.system_b,
        "horizon": args.horizon,
        "bootstrap": {
            "type": "paired_moving_block",
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
    name = f"{args.system_a.replace(':', '_')}_minus_{args.system_b.replace(':', '_')}.json"
    path = args.artifacts / "reports" / "paired" / f"h{args.horizon}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
