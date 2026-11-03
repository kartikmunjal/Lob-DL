"""Study 2 A2: return-unit evaluation of frozen Study 1 predictions.

Post-hoc exploratory. For every primary FI-2010 system and every LOBSTER transfer
system, compute per-cluster Spearman IC, cross-fitted out-of-sample R^2 in bps,
and cross-fitted decile mark-out curves versus the half-spread. Folds are test
days on FI-2010 and tickers on LOBSTER. Metrics are computed per fitted seed and
then averaged over seeds.

    python scripts/return_eval.py --artifacts artifacts --output reports/generated/study2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lobdl.inference import cluster_bootstrap_interval
from lobdl.return_eval import cluster_ic, crossfit_markout, crossfit_oos_r2, directional_signal

STATUS = "post_hoc_exploratory_study2_A2"
HORIZONS = (10, 20, 50, 100)
SEEDS = (11, 23, 47, 91, 137)
FI_SYSTEMS = {
    "B0 persistence": ("B0", "persistence"),
    "B1 logistic": ("B1", None),
    "B2 LightGBM": ("B2", None),
    "M1 DeepLOB": ("M1", None),
    "M2 transformer": ("M2", None),
}
TRANSFER_SYSTEMS = {
    "B0 persistence": "B0_persistence",
    "B1 logistic": "B1",
    "B2 LightGBM": "B2",
    "M1 DeepLOB": "M1",
    "M2 transformer": "M2",
}
TRANSFER_STOCHASTIC = {"B2", "M1", "M2"}


def evaluate_run(
    probabilities: np.ndarray,
    forward_returns: np.ndarray,
    half_spread_bps: np.ndarray,
    clusters: np.ndarray,
    folds: np.ndarray,
) -> dict:
    signal = directional_signal(probabilities)
    target_bps = np.asarray(forward_returns, dtype=np.float64) * 10_000.0
    ids, ics = cluster_ic(signal, target_bps, clusters)
    return {
        "cluster_ids": ids.tolist(),
        "cluster_ic": ics.tolist(),
        **crossfit_oos_r2(signal, target_bps, folds),
        "markout": crossfit_markout(signal, target_bps, half_spread_bps, folds),
    }


def summarize(runs: list[dict]) -> dict:
    """Average seed-level results; cluster IC uses the seed-mean per cluster."""
    ics = np.nanmean(np.array([run["cluster_ic"] for run in runs]), axis=0)
    finite = ics[np.isfinite(ics)]
    ic_summary = (
        cluster_bootstrap_interval(finite)
        if len(finite) >= 2
        else {"mean": float("nan"), "clusters": len(finite)}
    )
    bins = len(runs[0]["markout"])
    markout = []
    for index in range(bins):
        cells = [run["markout"][index] for run in runs if run["markout"][index]["count"]]
        keys = [k for k in (cells[0] if cells else {}) if k not in {"bin", "count"}]
        markout.append(
            {
                "bin": index + 1,
                "count_mean": float(np.mean([c["count"] for c in cells])) if cells else 0.0,
                **{k: float(np.nanmean([c[k] for c in cells])) for k in keys},
            }
        )
    return {
        "N_seeds": len(runs),
        "cluster_ids": runs[0]["cluster_ids"],
        "cluster_ic_seed_mean": ics.tolist(),
        "ic": ic_summary,
        "oos_r2_seed_mean": float(np.mean([run["oos_r2"] for run in runs])),
        "oos_r2_seed_sd": float(np.std([run["oos_r2"] for run in runs], ddof=0)),
        "markout_seed_mean": markout,
    }


def fi_runs(artifacts: Path, model: str, variant: str | None, horizon: int) -> list[dict]:
    from paired_comparison import STOCHASTIC, run_path

    seeds = SEEDS if model in STOCHASTIC else (11,)
    runs = []
    for seed in seeds:
        data = np.load(run_path(artifacts, model, variant, horizon, seed) / "predictions.npz")
        groups = np.asarray(data["groups"])
        runs.append(
            evaluate_run(
                data["probabilities"],
                data["forward_returns"],
                np.asarray(data["relative_spreads"]) * 10_000.0 / 2.0,
                clusters=groups,
                folds=groups[:, 0],
            )
        )
    return runs


def transfer_runs(artifacts: Path, system: str, horizon: int) -> list[dict]:
    from paired_transfer import prediction_path

    seeds = SEEDS if system in TRANSFER_STOCHASTIC else (None,)
    runs = []
    for seed in seeds:
        data = np.load(prediction_path(artifacts, system, horizon, seed))
        groups = np.asarray(data["groups"]).reshape(len(data["probabilities"]), -1)
        mid = np.asarray(data["mid"], dtype=np.float64)
        half_bps = np.asarray(data["spread"]) / np.maximum(mid, 1e-12) * 10_000.0 / 2.0
        runs.append(
            evaluate_run(
                data["probabilities"],
                data["forward_returns"],
                half_bps,
                clusters=groups[:, :1],
                folds=groups[:, 0],
            )
        )
    return runs


def table_rows(dataset: str, name: str, horizon: int, summary: dict) -> list[dict]:
    extremes = [summary["markout_seed_mean"][0], summary["markout_seed_mean"][-1]]
    return [
        {
            "status": STATUS,
            "dataset": dataset,
            "system": name,
            "horizon": horizon,
            "N_seeds": summary["N_seeds"],
            "clusters": summary["ic"].get("clusters"),
            "mean_cluster_ic": summary["ic"].get("mean"),
            "ic_cluster_ci95_lower": summary["ic"].get("ci_lower", ""),
            "ic_cluster_ci95_upper": summary["ic"].get("ci_upper", ""),
            "clusters_positive_ic": summary["ic"].get("clusters_favoring_a", ""),
            "oos_r2_bps": summary["oos_r2_seed_mean"],
            "oos_r2_seed_sd": summary["oos_r2_seed_sd"],
            "bottom_decile_move_bps": extremes[0].get("mean_forward_move_bps", ""),
            "top_decile_move_bps": extremes[1].get("mean_forward_move_bps", ""),
            "top_decile_half_spread_bps": extremes[1].get("mean_half_spread_bps", ""),
            "top_decile_taker_edge_bps": extremes[1].get("taker_edge_bps", ""),
            "bottom_decile_taker_edge_bps": extremes[0].get("taker_edge_bps", ""),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/study2"))
    args = parser.parse_args()

    payload: dict = {"status": STATUS, "protocol": "docs/STUDY2_PROTOCOL.md", "results": []}
    table: list[dict] = []
    for horizon in HORIZONS:
        for name, (model, variant) in FI_SYSTEMS.items():
            summary = summarize(fi_runs(args.artifacts, model, variant, horizon))
            payload["results"].append(
                {"dataset": "FI2010", "system": name, "horizon": horizon, **summary}
            )
            table += table_rows("FI2010", name, horizon, summary)
            print(f"FI2010 h{horizon} {name}: IC={summary['ic']['mean']:.4f}", flush=True)
        for name, system in TRANSFER_SYSTEMS.items():
            summary = summarize(transfer_runs(args.artifacts, system, horizon))
            payload["results"].append(
                {"dataset": "LOBSTER", "system": name, "horizon": horizon, **summary}
            )
            table += table_rows("LOBSTER", name, horizon, summary)
            print(f"LOBSTER h{horizon} {name}: IC={summary['ic']['mean']:.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "return_eval.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "return_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
