from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HORIZONS = (10, 20, 50, 100)
PRIMARY_SYSTEMS = (
    ("B0 majority", "B0", "majority"),
    ("B0 persistence", "B0", "persistence"),
    ("B1 logistic", "B1", None),
    ("B2 LightGBM", "B2", None),
    ("M1 DeepLOB", "M1", None),
    ("M2 transformer", "M2", None),
)
TRANSFER_NAMES = {
    "B0 majority": "B0_majority",
    "B0 persistence": "B0_persistence",
    "B1 logistic": "B1",
    "B2 LightGBM": "B2",
    "M1 DeepLOB": "M1",
    "M2 transformer": "M2",
}


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing registered report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fi_summary(artifacts: Path, model: str, horizon: int, variant: str | None) -> dict:
    suffix = f"_{variant}" if variant else ""
    return read_json(
        artifacts / "reports" / model / f"h{horizon}" / f"summary{suffix}.json"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(dataset: str, system: str, horizon: int, report: dict) -> dict:
    interval = report["ci95"]["macro_f1"]
    return {
        "dataset": dataset,
        "system": system,
        "horizon": horizon,
        "macro_f1": report["metrics"]["macro_f1"],
        "ci95_lower": interval["lower"],
        "ci95_upper": interval["upper"],
        "seed_sd": report["seed_macro_f1_sd"],
        "selection_N_trials": report["selection_N_trials"],
        "reported_variant_N_trials": report["reported_variant_N_trials"],
        "N_seeds": report["N_seeds"],
    }


def plot_metric_panel(rows: list[dict], path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for system in dict.fromkeys(row["system"] for row in rows):
        selected = [row for row in rows if row["system"] == system]
        x = np.array([row["horizon"] for row in selected])
        y = np.array([row["macro_f1"] for row in selected])
        lower = y - np.array([row["ci95_lower"] for row in selected])
        upper = np.array([row["ci95_upper"] for row in selected]) - y
        axis.errorbar(x, y, yerr=np.vstack((lower, upper)), marker="o", capsize=3, label=system)
    axis.set(xlabel="Horizon (underlying events)", ylabel="Macro-F1", title=title)
    axis.set_xticks(HORIZONS)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_fi(artifacts: Path, output: Path) -> list[dict]:
    rows = []
    for system, model, variant in PRIMARY_SYSTEMS:
        for horizon in HORIZONS:
            rows.append(metric_row("FI-2010", system, horizon, fi_summary(artifacts, model, horizon, variant)))
    write_csv(output / "tables" / "fi_primary_metrics.csv", rows)
    plot_metric_panel(rows, output / "figures" / "fi_primary_macro_f1.png", "FI-2010 untouched test")
    return rows


def generate_transfer(artifacts: Path, output: Path) -> list[dict]:
    rows = []
    for label_variant in ("fixed", "matched"):
        for system, _, _ in PRIMARY_SYSTEMS:
            model = TRANSFER_NAMES[system]
            for horizon in HORIZONS:
                report = read_json(
                    artifacts
                    / "reports"
                    / "transfer"
                    / model
                    / f"h{horizon}"
                    / f"summary_{label_variant}.json"
                )
                row = metric_row(f"LOBSTER_{label_variant}", system, horizon, report)
                rows.append(row)
        panel = [row for row in rows if row["dataset"] == f"LOBSTER_{label_variant}"]
        plot_metric_panel(
            panel,
            output / "figures" / f"lobster_{label_variant}_macro_f1.png",
            f"LOBSTER zero-shot ({label_variant} threshold)",
        )
    write_csv(output / "tables" / "lobster_transfer_metrics.csv", rows)
    return rows


def generate_ablation_table(artifacts: Path, output: Path) -> list[dict]:
    variants = (
        ("B1 primary", "B1"),
        ("B1 shipped-144", "B1_shipped144"),
        ("B1 leaky invalid", "B1_leaky"),
        ("B2 primary", "B2"),
        ("B2 shipped-144", "B2_shipped144"),
        ("M1 depth-10", "M1"),
        ("M1 depth-5", "M1_depth5"),
        ("M1 depth-1", "M1_depth1"),
        ("M1 days-1-7 refit", "M1_refit"),
        ("M2 depth-10", "M2"),
        ("M2 depth-5", "M2_depth5"),
        ("M2 depth-1", "M2_depth1"),
    )
    rows = []
    for label, model in variants:
        for horizon in HORIZONS:
            rows.append(metric_row("FI-2010", label, horizon, fi_summary(artifacts, model, horizon, None)))
    write_csv(output / "tables" / "fi_ablations.csv", rows)
    return rows


def generate_economics(artifacts: Path, output: Path) -> list[dict]:
    rows = []
    for label_variant in ("fixed", "matched"):
        for system, _, _ in PRIMARY_SYSTEMS:
            for horizon in HORIZONS:
                report = read_json(
                    artifacts
                    / "reports"
                    / "transfer"
                    / TRANSFER_NAMES[system]
                    / f"h{horizon}"
                    / f"summary_{label_variant}.json"
                )
                scenarios = report["economics_no_strategy_or_return_claim"]["fee_scenarios_bps"]
                for fee, values in scenarios.items():
                    rows.append(
                        {
                            "label_variant": label_variant,
                            "system": system,
                            "horizon": horizon,
                            "round_trip_fee_bps": fee,
                            "correct_direction_windows": values["correct_direction_windows"],
                            "mean_absolute_forward_move_dollars": values["mean_absolute_forward_move"],
                            "mean_spread_plus_fee_dollars": values["mean_spread_plus_fee"],
                            "mean_move_minus_cost_dollars": values["mean_move_minus_cost"],
                            "fraction_move_exceeds_cost": values["fraction_move_exceeds_cost"],
                            "move_minus_cost_ci95_lower": values["ci95"]["mean_move_minus_cost"]["lower"],
                            "move_minus_cost_ci95_upper": values["ci95"]["mean_move_minus_cost"]["upper"],
                            "selection_N_trials": report["selection_N_trials"],
                            "reported_variant_N_trials": report["reported_variant_N_trials"],
                        }
                    )
    write_csv(output / "tables" / "lobster_economic_significance.csv", rows)
    primary = [row for row in rows if row["round_trip_fee_bps"] == "1.0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for axis, label_variant in zip(axes, ("fixed", "matched"), strict=True):
        selected = [row for row in primary if row["label_variant"] == label_variant]
        for system in ("M1 DeepLOB", "M2 transformer"):
            values = [row for row in selected if row["system"] == system]
            axis.plot(
                [row["horizon"] for row in values],
                [row["mean_move_minus_cost_dollars"] for row in values],
                marker="o",
                label=system,
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(f"{label_variant} threshold")
        axis.set_xlabel("Horizon")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean absolute move minus spread and 1 bp ($)")
    axes[1].legend()
    fig.suptitle("Descriptive economic significance; no strategy or return claim")
    fig.tight_layout()
    fig.savefig(output / "figures" / "inside_spread_primary_fee.png", dpi=180)
    plt.close(fig)
    return rows


def generate_reliability(artifacts: Path, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for axis, horizon in zip(axes.flat, HORIZONS, strict=True):
        for label, model in (("M1 DeepLOB", "M1"), ("M2 transformer", "M2")):
            report = fi_summary(artifacts, model, horizon, None)
            curve = report["metrics"]["reliability_curve"]
            observed = [item for item in curve if item["count"] > 0]
            axis.plot(
                [item["mean_confidence"] for item in observed],
                [item["accuracy"] for item in observed],
                marker="o",
                label=label,
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set_title(f"h={horizon}")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.supxlabel("Mean confidence")
    fig.supylabel("Empirical accuracy")
    fig.suptitle("FI-2010 top-label reliability")
    fig.tight_layout()
    fig.savefig(output / "figures" / "fi_reliability.png", dpi=180)
    plt.close(fig)


def generate_compute(artifacts: Path, output: Path) -> list[dict]:
    rows = []
    for system, model, variant in PRIMARY_SYSTEMS:
        for horizon in HORIZONS:
            report = fi_summary(artifacts, model, horizon, variant)
            rows.append(
                {
                    "system": system,
                    "horizon": horizon,
                    "parameters": (
                        "not_applicable"
                        if model == "B0"
                        else read_json(
                            artifacts
                            / model
                            / f"h{horizon}"
                            / ("deterministic" if model == "B1" else "seed11")
                            / "report.json"
                        ).get("trainable_parameters", "fitted_model_specific")
                    ),
                    "total_compute_hours": report["compute_hours"],
                    "N_seeds": report["N_seeds"],
                    "selection_N_trials": report["selection_N_trials"],
                }
            )
    write_csv(output / "tables" / "compute_and_parameters.csv", rows)
    return rows


def generate_audits(artifacts: Path, output: Path) -> None:
    fi = read_json(artifacts / "audit" / "fi2010.json")
    fi_rows = [
        {
            "horizon": int(horizon),
            "formula_agree": values["agree"],
            "eligible_rows": values["n"],
            "agreement_rate": values["rate"],
        }
        for horizon, values in fi["label_agreement"].items()
        if int(horizon) in HORIZONS
    ]
    write_csv(output / "tables" / "fi_label_formula_agreement.csv", fi_rows)

    lobster = read_json(artifacts / "audit" / "lobster" / "lobster_all.json")
    overall, time_rows = [], []
    for ticker, values in lobster["tickers"].items():
        counts = values["event_counts"]
        exact = values["exact_match_rates"]
        first = values["first_top_10_mismatch"] or {}
        overall.append(
            {
                "ticker": ticker,
                "replayed_messages": values["replayed_messages"],
                "level_1_exact_rate": exact["level_1"],
                "top_5_exact_rate": exact["top_5"],
                "top_10_exact_rate": exact["top_10"],
                "first_mismatch_message": first.get("message_index"),
                "first_mismatch_seconds_since_midnight": first.get("seconds_since_midnight"),
                "eligible_exact_100_row_windows": values["eligible_exact_windows"],
                "unattributed_applied_events": counts["unattributed_applied_events"],
                "zero_clamps": counts["zero_clamps"],
                "absent_level_skips": counts["absent_level_skips"],
                "hidden_executions": counts["hidden_executions"],
            }
        )
        for item in values["time_bins"]:
            time_rows.append({"ticker": ticker, **item})
    write_csv(output / "tables" / "lobster_reconstruction_summary.csv", overall)
    write_csv(output / "tables" / "lobster_reconstruction_by_time.csv", time_rows)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for ticker in lobster["tickers"]:
        values = [row for row in time_rows if row["ticker"] == ticker]
        axis.plot(
            [row["minutes_since_open_start"] for row in values],
            [row["top_10_exact_rate"] for row in values],
            marker="o",
            label=ticker,
        )
    axis.set(
        xlabel="Minutes since 09:30",
        ylabel="Exact top-10 replay rate",
        title="Pure replay mismatch accumulation without reseeding",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "figures" / "lobster_replay_exact_rate.png", dpi=180)
    plt.close(fig)


def markdown_summary(fi_rows: list[dict], transfer_rows: list[dict], output: Path) -> None:
    lines = [
        "# Generated results summary",
        "",
        "Generated only by `scripts/generate_report_assets.py` from registered reports.",
        "",
        "## FI-2010 primary macro-F1",
        "",
        "| System | Horizon | Macro-F1 | 95% CI | N_trials | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in fi_rows:
        lines.append(
            f"| {row['system']} | {row['horizon']} | {row['macro_f1']:.4f} | "
            f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] | "
            f"{row['selection_N_trials']}+{row['reported_variant_N_trials']} | {row['N_seeds']} |"
        )
    lines.extend(
        [
            "",
            "## LOBSTER zero-shot macro-F1",
            "",
            "| Labels | System | Horizon | Macro-F1 | 95% CI | N_trials | Seeds |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in transfer_rows:
        lines.append(
            f"| {row['dataset']} | {row['system']} | {row['horizon']} | "
            f"{row['macro_f1']:.4f} | [{row['ci95_lower']:.4f}, "
            f"{row['ci95_upper']:.4f}] | {row['selection_N_trials']}+"
            f"{row['reported_variant_N_trials']} | {row['N_seeds']} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate every public table and figure")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated"))
    args = parser.parse_args()
    (args.output / "figures").mkdir(parents=True, exist_ok=True)
    (args.output / "tables").mkdir(parents=True, exist_ok=True)
    fi_rows = generate_fi(args.artifacts, args.output)
    transfer_rows = generate_transfer(args.artifacts, args.output)
    generate_ablation_table(args.artifacts, args.output)
    generate_economics(args.artifacts, args.output)
    generate_reliability(args.artifacts, args.output)
    generate_compute(args.artifacts, args.output)
    generate_audits(args.artifacts, args.output)
    markdown_summary(fi_rows, transfer_rows, args.output)
    print(json.dumps({"output": str(args.output), "figures": 6, "tables": 8}, indent=2))


if __name__ == "__main__":
    main()
