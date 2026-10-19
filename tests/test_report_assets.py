import json
from pathlib import Path

from scripts.generate_report_assets import (
    HORIZONS,
    PRIMARY_SYSTEMS,
    TRANSFER_NAMES,
    generate_ablation_table,
    generate_audits,
    generate_classification_metrics,
    generate_compute,
    generate_economics,
    generate_fi,
    generate_fi_economics,
    generate_mechanisms,
    generate_paired_comparisons,
    generate_reliability,
    generate_transfer,
    markdown_summary,
)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def summary() -> dict:
    curve = [
        {
            "bin": index,
            "count": 1,
            "mean_confidence": index / 15,
            "accuracy": index / 15,
        }
        for index in range(1, 16)
    ]
    economic = {
        "correct_direction_windows": 10,
        "mean_absolute_forward_move_bps": 2.0,
        "mean_spread_plus_fee_bps": 3.0,
        "mean_move_minus_cost_bps": -1.0,
        "fraction_move_exceeds_cost": 0.2,
        "ci95": {"mean_move_minus_cost_bps": {"lower": -2.0, "upper": 0.0}},
    }
    vector = [0.4, 0.5, 0.6]
    vector_ci = {"lower": [0.3, 0.4, 0.5], "upper": [0.5, 0.6, 0.7]}
    scalar_ci = {"lower": 0.4, "upper": 0.6}
    return {
        "metrics": {
            "precision": vector,
            "recall": vector,
            "f1": vector,
            "macro_f1": 0.5,
            "mcc": 0.5,
            "ece": 0.1,
            "reliability_curve": curve,
        },
        "ci95": {
            "precision": vector_ci,
            "recall": vector_ci,
            "f1": vector_ci,
            "macro_f1": scalar_ci,
            "mcc": scalar_ci,
            "ece": {"lower": 0.05, "upper": 0.15},
        },
        "seed_macro_f1_sd": 0.01,
        "selection_N_trials": 12,
        "reported_variant_N_trials": 0,
        "N_seeds": 5,
        "compute_hours": 1.0,
        "economics_no_strategy_or_return_claim": {
            "fee_scenarios_bps": {str(fee): economic for fee in (0.0, 0.5, 1.0, 2.0)}
        },
    }


def transfer_summary() -> dict:
    value = summary()
    economic = {
        "correct_direction_windows": 10,
        "mean_absolute_forward_move": 0.02,
        "mean_spread_plus_fee": 0.03,
        "mean_move_minus_cost": -0.01,
        "fraction_move_exceeds_cost": 0.2,
        "ci95": {"mean_move_minus_cost": {"lower": -0.02, "upper": 0.0}},
    }
    value["economics_no_strategy_or_return_claim"] = {
        "fee_scenarios_bps": {str(fee): economic for fee in (0.0, 0.5, 1.0, 2.0)}
    }
    return value


def paired_summary(system_a: str, system_b: str, horizon: int) -> dict:
    scalar = {"difference": 0.01, "lower": -0.01, "upper": 0.03}
    vector = {
        "difference": [0.01, 0.0, -0.01],
        "lower": [-0.01, -0.02, -0.03],
        "upper": [0.03, 0.02, 0.01],
    }
    return {
        "system_a": system_a,
        "system_b": system_b,
        "system_a_N_trials": {
            "selection_N_trials": 12,
            "reported_variant_N_trials": 0,
        },
        "system_b_N_trials": {
            "selection_N_trials": 12,
            "reported_variant_N_trials": 0,
        },
        "horizon": horizon,
        "bootstrap": {"replicates": 2000, "block_length": 100},
        "metric_differences": {
            "precision": vector,
            "recall": vector,
            "f1": vector,
            "macro_f1": scalar,
            "mcc": scalar,
            "ece": scalar,
        },
    }


def test_all_public_assets_generate_from_named_reports(tmp_path):
    artifacts, output = tmp_path / "artifacts", tmp_path / "generated"
    variants = {
        "B1_shipped144",
        "B1_leaky",
        "B2_shipped144",
        "M1_depth1",
        "M1_depth5",
        "M1_refit",
        "M1_anchor",
        "M2_depth1",
        "M2_depth5",
    }
    for _, model, variant in PRIMARY_SYSTEMS:
        for horizon in HORIZONS:
            suffix = f"_{variant}" if variant else ""
            write(
                artifacts / "reports" / model / f"h{horizon}" / f"summary{suffix}.json",
                summary(),
            )
            if model != "B0":
                run = "deterministic" if model == "B1" else "seed11"
                write(
                    artifacts / model / f"h{horizon}" / run / "report.json",
                    {"trainable_parameters": 100},
                )
    for model in variants:
        for horizon in HORIZONS:
            write(artifacts / "reports" / model / f"h{horizon}" / "summary.json", summary())
    for labels in ("fixed", "matched"):
        for system, _, _ in PRIMARY_SYSTEMS:
            for horizon in HORIZONS:
                write(
                    artifacts
                    / "reports"
                    / "transfer"
                    / TRANSFER_NAMES[system]
                    / f"h{horizon}"
                    / f"summary_{labels}.json",
                    transfer_summary(),
                )
    write(
        artifacts / "audit" / "fi2010.json",
        {
            "label_agreement": {
                str(horizon): {"agree": 8, "n": 10, "rate": 0.8}
                for horizon in HORIZONS
            }
        },
    )
    ticker_audit = {
        "replayed_messages": 10,
        "exact_match_rates": {"level_1": 0.5, "top_5": 0.4, "top_10": 0.3},
        "first_top_10_mismatch": {"message_index": 2, "seconds_since_midnight": 34201.0},
        "eligible_exact_windows": 0,
        "event_counts": {
            "unattributed_applied_events": 1,
            "zero_clamps": 1,
            "absent_level_skips": 1,
            "hidden_executions": 1,
        },
        "time_bins": [
            {
                "minutes_since_open_start": 0,
                "minutes_since_open_end": 30,
                "rows": 10,
                "level_1_exact_rate": 0.5,
                "top_5_exact_rate": 0.4,
                "top_10_exact_rate": 0.3,
            }
        ],
    }
    write(
        artifacts / "audit" / "lobster" / "lobster_all.json",
        {"tickers": {ticker: ticker_audit for ticker in ("AAPL", "AMZN")}},
    )
    for model_a, model_b in (("B2", "B1"), ("M1", "B2"), ("M2", "M1")):
        for horizon in HORIZONS:
            mechanism_rows = []
            for state in (
                "relative_spread",
                "volume_imbalance_10",
                "realized_volatility_50",
            ):
                for state_bin in range(1, 5):
                    for true_class in ("all", "down", "stationary", "up"):
                        mechanism_rows.append(
                            {
                                "state": state,
                                "bin": state_bin,
                                "true_class": true_class,
                                "windows": 10,
                                "mean_correctness_difference": 0.01,
                                "ci95": {"lower": -0.01, "upper": 0.03},
                            }
                        )
            write(
                artifacts
                / "reports"
                / "mechanisms"
                / f"h{horizon}"
                / f"{model_a}_minus_{model_b}.json",
                {
                    "system_a_N_trials": {"selection_N_trials": 12},
                    "system_b_N_trials": {"selection_N_trials": 12},
                    "bootstrap": {"replicates": 2000},
                    "rows": mechanism_rows,
                },
            )
    report_systems = (
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
    for horizon in HORIZONS:
        for left in range(len(report_systems)):
            for right in range(left + 1, len(report_systems)):
                system_a, system_b = report_systems[left], report_systems[right]
                name = f"{system_a.replace(':', '_')}_minus_{system_b.replace(':', '_')}.json"
                write(
                    artifacts / "reports" / "paired" / f"h{horizon}" / name,
                    paired_summary(system_a, system_b, horizon),
                )
        for system_a, system_b in (
            ("M1", "M1_refit"),
            ("M1", "M1_depth1"),
            ("M1", "M1_depth5"),
            ("M2", "M2_depth1"),
            ("M2", "M2_depth5"),
        ):
            write(
                artifacts
                / "reports"
                / "paired"
                / f"h{horizon}"
                / f"{system_a}_minus_{system_b}.json",
                paired_summary(system_a, system_b, horizon),
            )
        transfer_systems = tuple(TRANSFER_NAMES.values())
        for labels in ("fixed", "matched"):
            for left in range(len(transfer_systems)):
                for right in range(left + 1, len(transfer_systems)):
                    system_a, system_b = transfer_systems[left], transfer_systems[right]
                    write(
                        artifacts
                        / "reports"
                        / "transfer"
                        / "paired"
                        / f"h{horizon}"
                        / f"{system_a}_minus_{system_b}_{labels}.json",
                        paired_summary(system_a, system_b, horizon),
                    )
    (output / "figures").mkdir(parents=True)
    fi = generate_fi(artifacts, output)
    transfer = generate_transfer(artifacts, output)
    generate_ablation_table(artifacts, output)
    generate_economics(artifacts, output)
    generate_fi_economics(artifacts, output)
    generate_reliability(artifacts, output)
    mechanisms = generate_mechanisms(artifacts, output)
    paired = generate_paired_comparisons(artifacts, output)
    classification = generate_classification_metrics(artifacts, output)
    generate_compute(artifacts, output)
    generate_audits(artifacts, output)
    markdown_summary(fi, transfer, mechanisms, output)
    assert (output / "RESULTS.md").exists()
    assert len(list((output / "figures").glob("*.png"))) == 8
    assert len(list((output / "tables").glob("*.csv"))) == 12
    assert len(paired) == 3408  # 284 paired reports times 12 metric/class rows.
    assert len(classification) == 1296  # 108 summary reports times 12 rows.
