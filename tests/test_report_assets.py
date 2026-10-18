import json
from pathlib import Path

from scripts.generate_report_assets import (
    HORIZONS,
    PRIMARY_SYSTEMS,
    TRANSFER_NAMES,
    generate_ablation_table,
    generate_compute,
    generate_economics,
    generate_fi,
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
    return {
        "metrics": {"macro_f1": 0.5, "reliability_curve": curve},
        "ci95": {"macro_f1": {"lower": 0.4, "upper": 0.6}},
        "seed_macro_f1_sd": 0.01,
        "selection_N_trials": 12,
        "reported_variant_N_trials": 0,
        "N_seeds": 5,
        "compute_hours": 1.0,
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


def test_all_public_assets_generate_from_named_reports(tmp_path):
    artifacts, output = tmp_path / "artifacts", tmp_path / "generated"
    variants = {
        "B1_shipped144",
        "B1_leaky",
        "B2_shipped144",
        "M1_depth1",
        "M1_depth5",
        "M1_refit",
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
    (output / "figures").mkdir(parents=True)
    fi = generate_fi(artifacts, output)
    transfer = generate_transfer(artifacts, output)
    generate_ablation_table(artifacts, output)
    generate_economics(artifacts, output)
    generate_reliability(artifacts, output)
    generate_compute(artifacts, output)
    markdown_summary(fi, transfer, output)
    assert (output / "RESULTS.md").exists()
    assert len(list((output / "figures").glob("*.png"))) == 5
    assert len(list((output / "tables").glob("*.csv"))) == 5
