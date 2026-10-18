import json

from scripts.report_reconstructed_sensitivity import build_report


def test_zero_exact_windows_are_reported_without_relaxing_rule(tmp_path):
    audit = tmp_path / "lobster_all.json"
    audit.write_text(
        json.dumps(
            {
                "primary_t1_input": "vendor_book",
                "reconstructed_sensitivity_rule": "all_top10_cells_exact_for_entire_window",
                "window": 100,
                "tickers": {
                    "AAPL": {"eligible_exact_windows": 0},
                    "MSFT": {"eligible_exact_windows": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    report = build_report(audit)
    assert report["total_eligible_windows"] == 0
    assert report["metrics"] is None
    assert report["result_status"] == "not_estimable_without_relaxing_preregistered_rule"
