from scripts.generate_study2_assets import markdown, plot_forest, plot_markout, significance_summary


def _cluster_row(dataset, a, b, horizon, p_holm, lower, upper, metric="macro_f1", label=""):
    return {
        "dataset": dataset, "label_variant": label, "horizon": str(horizon),
        "system_a": a, "system_b": b, "metric": metric, "clusters": "15",
        "mean_cluster_difference": "0.01", "cluster_ci95_lower": "0.005",
        "cluster_ci95_upper": "0.02", "minimum_attainable_p": "6.1e-05", "holm_p": str(p_holm),
        "registered_block_ci95_lower": str(lower), "registered_block_ci95_upper": str(upper),
    }


def _return_row(dataset, system, horizon):
    return {
        "dataset": dataset, "system": system, "horizon": str(horizon),
        "mean_cluster_ic": "0.05", "oos_r2_bps": "0.001", "top_decile_move_bps": "1.2",
        "top_decile_half_spread_bps": "3.0", "top_decile_taker_edge_bps": "-1.8",
    }


def test_significance_counts_only_block_significant_comparisons():
    rows = [
        _cluster_row("FI2010", "B2", "M1", 10, 0.001, 0.02, 0.03),   # both significant
        _cluster_row("FI2010", "B2", "M2", 10, 0.40, 0.02, 0.03),    # block yes, Holm no
        _cluster_row("FI2010", "M1", "M2", 10, 0.001, -0.01, 0.01),  # block CI spans 0
        _cluster_row("FI2010", "B2", "M1", 10, 0.001, 0.02, 0.03, metric="mcc"),
    ]
    (summary,) = significance_summary(rows)
    assert summary["comparisons"] == 3
    assert summary["block_ci_excludes_zero"] == 2
    assert summary["also_holm_significant"] == 1
    assert summary["clusters"] == 15


def test_figures_and_markdown_render(tmp_path):
    cluster = [
        _cluster_row("FI2010", a, b, h, 0.01, 0.01, 0.02)
        for h in (10, 20, 50, 100)
        for a, b in (("B2", "M1"), ("B2", "M2"), ("B2", "B0:persistence"), ("M1", "M2"))
    ]
    systems = ["B0 persistence", "B1 logistic", "B2 LightGBM", "M1 DeepLOB", "M2 transformer"]
    returns = [
        _return_row(d, s, h)
        for d in ("FI2010", "LOBSTER") for s in systems for h in (10, 20, 50, 100)
    ]
    plot_forest(cluster, tmp_path / "a.png")
    plot_markout(returns, tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 5000
    assert (tmp_path / "b.png").stat().st_size > 5000
    text = markdown(significance_summary(cluster), returns)
    assert "post-hoc exploratory" in text and "Holm" in text
