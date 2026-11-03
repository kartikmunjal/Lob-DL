import numpy as np
import pytest

from scripts.return_eval import evaluate_run, summarize, table_rows


def _synthetic(seed):
    rng = np.random.default_rng(seed)
    rows = 3000
    move = rng.normal(0, 1e-4, rows)
    logits = np.column_stack((-move * 1e4, np.zeros(rows), move * 1e4))
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    groups = np.column_stack((np.repeat([8, 9, 10], 1000), np.tile(np.repeat(np.arange(5), 200), 3)))
    return probabilities, move + rng.normal(0, 5e-5, rows), np.full(rows, 0.5), groups


def test_informative_signal_has_positive_ic_r2_and_monotone_markout():
    runs = []
    for seed in (0, 1):
        probabilities, forward, half, groups = _synthetic(seed)
        runs.append(evaluate_run(probabilities, forward, half, groups, groups[:, 0]))
    summary = summarize(runs)
    assert summary["N_seeds"] == 2
    assert summary["ic"]["clusters"] == 15
    assert summary["ic"]["mean"] > 0.8
    assert summary["oos_r2_seed_mean"] > 0.5
    moves = [row["mean_forward_move_bps"] for row in summary["markout_seed_mean"]]
    assert moves == sorted(moves)
    row = table_rows("FI2010", "toy", 10, summary)[0]
    assert row["status"].startswith("post_hoc")
    assert row["top_decile_taker_edge_bps"] == pytest.approx(
        row["top_decile_move_bps"] - row["top_decile_half_spread_bps"]
    )
