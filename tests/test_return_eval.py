import numpy as np
import pytest
from scipy.stats import spearmanr

from lobdl.return_eval import (
    cluster_ic,
    crossfit_markout,
    crossfit_oos_r2,
    directional_signal,
    spearman_ic,
)


def test_directional_signal_uses_down_flat_up_order():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2]])
    assert directional_signal(p).tolist() == pytest.approx([-0.6, 0.7, 0.0])
    with pytest.raises(ValueError):
        directional_signal(np.ones((2, 2)))


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(-1, 2, 200).astype(float)  # heavy ties, like one-hot baselines
    y = s + rng.normal(0, 2, 200)
    assert spearman_ic(s, y) == pytest.approx(spearmanr(s, y).statistic)
    assert np.isnan(spearman_ic(np.zeros(10), y[:10]))


def test_cluster_ic_separates_clusters():
    s = np.r_[np.arange(50.0), np.arange(50.0)]
    y = np.r_[np.arange(50.0), -np.arange(50.0)]
    ids, ics = cluster_ic(s, y, np.repeat([[8, 0], [8, 1]], 50, axis=0))
    assert ids.tolist() == [[8, 0], [8, 1]]
    assert ics.tolist() == pytest.approx([1.0, -1.0])


def test_crossfit_r2_recovers_signal_and_penalizes_noise():
    rng = np.random.default_rng(1)
    s = rng.normal(size=3000)
    folds = np.repeat([8, 9, 10], 1000)
    y = 2.0 * s + rng.normal(0, 1, 3000)
    good = crossfit_oos_r2(s, y, folds)
    assert good["oos_r2"] == pytest.approx(0.8, abs=0.03)
    assert all(b == pytest.approx(2.0, abs=0.1) for b in good["per_fold_beta"].values())
    noise = crossfit_oos_r2(rng.normal(size=3000), y, folds)
    assert noise["oos_r2"] < 0.01


def test_crossfit_never_uses_scored_fold():
    # Fold 9 has a reversed relation; a leaky in-sample fit would partly absorb it.
    s = np.tile(np.linspace(-1, 1, 100), 3)
    y = np.r_[s[:100], -s[100:200], s[200:]]
    folds = np.repeat([8, 9, 10], 100)
    result = crossfit_oos_r2(s, y, folds)
    assert result["per_fold_beta"]["9"] == pytest.approx(1.0)
    assert result["per_fold_oos_r2"]["9"] == pytest.approx(1 - 4.0)


def test_markout_monotone_and_costs_reported():
    rng = np.random.default_rng(2)
    s = rng.normal(size=5000)
    y = 3.0 * s
    half = np.full(5000, 1.0)
    curve = crossfit_markout(s, y, half, np.repeat([1, 2], 2500), bins=5)
    moves = [row["mean_forward_move_bps"] for row in curve]
    assert moves == sorted(moves)
    assert sum(row["count"] for row in curve) == 5000
    assert curve[-1]["taker_edge_bps"] == pytest.approx(curve[-1]["mean_forward_move_bps"] - 1.0)
    assert curve[0]["taker_edge_bps"] == pytest.approx(-curve[0]["mean_forward_move_bps"] - 1.0)
