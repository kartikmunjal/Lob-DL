import itertools

import numpy as np
import pytest

from lobdl.inference import (
    cluster_bootstrap_interval,
    cluster_metric_differences,
    holm_adjust,
    paired_cluster_inference,
    sign_flip_pvalue,
)
from lobdl.metrics import classification_metrics


def _one_hot(classes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(classes), 3))
    out[np.arange(len(classes)), classes] = 1.0
    return out


def test_cluster_differences_match_direct_metric_per_cluster_and_seed_mean():
    rng = np.random.default_rng(0)
    rows = 600
    y = rng.integers(0, 3, rows)
    a = rng.dirichlet(np.ones(3), rows)
    b = rng.dirichlet(np.ones(3), rows)
    cluster = np.repeat([[8, 0], [8, 1], [9, 0]], rows // 3, axis=0)
    seed = np.tile(np.repeat([1, 2], rows // 6), 3)
    groups = np.column_stack((cluster, seed))
    ids, diffs = cluster_metric_differences(y, a, b, groups)
    assert ids.tolist() == [[8, 0], [8, 1], [9, 0]]
    for index, key in enumerate(ids):
        cells = []
        for s in (1, 2):
            mask = np.all(cluster == key, axis=1) & (seed == s)
            cells.append(
                classification_metrics(y[mask], a[mask])["macro_f1"]
                - classification_metrics(y[mask], b[mask])["macro_f1"]
            )
        assert diffs[index] == pytest.approx(np.mean(cells))


def test_exact_sign_flip_matches_brute_force_enumeration():
    values = np.array([0.3, -0.1, 0.2, 0.05, 0.4])
    observed = abs(values.mean())
    brute = np.mean(
        [abs(np.dot(signs, values)) / 5 >= observed - 1e-12
         for signs in itertools.product((1, -1), repeat=5)]
    )
    result = sign_flip_pvalue(values)
    assert result["method"] == "exact_sign_flip"
    assert result["p_value"] == pytest.approx(brute)
    assert result["minimum_attainable_p"] == pytest.approx(2 / 32)


def test_all_positive_clusters_reach_minimum_attainable_p():
    result = sign_flip_pvalue(np.full(15, 0.01))
    assert result["assignments"] == 2**15
    assert result["p_value"] == pytest.approx(2 / 2**15)


def test_five_clusters_cannot_reject_at_five_percent():
    # LOBSTER transfer has five tickers: even a unanimous effect gives p = 1/16.
    assert sign_flip_pvalue(np.full(5, 0.2))["p_value"] == pytest.approx(0.0625)


def test_monte_carlo_sign_flip_for_many_clusters():
    rng = np.random.default_rng(1)
    result = sign_flip_pvalue(rng.normal(0, 1, 40), monte_carlo_replicates=5000)
    assert result["method"] == "monte_carlo_sign_flip"
    assert 0 < result["p_value"] <= 1


def test_null_sign_flip_is_roughly_uniform():
    rng = np.random.default_rng(2)
    p = np.array([sign_flip_pvalue(rng.normal(0, 1, 10))["p_value"] for _ in range(400)])
    assert 0.02 < np.mean(p < 0.05) < 0.09


def test_cluster_bootstrap_brackets_mean_and_counts_signs():
    values = np.array([0.01, 0.02, -0.005, 0.03, 0.015])
    result = cluster_bootstrap_interval(values, replicates=4000)
    assert result["ci_lower"] < result["mean"] < result["ci_upper"]
    assert result["clusters_favoring_a"] == 4
    with pytest.raises(ValueError):
        cluster_bootstrap_interval(np.array([0.1]))


def test_holm_matches_hand_computation_and_preserves_order():
    p = np.array([0.04, 0.01, 0.03, 0.20])
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08->0.09, 0.20*1=0.20
    assert holm_adjust(p).tolist() == pytest.approx([0.09, 0.04, 0.09, 0.20])
    assert holm_adjust(np.array([0.9, 0.8])).max() <= 1.0


def test_identical_systems_give_zero_differences_and_unit_p():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 300)
    prob = _one_hot(rng.integers(0, 3, 300))
    groups = np.column_stack((np.repeat(np.arange(6), 50), np.ones(300, dtype=int)))
    result = paired_cluster_inference(y, prob, prob, groups, bootstrap_replicates=500)
    assert np.allclose(result["macro_f1"]["cluster_differences"], 0.0)
    assert result["macro_f1"]["sign_flip"]["p_value"] == pytest.approx(1.0)


def test_cluster_metric_by_seed_matches_paired_differences():
    from lobdl.inference import cluster_metric_by_seed

    rng = np.random.default_rng(5)
    rows = 900
    y = rng.integers(0, 3, rows)
    a, b = rng.dirichlet(np.ones(3), rows), rng.dirichlet(np.ones(3), rows)
    cluster = np.repeat([[8, 0], [8, 1], [9, 0]], rows // 3, axis=0)
    ids, va = cluster_metric_by_seed([y, y], [a, a], [cluster, cluster])
    _, vb = cluster_metric_by_seed([y, y], [b, b], [cluster, cluster])
    groups = np.column_stack((cluster, np.ones(rows, dtype=int)))
    _, diffs = cluster_metric_differences(y, a, b, groups)
    assert ids.tolist() == [[8, 0], [8, 1], [9, 0]]
    assert (va - vb) == pytest.approx(diffs)
    with pytest.raises(ValueError):
        cluster_metric_by_seed([y], [a], [cluster[:-3]])
    other = cluster.copy()
    other[0] = [7, 7]
    with pytest.raises(ValueError, match="identical clusters"):
        cluster_metric_by_seed([y, y], [a, a], [cluster, other])
