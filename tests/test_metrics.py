import numpy as np

from lobdl.metrics import (
    bootstrap_metrics,
    bootstrap_sufficient_sums,
    classification_metrics,
    classification_row_statistics,
    metrics_from_sufficient_sums,
    moving_block_indices,
    paired_bootstrap_metric_differences,
    reliability_curve,
)


def test_reliability_curve_preserves_empty_bins_and_all_observations():
    y = np.array([0, 1, 1])
    probabilities = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]])
    curve = reliability_curve(y, probabilities, bins=5)
    assert len(curve) == 5
    assert sum(item["count"] for item in curve) == 3
    assert curve[3]["count"] == 2
    assert curve[3]["mean_confidence"] == 0.75
    assert curve[3]["accuracy"] == 1.0


def test_bootstrap_reports_per_class_intervals_and_paired_identity():
    labels = np.tile(np.array([0, 1, 2, 1]), 30)
    probabilities = np.eye(3)[labels]
    groups = np.column_stack((np.ones(len(labels)), np.ones(len(labels))))
    intervals = bootstrap_metrics(
        labels, probabilities, groups, replicates=20, block_length=10, seed=2
    )
    assert len(intervals["f1"]["lower"]) == 3
    paired = paired_bootstrap_metric_differences(
        labels,
        probabilities,
        probabilities,
        groups,
        replicates=20,
        block_length=10,
        seed=2,
    )
    assert paired["macro_f1"] == {"difference": 0.0, "lower": 0.0, "upper": 0.0}


def test_classification_sufficient_statistics_match_direct_metrics():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
            [0.6, 0.2, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    direct = classification_metrics(labels, probabilities)
    sufficient = metrics_from_sufficient_sums(
        classification_row_statistics(labels, probabilities).sum(axis=0)
    )
    for name in ("precision", "recall", "f1"):
        assert np.allclose(sufficient[name][0], direct[name])
    for name in ("macro_f1", "mcc", "ece"):
        assert np.isclose(sufficient[name][0], direct[name])


def test_block_sufficient_sums_equal_materialized_single_group_resamples():
    rows = np.column_stack((np.arange(23), np.arange(23) ** 2))
    groups = np.ones((23, 2), dtype=np.int8)
    fast = bootstrap_sufficient_sums(rows, groups, 10, 5, seed=17)
    rng = np.random.default_rng(17)
    direct = np.vstack(
        [rows[moving_block_indices(groups, 5, rng)].sum(axis=0) for _ in range(10)]
    )
    assert np.array_equal(fast, direct)
