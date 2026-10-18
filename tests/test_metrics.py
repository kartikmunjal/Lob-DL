import numpy as np

from lobdl.metrics import (
    bootstrap_metrics,
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
