import numpy as np

from lobdl.metrics import bootstrap_metrics, paired_bootstrap_metric_differences


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
