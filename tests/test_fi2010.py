import json

import numpy as np

from lobdl.data.fi2010 import Segment, iter_window_indices, reconstruct_forward_mean_labels
from lobdl.datasets import fi_economic_arrays


def test_windows_never_cross_segments():
    segments = (Segment(1, 1, 0, 20), Segment(1, 2, 20, 45))
    indices = list(iter_window_indices(segments, window=5, horizon=3))
    assert indices
    for segment, end in indices:
        assert end - 4 >= segment.start
        assert end + 3 < segment.stop


def test_forward_mean_label_threshold_is_inclusive():
    features = np.zeros((4, 40), dtype=np.float32)
    mids = np.array([100.0, 100.2, 100.0, 99.8])
    features[:, 0] = mids
    features[:, 2] = mids
    labels = reconstruct_forward_mean_labels(features, horizon_events=10, alpha=0.002)
    assert labels[:3].tolist() == [2, 1, 0]


def test_fi_economic_arrays_follow_registered_window_order(tmp_path):
    matrix = np.zeros((110, 149), dtype=np.float32)
    matrix[:, 0] = np.arange(101, 211)
    matrix[:, 2] = matrix[:, 0] - 2
    np.save(tmp_path / "day_01.npy", matrix)
    (tmp_path / "segments.json").write_text(
        json.dumps([{"day": 1, "stock": 1, "start": 0, "stop": 110}]),
        encoding="utf-8",
    )
    returns, spreads = fi_economic_arrays(tmp_path, [1], 10)
    assert len(returns) == 10
    assert np.isclose(returns[0], 1 / 199)
    assert np.isclose(spreads[0], 2 / 199)
