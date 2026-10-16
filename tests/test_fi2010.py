import numpy as np

from lobdl.data.fi2010 import Segment, iter_window_indices, reconstruct_single_future_labels


def test_windows_never_cross_segments():
    segments = (Segment(1, 1, 0, 20), Segment(1, 2, 20, 45))
    indices = list(iter_window_indices(segments, window=5, horizon=3))
    assert indices
    for segment, end in indices:
        assert end - 4 >= segment.start
        assert end + 3 < segment.stop


def test_single_future_label_threshold_is_inclusive():
    features = np.zeros((13, 40), dtype=np.float32)
    mids = np.full(13, 100.0)
    mids[10:13] = [100.2, 100.0, 99.8]
    features[:, 0] = mids
    features[:, 2] = mids
    labels = reconstruct_single_future_labels(features, horizon_events=10, alpha=0.002)
    assert labels[:3].tolist() == [2, 1, 0]
    assert (labels[-10:] == -1).all()
