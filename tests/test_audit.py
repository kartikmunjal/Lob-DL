import numpy as np
import pytest

from lobdl.audit import TrainStandardizer, assert_split_days


def test_standardizer_rejects_nontraining_fit():
    with pytest.raises(AssertionError):
        TrainStandardizer.fit(np.zeros((2, 3, 4)), "test")


def test_standardizer_uses_stored_statistics():
    train = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    test = np.full((1, 2, 4), 100.0, dtype=np.float32)
    scaler = TrainStandardizer.fit(train, "train")
    transformed = scaler.transform(test)
    assert np.all(transformed > 1)
    assert scaler.fit_split == "train"


def test_strict_temporal_splits():
    assert_split_days(set(range(1, 7)), {7}, {8, 9, 10})
    with pytest.raises(AssertionError):
        assert_split_days({1, 2, 7}, {7}, {8})

