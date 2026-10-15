import numpy as np

from lobdl.features import causal_features


def test_features_are_finite_and_causal_prefix_stable():
    rng = np.random.default_rng(3)
    raw = np.zeros((100, 40), dtype=np.float32)
    for level in range(10):
        raw[:, 4 * level] = 101 + level
        raw[:, 4 * level + 1] = rng.integers(1, 100, 100)
        raw[:, 4 * level + 2] = 99 - level
        raw[:, 4 * level + 3] = rng.integers(1, 100, 100)
    full, names = causal_features(raw)
    prefix, prefix_names = causal_features(raw[:70])
    assert names == prefix_names
    np.testing.assert_allclose(full[:70], prefix)
    assert np.isfinite(full).all()

