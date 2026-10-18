import numpy as np

from lobdl.transfer import (
    apply_decimal_precision,
    decimal_precision_powers,
    eligible_ends,
    forward_mean_returns,
    label_returns,
    sample_message_blocks,
    select_distribution_matched_alpha,
)


def test_message_sampling_and_decimal_precision_are_frozen_scale_only():
    raw = np.arange(1, 21 * 40 + 1, dtype=np.int64).reshape(21, 40)
    sampled = sample_message_blocks(raw)
    assert np.array_equal(sampled, raw[[9, 19]])

    calibration = np.array([[0, 10, 999], [0, 99, 1000]], dtype=np.int64)
    powers = decimal_precision_powers(calibration)
    assert powers.tolist() == [0, 2, 4]
    scaled = apply_decimal_precision(calibration, powers)
    assert np.allclose(np.max(np.abs(scaled[:, 1:]), axis=0), [0.99, 0.1])


def test_forward_labels_and_split_ends_do_not_cross_boundary():
    book = np.zeros((210, 40), dtype=np.int64)
    book[:, 0] = np.arange(100, 310)
    book[:, 2] = book[:, 0] - 2
    returns = forward_mean_returns(book, 2)
    expected = ((100 + 101) / 2 - 99) / 99
    assert np.isclose(returns[0], expected)
    assert np.isnan(returns[-2:]).all()
    assert label_returns(np.array([-0.2, -0.1, 0.0, 0.1, 0.2]), 0.1).tolist() == [0, 0, 1, 2, 2]

    calibration = eligible_ends(210, 150, 10, split="calibration")
    evaluation = eligible_ends(210, 150, 10, split="evaluation")
    assert calibration[0] == 99 and calibration[-1] == 139
    assert evaluation[0] == 150 and evaluation[-1] == 199


def test_distribution_match_uses_ascending_grid_tie_break():
    returns = np.array([-0.2, 0.0, 0.2])
    alpha, distance, proportions = select_distribution_matched_alpha(
        returns,
        np.array([1 / 3, 1 / 3, 1 / 3]),
        np.array([0.1, 0.15]),
    )
    assert alpha == 0.1
    assert distance == 0.0
    assert np.allclose(proportions, 1 / 3)
