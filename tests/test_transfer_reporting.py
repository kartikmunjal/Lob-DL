import numpy as np

from scripts.aggregate_results import fi_economics
from scripts.aggregate_transfer import economics


def test_economics_uses_only_correct_nonstationary_predictions():
    labels = np.array([0, 1, 2, 2])
    probabilities = np.eye(3)[[0, 1, 2, 0]]
    result = economics(
        labels,
        probabilities,
        np.array([-0.01, 0.0, 0.02, 0.03]),
        np.full(4, 100.0),
        np.full(4, 0.5),
        1.0,
    )
    assert result["correct_direction_windows"] == 2
    assert result["mean_absolute_forward_move"] == 1.5
    assert np.isclose(result["mean_spread_plus_fee"], 0.51)
    assert result["fraction_move_exceeds_cost"] == 1.0


def test_fi_economics_reports_relative_values_in_basis_points():
    labels = np.array([0, 1, 2, 2])
    probabilities = np.eye(3)[[0, 1, 2, 0]]
    result = fi_economics(
        labels,
        probabilities,
        np.array([-0.001, 0.0, 0.002, 0.003]),
        np.full(4, 0.00005),
        1.0,
    )
    assert result["correct_direction_windows"] == 2
    assert result["mean_absolute_forward_move_bps"] == 15.0
    assert result["mean_spread_plus_fee_bps"] == 1.5
    assert result["mean_move_minus_cost_bps"] == 13.5
