import numpy as np

from scripts.audit_lobster_all import eligible_window_count


def test_eligible_window_requires_every_row_exact():
    mask = np.array([True, True, True, False, True, True, True])
    assert eligible_window_count(mask, 3) == 2
