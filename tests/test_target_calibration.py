import numpy as np
import pytest

from bearing_uq import bearings as B
from bearing_uq import splits as S
from bearing_uq.target_calibration import choose_real_calibration


def test_choose_real_calibration_takes_n_test_bearings_per_class():
    split = S.artificial_to_real()
    chosen = choose_real_calibration(split, 2, np.random.default_rng(0))
    assert len(chosen) == 2 * len(B.CLASSES)
    assert all(split[c] == S.TEST for c in chosen)
    for label in B.CLASSES:
        assert sum(B.get(c).label == label for c in chosen) == 2


def test_choose_real_calibration_keeps_a_test_bearing_per_class():
    with pytest.raises(ValueError):
        choose_real_calibration(S.artificial_to_real(), 3, np.random.default_rng(0))  # only 3 target healthy
