import numpy as np

from bearing_uq.bootstrap import bearing_indices, resample


def test_resample_keeps_bearings_per_class():
    bearing = np.array(["A", "A", "B", "B", "C", "C", "C"])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    groups = bearing_indices(bearing, y)
    assert {k: len(v) for k, v in groups.items()} == {0: 2, 1: 1}

    idx = resample(groups, np.random.default_rng(0))
    picked = [b for b in bearing[idx]]
    assert picked.count("C") == 3  # the only class-1 bearing is always drawn, whole
    assert len(idx) in (4 + 3, 2 + 2 + 3)
