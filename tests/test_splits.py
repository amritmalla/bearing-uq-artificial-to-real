import pytest

from bearing_uq import bearings as B
from bearing_uq import splits as S


def test_every_bearing_in_exactly_one_split():
    split = S.artificial_to_real()
    assert set(split) == set(B.BY_CODE)


def test_test_split_is_real_damage_plus_target_healthy():
    split = S.artificial_to_real()
    for code in S.codes_in(split, S.TEST):
        b = B.get(code)
        assert b.origin == B.REAL or code in S.DEFAULT_TARGET_HEALTHY


def test_no_real_damage_in_training():
    split = S.artificial_to_real()
    assert all(B.get(c).origin != B.REAL for c in S.codes_in(split, S.TRAIN))


def test_healthy_overlap_rejected():
    with pytest.raises(ValueError):
        S.artificial_to_real(source_healthy=("K001",), target_healthy=("K001",))


def test_target_calibration_moves_one_per_class():
    base = S.artificial_to_real()
    moved = S.with_target_calibration(base, n_per_class=1)
    new_calib = set(S.codes_in(moved, S.CALIB)) - set(S.codes_in(base, S.CALIB))
    assert len(new_calib) == len(B.CLASSES)
    assert all(base[c] == S.TEST for c in new_calib)
