import pytest

from bearing_uq import geometry


def test_6203_fault_frequencies_at_1500_rpm():
    shaft = 25.0
    assert geometry.bpfo(shaft) == pytest.approx(76.4, abs=0.2)
    assert geometry.bpfi(shaft) == pytest.approx(123.6, abs=0.2)


def test_ftf_below_shaft_speed():
    assert 0 < geometry.ftf(25.0) < 25.0
