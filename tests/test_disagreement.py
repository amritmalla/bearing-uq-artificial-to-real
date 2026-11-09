import numpy as np
import pytest

from bearing_uq.disagreement import HEALTHY, accept_mask, accepted_metrics


def test_agreement_only_removes_windows():
    conf = np.array([0.9, 0.9, 0.5, 0.95])
    label = np.array([0, 1, 2, 2])
    other = np.array([0, 2, 2, 2])
    base = accept_mask(conf, label, other, 0.8, "confidence")
    agree = accept_mask(conf, label, other, 0.8, "agreement")
    assert base.tolist() == [True, True, False, True]
    assert agree.tolist() == [True, False, False, True]
    assert not (agree & ~base).any()


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        accept_mask([0.9], [0], [0], 0.5, "vote")


def test_accepted_metrics_counts_missed_faults():
    fault = 1 if HEALTHY != 1 else 2
    y = np.array([fault, fault, HEALTHY, fault])
    label = np.array([HEALTHY, fault, fault, HEALTHY])
    keep = np.array([True, True, True, False])
    m = accepted_metrics(keep, label, y)
    assert m["automation_rate"] == pytest.approx(0.75)
    assert m["automated_error"] == pytest.approx(2 / 3)
    assert m["missed_share_of_errors"] == pytest.approx(0.5)  # one missed fault, one false alarm


def test_nothing_accepted_gives_nan():
    m = accepted_metrics(np.zeros(3, bool), np.zeros(3, int), np.ones(3, int))
    assert m["automation_rate"] == 0 and np.isnan(m["automated_error"])
