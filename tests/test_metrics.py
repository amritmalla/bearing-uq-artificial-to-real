import numpy as np
import pytest

from bearing_uq import metrics as M


def test_ece_zero_for_perfectly_calibrated_bins():
    # 10 predictions at confidence 0.8, 8 of them correct
    probs = np.tile([0.8, 0.1, 0.1], (10, 1))
    y = np.array([0] * 8 + [1] * 2)
    assert M.ece(probs, y) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    probs = np.tile([0.99, 0.005, 0.005], (10, 1))
    y = np.array([0] * 5 + [1] * 5)
    assert M.ece(probs, y) == pytest.approx(0.49)


def test_brier_perfect_is_zero():
    assert M.brier(np.eye(3), np.arange(3)) == 0.0


def test_risk_coverage_orders_by_confidence():
    cov, risk = M.risk_coverage([0.9, 0.8, 0.7, 0.6], np.array([True, True, False, True]))
    np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0, 0, 1 / 3, 1 / 4])


def test_threshold_for_target_and_selective():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True])
    t = M.threshold_for_target(conf, correct, target=0.0)
    assert t == 0.8
    assert M.selective(conf, correct, t) == {"automation_rate": 0.5, "automated_error": 0.0}


def test_threshold_respects_ties():
    # both 0.9 predictions must be taken together; together they have 50 % error
    conf = np.array([0.9, 0.9])
    correct = np.array([True, False])
    assert M.threshold_for_target(conf, correct, target=0.1) == float("inf")
