import numpy as np

from bearing_uq.plots.curves import reliability_bins, risk_on_grid


def test_risk_on_grid_matches_cumulative_errors():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True])
    np.testing.assert_allclose(risk_on_grid(conf, correct, np.array([0.25, 0.75, 1.0])), [0, 1 / 3, 0.25])


def test_reliability_bins_drop_sparse_bins():
    probs = np.vstack([np.tile([0.95, 0.03, 0.02], (30, 1)), np.tile([0.55, 0.4, 0.05], (5, 1))])
    y = np.zeros(35, dtype=int)
    out = reliability_bins(probs, y, min_count=20)
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], [0.95, 1.0, 30])
