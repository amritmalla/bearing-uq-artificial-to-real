import numpy as np

from bearing_uq.calibration import IsotonicCalibrator, TemperatureScaler


def _overconfident(n=2000, seed=0):
    """Labels drawn from soft probabilities; the model reports sharpened versions of them."""
    rng = np.random.default_rng(seed)
    true_p = rng.dirichlet([2, 2, 2], size=n)
    y = np.array([rng.choice(3, p=p) for p in true_p])
    sharp = true_p ** 4
    return sharp / sharp.sum(axis=1, keepdims=True), y


def _nll(p, y):
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))


def test_temperature_scaling_softens_overconfident_model():
    p, y = _overconfident()
    ts = TemperatureScaler().fit(p, y)
    assert ts.temperature > 1.5
    assert _nll(ts.transform(p), y) < _nll(p, y)


def test_isotonic_outputs_are_probabilities():
    p, y = _overconfident()
    out = IsotonicCalibrator().fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all()
