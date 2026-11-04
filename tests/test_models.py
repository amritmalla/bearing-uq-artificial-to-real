import numpy as np

from bearing_uq.models import fit, make_models


def test_models_fit_and_return_probabilities():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((90, 4))
    y = np.repeat([0, 1, 2], 30)
    X[y == 1, 0] += 3
    X[y == 2, 1] += 3
    for name, model in make_models().items():
        p = fit(model, X, y).predict_proba(X)
        assert p.shape == (90, 3), name
        np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)
