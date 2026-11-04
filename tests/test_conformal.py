import numpy as np

from bearing_uq.conformal import SplitConformal, set_metrics


def _data(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([1, 1, 1], size=n)
    y = np.array([rng.choice(3, p=row) for row in p])
    return p, y


def test_marginal_coverage_close_to_target():
    p_cal, y_cal = _data(3000, 0)
    p_test, y_test = _data(3000, 1)
    cp = SplitConformal(alpha=0.1).fit(p_cal, y_cal)
    cov = set_metrics(cp.predict_sets(p_test), y_test)["coverage"]
    assert 0.87 < cov < 0.93


def test_class_conditional_has_threshold_per_class():
    p, y = _data(600, 2)
    cp = SplitConformal(alpha=0.1, class_conditional=True).fit(p, y)
    assert cp.qhat.shape == (3,)
