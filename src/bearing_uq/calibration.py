"""Post-hoc probability calibration, fitted on a held-out calibration set."""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

EPS = 1e-12


def _normalise(p: np.ndarray) -> np.ndarray:
    s = p.sum(axis=1, keepdims=True)
    uniform = np.full_like(p, 1.0 / p.shape[1])
    return np.where(s > 0, p / np.where(s > 0, s, 1), uniform)


class Identity:
    def fit(self, probs, y):
        return self

    def transform(self, probs):
        return np.asarray(probs, dtype=float)


class TemperatureScaler:
    """Softmax(log p / T) with T chosen to minimise negative log-likelihood."""

    def __init__(self):
        self.temperature = 1.0

    @staticmethod
    def _scale(probs, t):
        z = np.log(np.clip(probs, EPS, 1.0)) / t
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, probs, y):
        probs, y = np.asarray(probs, float), np.asarray(y)

        def nll(log_t):
            p = self._scale(probs, np.exp(log_t))
            return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)))

        self.temperature = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        return self

    def transform(self, probs):
        return self._scale(np.asarray(probs, float), self.temperature)


class IsotonicCalibrator:
    """One-vs-rest isotonic regression per class, then renormalised."""

    def fit(self, probs, y):
        probs, y = np.asarray(probs, float), np.asarray(y)
        self.models = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                       .fit(probs[:, k], (y == k).astype(float)) for k in range(probs.shape[1])]
        return self

    def transform(self, probs):
        probs = np.asarray(probs, float)
        out = np.column_stack([m.predict(probs[:, k]) for k, m in enumerate(self.models)])
        return _normalise(out)


CALIBRATORS = {"raw": Identity, "temperature": TemperatureScaler, "isotonic": IsotonicCalibrator}
