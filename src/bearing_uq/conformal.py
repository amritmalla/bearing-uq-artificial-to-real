"""Split conformal prediction for classification (score = 1 - p(true class))."""

import math

import numpy as np


def _qhat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


class SplitConformal:
    """Marginal (one threshold) or class-conditional (one threshold per class)."""

    def __init__(self, alpha: float = 0.1, class_conditional: bool = False):
        self.alpha = alpha
        self.class_conditional = class_conditional

    def fit(self, probs, y):
        probs, y = np.asarray(probs, float), np.asarray(y)
        scores = 1.0 - probs[np.arange(len(y)), y]
        k = probs.shape[1]
        if self.class_conditional:
            self.qhat = np.array([_qhat(scores[y == c], self.alpha) for c in range(k)])
        else:
            self.qhat = np.full(k, _qhat(scores, self.alpha))
        return self

    def predict_sets(self, probs) -> np.ndarray:
        """Boolean array (n, k): True where the class is in the prediction set."""
        return (1.0 - np.asarray(probs, float)) <= self.qhat[None, :]


def set_metrics(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    single = sizes == 1
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(single.mean()),
        "singleton_error": float(1 - covered[single].mean()) if single.any() else float("nan"),
    }
