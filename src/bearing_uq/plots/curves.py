"""Curve helpers for figures."""

import numpy as np

from bearing_uq import metrics as M


def risk_on_grid(conf, correct, grid: np.ndarray) -> np.ndarray:
    """Error among the most confident cases at each coverage level in grid (step interpolation)."""
    coverage, risk = M.risk_coverage(conf, correct)
    idx = np.clip(np.searchsorted(coverage, grid, side="left"), 0, len(risk) - 1)
    return risk[idx]


def reliability_bins(probs, y, n_bins: int = 10, min_count: int = 20):
    """Mean confidence, accuracy and count per equal-width confidence bin (sparse bins dropped)."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows = [(conf[bins == b].mean(), correct[bins == b].mean(), int((bins == b).sum()))
            for b in range(n_bins) if (bins == b).sum() >= min_count]
    return np.array(rows).reshape(-1, 3)
