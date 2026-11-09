"""CORAL feature alignment (Sun et al. 2016) of target windows onto the source feature statistics.

Setup (fixed on 2026-11-08, before any result of it was computed):

- Feature space: the four log fault-frequency features used by the feature-based classifiers.
- Direction: target to source, so each rotation's classifiers, calibrators, selective thresholds and conformal
  quantiles stay exactly as fitted on source bearings; only the target test features change.
- Source statistics: mean and covariance of the rotation's training windows.
- Target statistics: for each target test bearing, the mean and covariance of the unlabelled windows of the
  other 13 target bearings. A bearing is never aligned with statistics that include its own windows.
- Map: x' = (x - mu_t) C_t^(-1/2) C_s^(1/2) + mu_s, i.e. CORAL with mean alignment. Each covariance gets a
  ridge of 1e-3 times its mean variance on the diagonal, for numerical stability only.
- Evaluation: the same metrics, rotations and hierarchical bootstrap (same seed) as the main analysis.

Hypothesis stated in advance: alignment may change accuracy, but it will not bring the error among accepted
diagnoses close to the 5 % target, because the damaged bearings with weak fault peaks already lie in the region
occupied by healthy source bearings, and matching the marginal statistics cannot move them out of it.
"""

import numpy as np

RIDGE = 1e-3


def _sqrt_pair(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric square root and inverse square root of a covariance matrix (with the stability ridge)."""
    cov = cov + RIDGE * np.mean(np.diag(cov)) * np.eye(len(cov))
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    return (v * np.sqrt(w)) @ v.T, (v / np.sqrt(w)) @ v.T


def coral_map(x: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map rows of x, drawn like `target`, so that `target` would match the mean and covariance of `source`."""
    s_sqrt, _ = _sqrt_pair(np.cov(source, rowvar=False))
    _, t_isqrt = _sqrt_pair(np.cov(target, rowvar=False))
    return (x - target.mean(0)) @ t_isqrt @ s_sqrt + source.mean(0)


def align_held_out_bearings(x_train: np.ndarray, x_test: np.ndarray, bearing: np.ndarray) -> np.ndarray:
    """Align each test bearing using the statistics of all other test bearings (never its own windows)."""
    bearing = np.asarray(bearing)
    out = np.empty_like(x_test, dtype=float)
    for code in np.unique(bearing):
        own = bearing == code
        out[own] = coral_map(x_test[own], x_train, x_test[~own])
    return out
