"""Time-domain statistics of a vibration window."""

import numpy as np
from scipy.stats import kurtosis, skew


def time_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    return {
        "rms": rms,
        "kurtosis": float(kurtosis(x, fisher=False)),
        "skewness": float(skew(x)),
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "peak_to_peak": float(np.ptp(x)),
    }
