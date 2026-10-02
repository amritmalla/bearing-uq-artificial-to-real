"""Split a long signal into fixed-length windows."""

import numpy as np


def segment(signal: np.ndarray, window: int, overlap: float = 0.0) -> np.ndarray:
    """Return an array of shape (n_windows, window). Trailing samples are dropped."""
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    step = max(1, int(round(window * (1 - overlap))))
    if len(signal) < window:
        return np.empty((0, window))
    starts = range(0, len(signal) - window + 1, step)
    return np.stack([signal[s:s + window] for s in starts])
