"""Energy at characteristic fault frequencies in the envelope spectrum.

Each value is the peak amplitude near k * f_fault, divided by the median
amplitude of the envelope spectrum over a reference range. Values well above
1 indicate a periodic impact at that fault frequency.
"""

import numpy as np

REFERENCE_RANGE = (10.0, 500.0)  # Hz, for the noise-floor median
TOLERANCE = 0.02                 # +/- 2 % search window around each target


def peak_near(freqs: np.ndarray, amps: np.ndarray, target: float,
              tolerance: float = TOLERANCE) -> float:
    lo, hi = target * (1 - tolerance), target * (1 + tolerance)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(amps[mask].max()) if mask.any() else 0.0


def noise_floor(freqs: np.ndarray, amps: np.ndarray,
                ref: tuple[float, float] = REFERENCE_RANGE) -> float:
    mask = (freqs >= ref[0]) & (freqs <= ref[1])
    floor = float(np.median(amps[mask])) if mask.any() else 0.0
    return floor if floor > 0 else 1e-12


def fault_band_features(freqs: np.ndarray, amps: np.ndarray,
                        fault_freqs: dict[str, float],
                        harmonics: int = 2) -> dict[str, float]:
    """e.g. fault_freqs={"bpfo": 76.4, "bpfi": 123.6} -> bpfo_h1, bpfo_h2, ..."""
    floor = noise_floor(freqs, amps)
    out = {}
    for name, f in fault_freqs.items():
        for k in range(1, harmonics + 1):
            out[f"{name}_h{k}"] = peak_near(freqs, amps, k * f) / floor
    return out
