"""Choose the demodulation band from the signal's spectral kurtosis.

Spectral kurtosis (SK) is large in frequency bands dominated by repetitive impacts, so it
indicates where bearing-fault impulses are strongest. We compute SK from a short-time Fourier
transform, smooth it over the band width, and centre a band of fixed width on its maximum.
This is a simplified alternative to the fast kurtogram, which also searches over band widths.
"""

import numpy as np
from scipy.signal import stft

NPERSEG = 256          # 250 Hz resolution at 64 kHz
BANDWIDTH = 4_000.0    # Hz
F_MIN, F_MAX = 1_000.0, 30_000.0


def spectral_kurtosis(x: np.ndarray, fs: float, nperseg: int = NPERSEG):
    """Return (frequencies, SK) with SK(f) = E|X|^4 / (E|X|^2)^2 - 2."""
    f, _, z = stft(np.asarray(x, dtype=float), fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    power = np.abs(z) ** 2
    mean_power = power.mean(axis=1)
    sk = (power ** 2).mean(axis=1) / np.where(mean_power > 0, mean_power ** 2, np.inf) - 2
    return f, sk


def sk_band(x: np.ndarray, fs: float, bandwidth: float = BANDWIDTH,
            f_min: float = F_MIN, f_max: float = F_MAX) -> tuple[float, float]:
    """Band of the given width centred where band-averaged SK is largest."""
    f, sk = spectral_kurtosis(x, fs)
    step = f[1] - f[0]
    width = max(1, int(round(bandwidth / step)))
    smoothed = np.convolve(sk, np.ones(width) / width, mode="same")
    allowed = (f >= f_min + bandwidth / 2) & (f <= f_max - bandwidth / 2)
    centre = float(f[allowed][np.argmax(smoothed[allowed])])
    return centre - bandwidth / 2, centre + bandwidth / 2
