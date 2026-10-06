"""Demodulation band from the fast kurtogram (Antoni, 2007): searches over band centres *and* widths.

The fast kurtogram evaluates the kurtosis of the complex envelope in every band of a 1/3-binary tree. At level k
the range 0..fs/2 is split into 2^k bands; at the intermediate level k + log2(3) - 1 (written k.6) into 3 * 2^(k-1)
bands. Levels: 0, 1, 1.6, 2, 2.6, 3, ... The band with the largest kurtosis is where impacts stand out most from
the background. Antoni computes the bands with a tree of short quasi-analytic filters; here each band's complex
envelope is taken directly from the FFT (ideal band-pass, decimated), which gives the same band grid with sharper
band edges.

As for the spectral-kurtosis band (spectral_kurtosis.py), frequencies outside F_MIN..F_MAX are excluded, and
bands narrower than MIN_WIDTH are skipped so that the envelope keeps the second harmonic of the inner-race
frequency (about 250 Hz at 1500 rpm) on both sides of the carrier.
"""

import numpy as np

from bearing_uq.features.envelope import envelope_spectrum
from bearing_uq.features.fault_bands import fault_band_features

F_MIN, F_MAX = 1_000.0, 30_000.0
MIN_WIDTH = 1_000.0  # Hz


def band_grid(fs: float, min_width: float = MIN_WIDTH) -> list[tuple[float, float, float]]:
    """(level, low, high) for every band of the 1/3-binary tree that is at least min_width wide."""
    nyq = fs / 2
    grid = [(0.0, 0.0, nyq)]
    k = 1
    while nyq / 2 ** k >= min_width or nyq / (3 * 2 ** (k - 1)) >= min_width:
        for n_bands, level in ((2 ** k, float(k)), (3 * 2 ** (k - 1), k + np.log2(3) - 1)):
            width = nyq / n_bands
            if width >= min_width:
                grid += [(level, i * width, (i + 1) * width) for i in range(n_bands)]
        k += 1
    return grid


def kurtogram(x: np.ndarray, fs: float, f_min: float = F_MIN, f_max: float = F_MAX,
              min_width: float = MIN_WIDTH) -> list[tuple[float, float, float, float]]:
    """(level, low, high, kurtosis) per band, each band clipped to f_min..f_max.

    Kurtosis of the complex envelope c: E|c|^4 / (E|c|^2)^2 - 2, which is 0 for Gaussian noise.
    """
    x = np.asarray(x, dtype=float)
    spectrum = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    out = []
    for level, low, high in band_grid(fs, min_width):
        low, high = max(low, f_min), min(high, f_max)
        if high - low < min_width:
            continue
        band = spectrum[(freqs >= low) & (freqs < high)]
        power = np.abs(np.fft.ifft(band)) ** 2
        mean_power = power.mean()
        k = (power ** 2).mean() / mean_power ** 2 - 2 if mean_power > 0 else 0.0
        out.append((level, low, high, float(k)))
    return out


def kurtogram_band(x: np.ndarray, fs: float) -> tuple[float, float, float]:
    """(low, high, level) of the band with the largest kurtosis."""
    level, low, high, _ = max(kurtogram(x, fs), key=lambda row: row[3])
    return low, high, level


def kurtogram_features(x: np.ndarray, fs: float, fault_freqs: dict[str, float]) -> dict[str, float]:
    """Fault features (suffix _fk) from the kurtogram-selected band, plus the band's edges and level."""
    low, high, level = kurtogram_band(x, fs)
    freqs, amps = envelope_spectrum(x, fs, (low, high))
    out = {f"{k}_fk": v for k, v in fault_band_features(freqs, amps, fault_freqs).items()}
    out.update(fk_band_low_hz=low, fk_band_high_hz=high, fk_level=level)
    return out
