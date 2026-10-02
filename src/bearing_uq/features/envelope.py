"""Envelope spectrum: band-pass, Hilbert envelope, then FFT."""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

# Default demodulation band (Hz). A fixed band keeps the pipeline simple;
# a kurtogram-selected band is a possible later refinement.
DEFAULT_BAND = (2_000.0, 10_000.0)


def envelope_spectrum(x: np.ndarray, fs: float,
                      band: tuple[float, float] = DEFAULT_BAND
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, amplitudes) of the envelope spectrum."""
    low, high = band
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(x, dtype=float))
    env = np.abs(hilbert(filtered))
    env -= env.mean()
    amps = np.abs(np.fft.rfft(env)) / len(env)
    freqs = np.fft.rfftfreq(len(env), d=1 / fs)
    return freqs, amps
