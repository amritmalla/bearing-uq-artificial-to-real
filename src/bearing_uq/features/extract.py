"""Combine all features for one window into a flat dict."""

import numpy as np

from bearing_uq import config, geometry
from bearing_uq.features.envelope import envelope_spectrum
from bearing_uq.features.fault_bands import fault_band_features
from bearing_uq.features.spectral_kurtosis import sk_band
from bearing_uq.features.time_domain import time_features


def default_fault_freqs(shaft_hz: float = config.SHAFT_SPEED_HZ) -> dict[str, float]:
    return {"bpfo": geometry.bpfo(shaft_hz), "bpfi": geometry.bpfi(shaft_hz)}


def window_features(x: np.ndarray, fs: float = config.SAMPLING_RATE,
                    fault_freqs: dict[str, float] | None = None,
                    adaptive: bool = False) -> dict[str, float]:
    """Time-domain and fixed-band fault features; with adaptive=True also the
    fault features from a spectral-kurtosis-selected band (suffix _sk) and that band's centre."""
    fault_freqs = fault_freqs or default_fault_freqs()
    freqs, amps = envelope_spectrum(x, fs)
    out = {**time_features(x), **fault_band_features(freqs, amps, fault_freqs)}
    if adaptive:
        band = sk_band(x, fs)
        freqs_sk, amps_sk = envelope_spectrum(x, fs, band)
        out.update({f"{k}_sk": v for k, v in fault_band_features(freqs_sk, amps_sk, fault_freqs).items()})
        out["sk_band_centre_hz"] = (band[0] + band[1]) / 2
    return out
