"""Combine all features for one window into a flat dict."""

import numpy as np

from bearing_uq import config, geometry
from bearing_uq.features.envelope import envelope_spectrum
from bearing_uq.features.fault_bands import fault_band_features
from bearing_uq.features.time_domain import time_features


def default_fault_freqs(shaft_hz: float = config.SHAFT_SPEED_HZ) -> dict[str, float]:
    return {"bpfo": geometry.bpfo(shaft_hz), "bpfi": geometry.bpfi(shaft_hz)}


def window_features(x: np.ndarray, fs: float = config.SAMPLING_RATE,
                    fault_freqs: dict[str, float] | None = None) -> dict[str, float]:
    fault_freqs = fault_freqs or default_fault_freqs()
    freqs, amps = envelope_spectrum(x, fs)
    return {**time_features(x), **fault_band_features(freqs, amps, fault_freqs)}
