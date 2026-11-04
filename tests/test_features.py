import numpy as np

from bearing_uq.features.extract import default_fault_freqs, window_features
from bearing_uq.features.time_domain import time_features

FS = 64_000


def _impacts(fault_hz: float, seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """Synthetic bearing fault: decaying 5 kHz bursts repeating at fault_hz, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * seconds)) / FS
    x = 0.1 * rng.standard_normal(t.size)
    burst_t = np.arange(int(0.002 * FS)) / FS
    burst = np.exp(-2_000 * burst_t) * np.sin(2 * np.pi * 5_000 * burst_t)
    for start in np.arange(0, seconds, 1 / fault_hz):
        i = int(start * FS)
        n = min(burst.size, x.size - i)
        x[i:i + n] += burst[:n]
    return x


def test_outer_race_impacts_show_at_bpfo():
    freqs = default_fault_freqs()
    f = window_features(_impacts(freqs["bpfo"]))
    assert f["bpfo_h1"] > 5 * f["bpfi_h1"]


def test_inner_race_impacts_show_at_bpfi():
    freqs = default_fault_freqs()
    f = window_features(_impacts(freqs["bpfi"]))
    assert f["bpfi_h1"] > 5 * f["bpfo_h1"]


def test_impacts_raise_kurtosis_over_noise():
    noise = np.random.default_rng(1).standard_normal(FS)
    assert time_features(_impacts(76.4))["kurtosis"] > time_features(noise)["kurtosis"]
