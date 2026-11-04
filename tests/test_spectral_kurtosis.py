import numpy as np

from bearing_uq.features.extract import default_fault_freqs, window_features
from bearing_uq.features.spectral_kurtosis import sk_band

FS = 64_000


def _impacts_at(carrier_hz: float, fault_hz: float, seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """Decaying bursts at carrier_hz repeating at fault_hz, buried in noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * seconds)) / FS
    x = 0.3 * rng.standard_normal(t.size)
    bt = np.arange(int(0.002 * FS)) / FS
    burst = np.exp(-3_000 * bt) * np.sin(2 * np.pi * carrier_hz * bt)
    for start in np.arange(0, seconds, 1 / fault_hz):
        i = int(start * FS)
        n = min(burst.size, x.size - i)
        x[i:i + n] += burst[:n]
    return x


def test_sk_band_finds_the_impact_resonance():
    low, high = sk_band(_impacts_at(18_000, 76.4), FS)
    assert low <= 18_000 <= high


def test_adaptive_band_recovers_fault_missed_by_fixed_band():
    bpfo = default_fault_freqs()["bpfo"]
    f = window_features(_impacts_at(18_000, bpfo), adaptive=True)
    # resonance at 18 kHz lies outside the fixed 2-10 kHz band
    assert f["bpfo_h1_sk"] > 3 * f["bpfo_h1"]
    assert 14_000 <= f["sk_band_centre_hz"] <= 22_000


def test_non_adaptive_output_unchanged():
    x = _impacts_at(5_000, 76.4)
    assert not any(k.endswith("_sk") for k in window_features(x))
