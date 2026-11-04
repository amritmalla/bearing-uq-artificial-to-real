import numpy as np

from bearing_uq.features.extract import default_fault_freqs, window_features
from bearing_uq.features.kurtogram import band_grid, kurtogram, kurtogram_band, kurtogram_features

FS = 64_000


def _impacts_at(carrier_hz, fault_hz, seconds=1.0, seed=0, decay=3_000):
    """Decaying bursts at carrier_hz repeating at fault_hz, buried in noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * seconds)) / FS
    x = 0.3 * rng.standard_normal(t.size)
    bt = np.arange(int(0.002 * FS)) / FS
    burst = np.exp(-decay * bt) * np.sin(2 * np.pi * carrier_hz * bt)
    for start in np.arange(0, seconds, 1 / fault_hz):
        i = int(start * FS)
        n = min(burst.size, x.size - i)
        x[i:i + n] += burst[:n]
    return x


def test_band_grid_follows_the_one_third_binary_tree():
    grid = band_grid(FS)
    counts = {}
    for level, low, high in grid:
        counts[round(level, 1)] = counts.get(round(level, 1), 0) + 1
    assert counts == {0.0: 1, 1.0: 2, 1.6: 3, 2.0: 4, 2.6: 6, 3.0: 8, 3.6: 12, 4.0: 16, 4.6: 24, 5.0: 32}
    assert min(high - low for _, low, high in grid) >= 1_000


def test_gaussian_noise_has_kurtosis_near_zero():
    x = np.random.default_rng(1).standard_normal(FS)
    assert max(abs(k) for *_, k in kurtogram(x, FS)) < 0.5


def test_kurtogram_finds_the_impact_resonance():
    low, high, level = kurtogram_band(_impacts_at(18_000, 76.4), FS)
    assert low <= 18_000 <= high
    assert high - low < 10_000  # narrower than the fixed 2-10 kHz band's search would allow


def test_band_stays_inside_the_allowed_range():
    low, high, _ = kurtogram_band(_impacts_at(31_500, 76.4), FS)
    assert 1_000 <= low < high <= 30_000


def test_kurtogram_features_recover_fault_missed_by_fixed_band():
    freqs = default_fault_freqs()
    x = _impacts_at(18_000, freqs["bpfo"])
    fixed = window_features(x)
    fk = kurtogram_features(x, FS, freqs)
    assert fk["bpfo_h1_fk"] > 3 * fixed["bpfo_h1"]
    assert set(fk) == {"bpfo_h1_fk", "bpfo_h2_fk", "bpfi_h1_fk", "bpfi_h2_fk",
                       "fk_band_low_hz", "fk_band_high_hz", "fk_level"}
