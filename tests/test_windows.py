import numpy as np
import pytest

from bearing_uq.data.windows import segment


def test_non_overlapping_windows_drop_tail():
    out = segment(np.arange(10), window=4)
    assert out.shape == (2, 4)
    np.testing.assert_array_equal(out[1], [4, 5, 6, 7])


def test_half_overlap():
    assert segment(np.arange(10), window=4, overlap=0.5).shape == (4, 4)


def test_short_signal_gives_no_windows():
    assert segment(np.arange(3), window=4).shape == (0, 4)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        segment(np.arange(10), window=4, overlap=1.0)
