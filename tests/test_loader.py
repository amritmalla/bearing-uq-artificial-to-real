import numpy as np
import pytest
from scipy.io import savemat

from bearing_uq.data.loader import LoaderError, load_vibration


def _write_fake_paderborn(path, signal):
    channels = np.zeros((1, 2), dtype=[("Name", "O"), ("Data", "O")])
    channels[0, 0] = ("force", np.zeros((1, 10)))
    channels[0, 1] = ("vibration_1", signal.reshape(1, -1))
    savemat(path, {path.stem: {"Y": channels}})


def test_loads_vibration_channel(tmp_path):
    path = tmp_path / "N15_M07_F10_KA01_1.mat"
    signal = np.arange(100, dtype=float)
    _write_fake_paderborn(path, signal)
    np.testing.assert_array_equal(load_vibration(path), signal)


def test_missing_channel_raises(tmp_path):
    path = tmp_path / "N15_M07_F10_KA01_2.mat"
    _write_fake_paderborn(path, np.ones(5))
    with pytest.raises(LoaderError, match="not found"):
        load_vibration(path, channel="vibration_2")


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / "N15_M07_F10_KA01_3.mat"
    path.write_bytes(b"not a mat file")
    with pytest.raises(LoaderError):
        load_vibration(path)
