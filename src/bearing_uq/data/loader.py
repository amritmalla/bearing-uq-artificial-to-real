"""Read the vibration signal from a Paderborn .mat file.

Expected layout (to be confirmed on real files): the file holds one struct
named after the file stem; its field ``Y`` is an array of channel structs,
each with ``Name`` and ``Data``. The vibration channel is named ``vibration_1``.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

VIBRATION_CHANNEL = "vibration_1"


class LoaderError(RuntimeError):
    pass


def _channel_name(channel) -> str:
    name = channel["Name"]
    while isinstance(name, np.ndarray):
        name = name.flat[0]
    return str(name)


def load_vibration(path: Path, channel: str = VIBRATION_CHANNEL) -> np.ndarray:
    """Return the vibration signal as a 1-D float array."""
    path = Path(path)
    try:
        mat = loadmat(path)
    except Exception as exc:  # some Paderborn files are known to be unreadable
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc

    if path.stem not in mat:
        raise LoaderError(f"{path.name}: expected a struct named {path.stem!r}")

    record = mat[path.stem][0, 0]
    channels = record["Y"].ravel()
    for ch in channels:
        if _channel_name(ch) == channel:
            return np.asarray(ch["Data"], dtype=float).ravel()

    found = [_channel_name(ch) for ch in channels]
    raise LoaderError(f"{path.name}: channel {channel!r} not found (have {found})")
