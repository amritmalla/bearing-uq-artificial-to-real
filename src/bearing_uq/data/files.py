"""Locate Paderborn recording files on disk.

Files are named <condition>_<bearing>_<n>.mat, e.g. N15_M07_F10_KA01_1.mat,
and stored in one folder per bearing code.
"""

from pathlib import Path

from bearing_uq import config


def recording_path(bearing_code: str, number: int,
                   condition: str = config.OPERATING_CONDITION,
                   root: Path = config.RAW_DATA_DIR) -> Path:
    return Path(root) / bearing_code / f"{condition}_{bearing_code}_{number}.mat"


def recording_paths(bearing_code: str,
                    condition: str = config.OPERATING_CONDITION,
                    root: Path = config.RAW_DATA_DIR,
                    count: int = config.RECORDINGS_PER_BEARING) -> list[Path]:
    """All expected recording paths for one bearing (existence not checked)."""
    return [recording_path(bearing_code, n, condition, root) for n in range(1, count + 1)]


def missing_files(bearing_codes, condition: str = config.OPERATING_CONDITION,
                  root: Path = config.RAW_DATA_DIR) -> list[Path]:
    """Expected files that are not present on disk."""
    return [p for code in bearing_codes for p in recording_paths(code, condition, root)
            if not p.exists()]
