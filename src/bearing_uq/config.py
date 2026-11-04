"""Project-wide settings."""

import os
from pathlib import Path

# Where the extracted Paderborn bearing folders live (one folder per bearing code, e.g. data/raw/KA01/).
# Override with the BEARING_RAW_DIR environment variable (e.g. a local disk in Colab).
RAW_DATA_DIR = Path(os.environ.get("BEARING_RAW_DIR", "data/raw"))

# Where computed feature tables are written.
FEATURES_DIR = Path("data/features")

# Paderborn vibration sampling rate (Hz).
SAMPLING_RATE = 64_000

# The four Paderborn operating conditions: shaft speed, load torque, radial force.
CONDITIONS = {
    "N15_M07_F10": {"rpm": 1500, "torque_nm": 0.7, "radial_force_n": 1000},
    "N09_M07_F10": {"rpm": 900, "torque_nm": 0.7, "radial_force_n": 1000},
    "N15_M01_F10": {"rpm": 1500, "torque_nm": 0.1, "radial_force_n": 1000},
    "N15_M07_F04": {"rpm": 1500, "torque_nm": 0.7, "radial_force_n": 400},
}

# Main operating condition of the study: 1500 rpm, 0.7 Nm, 1000 N.
OPERATING_CONDITION = "N15_M07_F10"
SHAFT_SPEED_HZ = 1500 / 60


def shaft_speed_hz(condition: str = OPERATING_CONDITION) -> float:
    return CONDITIONS[condition]["rpm"] / 60


def features_path(condition: str = OPERATING_CONDITION) -> Path:
    return FEATURES_DIR / f"features_{condition}.csv"


def result_suffix(feature_set: str, condition: str = OPERATING_CONDITION) -> str:
    """File-name suffix for results; empty for the original setting (all features, main condition)."""
    suffix = "" if feature_set == "all" else f"_{feature_set}"
    return suffix + ("" if condition == OPERATING_CONDITION else f"_{condition}")

# Number of 4-second recordings per bearing per operating condition.
RECORDINGS_PER_BEARING = 20

# Window length for feature extraction (seconds) and overlap fraction.
WINDOW_SECONDS = 1.0
WINDOW_OVERLAP = 0.0
