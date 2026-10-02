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

# Operating condition used in the study: 1500 rpm, 0.7 Nm, 1000 N.
OPERATING_CONDITION = "N15_M07_F10"
SHAFT_SPEED_HZ = 1500 / 60

# Number of 4-second recordings per bearing per operating condition.
RECORDINGS_PER_BEARING = 20

# Window length for feature extraction (seconds) and overlap fraction.
WINDOW_SECONDS = 1.0
WINDOW_OVERLAP = 0.0
