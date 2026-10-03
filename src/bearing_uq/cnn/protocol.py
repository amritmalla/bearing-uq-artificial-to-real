"""Bearings, bearing-level splits and calibration rotations for the CNN runs.

A self-contained copy of the project's split definitions (bearing_uq.bearings and bearing_uq.splits);
tests/test_cnn_protocol.py checks that both give the same splits and rotations.
"""

from itertools import product

import pandas as pd

HEALTHY, INNER, OUTER = "healthy", "inner_race", "outer_race"
LABELS = [HEALTHY, INNER, OUTER]  # class index = position in this list

BEARINGS = pd.DataFrame(
    [(c, HEALTHY, "none") for c in ["K001", "K002", "K003", "K004", "K005", "K006"]]
    + [(c, OUTER, "artificial") for c in ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"]]
    + [(c, INNER, "artificial") for c in ["KI01", "KI03", "KI05", "KI07", "KI08"]]
    + [(c, OUTER, "real") for c in ["KA04", "KA15", "KA16", "KA22", "KA30"]]
    + [(c, INNER, "real") for c in ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]],
    columns=["bearing", "label", "origin"],
)

SOURCE_HEALTHY = ["K001", "K002", "K004"]
TARGET_HEALTHY = ["K003", "K005", "K006"]


def _codes(label, origin):
    rows = BEARINGS[(BEARINGS.label == label) & (BEARINGS.origin == origin)]
    return sorted(rows.bearing)


ARTIFICIAL_INNER = _codes(INNER, "artificial")
ARTIFICIAL_OUTER = _codes(OUTER, "artificial")
SOURCE = SOURCE_HEALTHY + ARTIFICIAL_OUTER + ARTIFICIAL_INNER
TARGET = TARGET_HEALTHY + _codes(OUTER, "real") + _codes(INNER, "real")


def rotations():
    """Yield (key, calibration bearings, training bearings) for all 3 x 5 x 7 = 105 rotations.

    The key ("K001+KI01+KA01") matches the file names used by the project's other experiments.
    """
    for calib in product(SOURCE_HEALTHY, ARTIFICIAL_INNER, ARTIFICIAL_OUTER):
        train = sorted(set(SOURCE) - set(calib))
        yield "+".join(calib), list(calib), train
