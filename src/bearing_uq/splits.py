"""Bearing-level splits: every bearing belongs to exactly one split.

Splits:
  train  - source bearings used to fit the classifier
  calib  - held-out bearings used to fit calibration / conformal thresholds
  test   - target bearings (real damage) used for evaluation

Default choices are the proposals in docs/status.md and still need confirmation.
"""

import random

from bearing_uq import bearings as B

TRAIN, CALIB, TEST = "train", "calib", "test"

# Healthy bearings differ: K001, K002, K003 and K006 have median kurtosis 14-16, K004 and K005
# only 4-5. Each side gets both kinds, so the healthy class does not shift between source and target.
DEFAULT_SOURCE_HEALTHY = ("K001", "K002", "K004")
DEFAULT_TARGET_HEALTHY = ("K003", "K005", "K006")
LOW_KURTOSIS_HEALTHY = ("K004", "K005")
DEFAULT_SOURCE_CALIB = ("K002", "KA09", "KI08")


def artificial_to_real(source_healthy=DEFAULT_SOURCE_HEALTHY,
                       target_healthy=DEFAULT_TARGET_HEALTHY,
                       source_calib=DEFAULT_SOURCE_CALIB) -> dict[str, str]:
    """Map bearing code -> split for the main artificial-to-real experiment."""
    overlap = set(source_healthy) & set(target_healthy)
    if overlap:
        raise ValueError(f"Healthy bearings in both source and target: {sorted(overlap)}")

    source = list(source_healthy) + [b.code for b in B.select(origin=B.ARTIFICIAL)]
    target = list(target_healthy) + [b.code for b in B.select(origin=B.REAL)]

    unknown = set(source_calib) - set(source)
    if unknown:
        raise ValueError(f"Calibration bearings must come from the source: {sorted(unknown)}")

    split = {code: TRAIN for code in source}
    split.update({code: CALIB for code in source_calib})
    split.update({code: TEST for code in target})
    return split


def with_target_calibration(split: dict[str, str], n_per_class: int,
                            seed: int = 0) -> dict[str, str]:
    """Move n target bearings per class from test into calib (for the 0/1/2 experiment)."""
    rng = random.Random(seed)
    out = dict(split)
    for label in B.CLASSES:
        candidates = sorted(c for c, s in split.items()
                            if s == TEST and B.get(c).label == label)
        if n_per_class >= len(candidates):
            raise ValueError(f"Not enough {label} test bearings to move {n_per_class}")
        for code in rng.sample(candidates, n_per_class):
            out[code] = CALIB
    return out


def codes_in(split: dict[str, str], name: str) -> list[str]:
    return sorted(c for c, s in split.items() if s == name)
