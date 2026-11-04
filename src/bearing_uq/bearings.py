"""Catalogue of Paderborn bearing codes used in the study.

Verify these codes against the dataset's own fact sheets after download.
Combined-damage bearings (KB23, KB24, KB27) are excluded from the study.
"""

from dataclasses import dataclass

HEALTHY = "healthy"
INNER = "inner_race"
OUTER = "outer_race"
CLASSES = (HEALTHY, INNER, OUTER)

NONE = "none"
ARTIFICIAL = "artificial"
REAL = "real"


@dataclass(frozen=True)
class Bearing:
    code: str
    label: str
    origin: str  # NONE for healthy, ARTIFICIAL or REAL for damaged


def _make(codes, label, origin):
    return [Bearing(code, label, origin) for code in codes]


BEARINGS = (
    _make(["K001", "K002", "K003", "K004", "K005", "K006"], HEALTHY, NONE)
    + _make(["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"], OUTER, ARTIFICIAL)
    + _make(["KI01", "KI03", "KI05", "KI07", "KI08"], INNER, ARTIFICIAL)
    + _make(["KA04", "KA15", "KA16", "KA22", "KA30"], OUTER, REAL)
    + _make(["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"], INNER, REAL)
)

BY_CODE = {b.code: b for b in BEARINGS}


def get(code: str) -> Bearing:
    try:
        return BY_CODE[code]
    except KeyError:
        raise KeyError(f"Unknown or excluded bearing code: {code}") from None


def select(label: str | None = None, origin: str | None = None) -> list[Bearing]:
    """Return bearings matching the given label and/or damage origin."""
    return [
        b for b in BEARINGS
        if (label is None or b.label == label) and (origin is None or b.origin == origin)
    ]
