"""Bearing geometry and characteristic fault frequencies."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    n_balls: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0

    @property
    def _ratio(self) -> float:
        return (self.ball_diameter_mm / self.pitch_diameter_mm) * math.cos(
            math.radians(self.contact_angle_deg)
        )


# Deep groove ball bearing 6203 used in the Paderborn test rig.
BEARING_6203 = BearingGeometry(n_balls=8, ball_diameter_mm=6.75, pitch_diameter_mm=28.55)


def bpfo(shaft_hz: float, geom: BearingGeometry = BEARING_6203) -> float:
    """Ball pass frequency, outer race (Hz)."""
    return geom.n_balls / 2 * (1 - geom._ratio) * shaft_hz


def bpfi(shaft_hz: float, geom: BearingGeometry = BEARING_6203) -> float:
    """Ball pass frequency, inner race (Hz)."""
    return geom.n_balls / 2 * (1 + geom._ratio) * shaft_hz


def bsf(shaft_hz: float, geom: BearingGeometry = BEARING_6203) -> float:
    """Ball spin frequency (Hz)."""
    d, D = geom.ball_diameter_mm, geom.pitch_diameter_mm
    return D / (2 * d) * (1 - geom._ratio**2) * shaft_hz


def ftf(shaft_hz: float, geom: BearingGeometry = BEARING_6203) -> float:
    """Fundamental train (cage) frequency (Hz)."""
    return 0.5 * (1 - geom._ratio) * shaft_hz
