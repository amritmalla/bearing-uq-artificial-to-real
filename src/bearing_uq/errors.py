"""What kind of errors the models make on real damage, from saved per-rotation predictions."""

import numpy as np
import pandas as pd

from bearing_uq import dataset as D
from bearing_uq.predictions import TARGETS

HEALTHY = D.LABELS.index("healthy")


def missed_fault_shares(pred: dict, calibrator: str = "temperature", target: float = 0.05) -> pd.DataFrame:
    """Share of errors that call a damaged bearing healthy, over all test windows and over the windows
    decided automatically at the error target; mean over rotations."""
    ci, ti = pred["calibrators"].index(calibrator), TARGETS.index(target)
    y = pred["y"]
    rows = []
    for mi, model in enumerate(pred["models"]):
        shares, shares_auto = [], []
        for r in range(pred["probs"].shape[0]):
            p = pred["probs"][r, mi, ci]
            guess = p.argmax(1)
            error = guess != y
            missed = error & (y != HEALTHY) & (guess == HEALTHY)
            auto = p.max(1) >= pred["thresholds"][r, mi, ci, ti]
            shares.append(missed.sum() / error.sum() if error.any() else np.nan)
            shares_auto.append((missed & auto).sum() / (error & auto).sum() if (error & auto).any() else np.nan)
        rows.append({"model": model, "missed_share": np.nanmean(shares),
                     "missed_share_automated": np.nanmean(shares_auto)})
    return pd.DataFrame(rows)
