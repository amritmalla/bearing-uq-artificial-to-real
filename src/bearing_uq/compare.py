"""Per-rotation and per-bearing summaries of saved predictions, for comparing variants run the same way.

Used by scripts/run_coral.py (baseline vs CORAL) and scripts/run_seeds.py (training seeds).
"""

import numpy as np
import pandas as pd

from bearing_uq import metrics as M
from bearing_uq.predictions import CONFORMAL_METHODS, TARGETS

HEALTHY = 0


def per_rotation(pred: dict, variant: str) -> pd.DataFrame:
    """Target metrics for every rotation and model (temperature scaling; conformal on raw probabilities)."""
    y, rows = pred["y"], []
    ti5 = TARGETS.index(0.05)
    for r, rot in enumerate(pred["rotations"]):
        for mi, model in enumerate(pred["models"]):
            raw = pred["probs"][r, mi, list(pred["calibrators"]).index("raw")]
            row = {"variant": variant, "rotation": rot, "model": model}
            for qi, method in enumerate(CONFORMAL_METHODS):
                sets = (1 - raw) <= pred["qhat"][r, mi, qi][None, :]
                row[f"coverage_{method}"] = sets[np.arange(len(y)), y].mean()
            ci = list(pred["calibrators"]).index("temperature")
            p = pred["probs"][r, mi, ci]
            conf, correct = p.max(1), p.argmax(1) == y
            row.update(accuracy=correct.mean(), ece=M.ece(p, y))
            for ti, target in enumerate(TARGETS):
                sel = M.selective(conf, correct, pred["thresholds"][r, mi, ci, ti])
                row[f"auto_rate@{target:.0%}"] = sel["automation_rate"]
                row[f"auto_error@{target:.0%}"] = sel["automated_error"]
            keep = conf >= pred["thresholds"][r, mi, ci, ti5]
            wrong = keep & ~correct
            missed = wrong & (p.argmax(1) == HEALTHY) & (y != HEALTHY)
            row["missed_share_of_accepted_errors@5%"] = missed.sum() / wrong.sum() if wrong.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def per_bearing(pred: dict, variant: str) -> pd.DataFrame:
    """Accuracy, confidence and acceptance at the 5 % target for each target bearing, averaged over rotations."""
    y, b, rows = pred["y"], pred["bearing"], []
    ci, ti5 = list(pred["calibrators"]).index("temperature"), TARGETS.index(0.05)
    for r in range(len(pred["rotations"])):
        for mi, model in enumerate(pred["models"]):
            p = pred["probs"][r, mi, ci]
            conf, label = p.max(1), p.argmax(1)
            keep = conf >= pred["thresholds"][r, mi, ci, ti5]
            for code in np.unique(b):
                m = b == code
                rows.append({"variant": variant, "model": model, "bearing": code,
                             "accuracy": (label[m] == y[m]).mean(), "mean_confidence": conf[m].mean(),
                             "accepted": keep[m].mean(), "wrong_accepted": (keep & (label != y))[m].mean(),
                             "share_called_healthy": (label[m] == HEALTHY).mean()})
    return pd.DataFrame(rows).groupby(["variant", "model", "bearing"]).mean().reset_index()


def summarise(rot: pd.DataFrame, by=("variant", "model")) -> pd.DataFrame:
    """Means over rotations, plus the share of rotations (with any accepted window) above the 5 % target."""
    by = list(by)
    accepted = rot[rot["auto_rate@5%"] > 0].assign(exceeded=lambda d: d["auto_error@5%"] > 0.05)
    out = rot.groupby(by).mean(numeric_only=True)
    out["exceeded_5%_share"] = accepted.groupby(by)["exceeded"].mean()
    return out


def stack_runs(preds: dict[str, dict]) -> dict:
    """Join several runs of the same rotations (e.g. training seeds) along the rotation axis.

    Drawing a rotation uniformly from the result is the same as drawing a rotation and then a run, so
    ``bootstrap.run`` on the stacked predictions adds the run as one more level of the hierarchy."""
    runs = list(preds.values())
    first = runs[0]
    for other in runs[1:]:
        if not (np.array_equal(other["bearing"], first["bearing"]) and np.array_equal(other["y"], first["y"])
                and other["rotations"] == first["rotations"] and list(other["models"]) == list(first["models"])):
            raise ValueError("Runs do not cover the same rotations, models and test windows")
    return {"probs": np.concatenate([p["probs"] for p in runs]),
            "thresholds": np.concatenate([p["thresholds"] for p in runs]),
            "qhat": np.concatenate([p["qhat"] for p in runs]),
            "y": first["y"], "bearing": first["bearing"], "models": first["models"],
            "calibrators": first["calibrators"],
            "rotations": [f"{name}:{rot}" for name, p in preds.items() for rot in p["rotations"]]}
