"""Hierarchical bootstrap: resample the calibration rotation and the test bearings.

Each draw picks one rotation at random, then resamples test bearings with replacement within each
class (keeping the number of bearings per class), and recomputes the metrics on their windows.
"""

import numpy as np
import pandas as pd

from bearing_uq import metrics as M
from bearing_uq.predictions import CONFORMAL_METHODS, TARGETS


def bearing_indices(bearing: np.ndarray, y: np.ndarray) -> dict[int, list[np.ndarray]]:
    """Window indices of each test bearing, grouped by class."""
    groups: dict[int, list[np.ndarray]] = {}
    for code in np.unique(bearing):
        idx = np.flatnonzero(bearing == code)
        groups.setdefault(int(y[idx[0]]), []).append(idx)
    return groups


def resample(groups, rng) -> np.ndarray:
    picks = [g[i] for g in groups.values() for i in rng.integers(0, len(g), size=len(g))]
    return np.concatenate(picks)


def _draw_metrics(p, y, thresholds, qhat_row) -> dict:
    conf, correct = p.max(1), p.argmax(1) == y
    out = {"accuracy": correct.mean(), "ece": M.ece(p, y), "aurc": M.aurc(conf, correct)}
    for ti, target in enumerate(TARGETS):
        sel = M.selective(conf, correct, thresholds[ti])
        pct = f"{target:.0%}"
        out[f"auto_rate@{pct}"] = sel["automation_rate"]
        out[f"auto_error@{pct}"] = sel["automated_error"]
        oracle = M.threshold_for_target(conf, correct, target)
        out[f"oracle_auto_rate@{pct}"] = M.selective(conf, correct, oracle)["automation_rate"]
    return out


def run(pred: dict, n_draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = bearing_indices(pred["bearing"], pred["y"])
    n_rot = pred["probs"].shape[0]
    rows = []
    for draw in range(n_draws):
        r = rng.integers(n_rot)
        idx = resample(groups, rng)
        y = pred["y"][idx]
        for mi, model in enumerate(pred["models"]):
            raw = pred["probs"][r, mi, list(pred["calibrators"]).index("raw")][idx]
            for qi, method in enumerate(CONFORMAL_METHODS):
                sets = (1 - raw) <= pred["qhat"][r, mi, qi][None, :]
                rows.append({"draw": draw, "model": model, "calibration": f"conformal_{method}",
                             "coverage": sets[np.arange(len(y)), y].mean(),
                             "mean_set_size": sets.sum(1).mean()})
            for ci, cal in enumerate(pred["calibrators"]):
                p = pred["probs"][r, mi, ci][idx].astype(float)
                rows.append({"draw": draw, "model": model, "calibration": cal,
                             **_draw_metrics(p, y, pred["thresholds"][r, mi, ci], None)})
    return pd.DataFrame(rows)


def summarise(draws: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95 % percentile interval per model, calibration and metric."""
    long = draws.melt(id_vars=["draw", "model", "calibration"], var_name="metric").dropna(subset=["value"])
    g = long.groupby(["model", "calibration", "metric"])["value"]
    return pd.DataFrame({"mean": g.mean(), "ci_low": g.quantile(0.025),
                         "ci_high": g.quantile(0.975)}).reset_index()
