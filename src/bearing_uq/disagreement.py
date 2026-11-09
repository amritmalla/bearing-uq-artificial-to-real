"""Selective automation that also requires agreement between a feature-based classifier and the 1D-CNN.

Rule (fixed on 2026-11-08, before any result of it was computed):

- For each feature-based classifier (random forest, SVM, XGBoost) and each rotation, a target window is accepted
  automatically only if (a) the classifier's temperature-scaled confidence meets the threshold already selected on
  that rotation's source calibration bearings for the error target, exactly as in the paper, and (b) the 1D-CNN
  trained on the same rotation predicts the same class (argmax of its uncalibrated probabilities).
- The accepted diagnosis is the feature classifier's class. Everything else is referred.
- No new threshold or parameter is fitted, so nothing is chosen on target data.
- Uncalibrated feature probabilities (with their own source thresholds) are reported as a sensitivity check only.

The baseline is the confidence threshold alone. Both rules are evaluated on the same rotations and on the same
hierarchical bootstrap draws (same seed and resampling order as ``bootstrap.run``), so the comparison is paired.
"""

import numpy as np
import pandas as pd

from bearing_uq import dataset as D
from bearing_uq.bootstrap import bearing_indices, resample
from bearing_uq.predictions import TARGETS

HEALTHY = D.LABELS.index("healthy")
RULES = ("confidence", "agreement")


def accept_mask(conf, label, other_label, threshold, rule: str) -> np.ndarray:
    """Windows accepted automatically under a rule."""
    keep = np.asarray(conf) >= threshold
    if rule == "agreement":
        keep &= np.asarray(label) == np.asarray(other_label)
    elif rule != "confidence":
        raise ValueError(f"Unknown rule {rule!r}")
    return keep


def accepted_metrics(keep, label, y) -> dict[str, float]:
    """Automation rate, error among accepted windows and the share of accepted errors that are missed faults."""
    keep, label, y = np.asarray(keep), np.asarray(label), np.asarray(y)
    wrong = keep & (label != y)
    missed = wrong & (label == HEALTHY) & (y != HEALTHY)
    return {"automation_rate": float(keep.mean()),
            "automated_error": float(wrong.sum() / keep.sum()) if keep.any() else float("nan"),
            "missed_share_of_errors": float(missed.sum() / wrong.sum()) if wrong.any() else float("nan")}


def _parts(feat: dict, cnn: dict, r: int, mi: int, calibrator: str):
    ci = list(feat["calibrators"]).index(calibrator)
    p = feat["probs"][r, mi, ci].astype(float)
    cnn_label = cnn["probs"][r, 0, list(cnn["calibrators"]).index("raw")].argmax(1)
    return p.max(1), p.argmax(1), cnn_label, feat["thresholds"][r, mi, ci]


def check_aligned(feat: dict, cnn: dict) -> None:
    if feat["rotations"] != cnn["rotations"]:
        raise ValueError("Feature and CNN predictions cover different rotations")
    if not (np.array_equal(feat["bearing"], cnn["bearing"]) and np.array_equal(feat["y"], cnn["y"])):
        raise ValueError("Feature and CNN test windows are not in the same order")


def per_rotation(feat: dict, cnn: dict, calibrator: str = "temperature") -> pd.DataFrame:
    """Metrics on the full target test set for every rotation, model, target and rule."""
    check_aligned(feat, cnn)
    y, rows = feat["y"], []
    for r, rot in enumerate(feat["rotations"]):
        for mi, model in enumerate(feat["models"]):
            conf, label, cnn_label, thr = _parts(feat, cnn, r, mi, calibrator)
            for ti, target in enumerate(TARGETS):
                for rule in RULES:
                    keep = accept_mask(conf, label, cnn_label, thr[ti], rule)
                    rows.append({"rotation": rot, "model": model, "target": target, "rule": rule,
                                 **accepted_metrics(keep, label, y)})
    return pd.DataFrame(rows)


def per_bearing(feat: dict, cnn: dict, target: float = 0.05, calibrator: str = "temperature") -> pd.DataFrame:
    """Share of each bearing's windows accepted, and the error among them, averaged over rotations."""
    check_aligned(feat, cnn)
    y, bearing, ti, rows = feat["y"], feat["bearing"], TARGETS.index(target), []
    for r in range(len(feat["rotations"])):
        for mi, model in enumerate(feat["models"]):
            conf, label, cnn_label, thr = _parts(feat, cnn, r, mi, calibrator)
            for rule in RULES:
                keep = accept_mask(conf, label, cnn_label, thr[ti], rule)
                for code in np.unique(bearing):
                    m = bearing == code
                    rows.append({"model": model, "rule": rule, "bearing": code, "label": D.LABELS[y[m][0]],
                                 **accepted_metrics(keep[m], label[m], y[m])})
    out = pd.DataFrame(rows)
    return (out.groupby(["model", "rule", "bearing", "label"])[["automation_rate", "automated_error"]]
            .mean().reset_index())


def bootstrap_draws(feat: dict, cnn: dict, n_draws: int = 2000, seed: int = 0,
                    calibrator: str = "temperature") -> pd.DataFrame:
    """Hierarchical bootstrap with the same draws as ``bootstrap.run`` (same seed gives the same rotations and bearings)."""
    check_aligned(feat, cnn)
    rng = np.random.default_rng(seed)
    groups = bearing_indices(feat["bearing"], feat["y"])
    n_rot, rows = feat["probs"].shape[0], []
    for draw in range(n_draws):
        r = rng.integers(n_rot)
        idx = resample(groups, rng)
        y = feat["y"][idx]
        for mi, model in enumerate(feat["models"]):
            conf, label, cnn_label, thr = _parts(feat, cnn, r, mi, calibrator)
            conf, label, cnn_label = conf[idx], label[idx], cnn_label[idx]
            for ti, target in enumerate(TARGETS):
                for rule in RULES:
                    keep = accept_mask(conf, label, cnn_label, thr[ti], rule)
                    rows.append({"draw": draw, "model": model, "target": target, "rule": rule,
                                 **accepted_metrics(keep, label, y)})
    return pd.DataFrame(rows)


def summarise_rotations(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean over rotations, and the share of rotations (with any accepted window) that met the target."""
    rows = rows.assign(any_accepted=rows["automation_rate"] > 0,
                       met=rows["automated_error"] <= rows["target"])
    g = rows.groupby(["model", "target", "rule"])
    out = g[["automation_rate", "automated_error", "missed_share_of_errors"]].mean()
    out["rotations_with_acceptance"] = g["any_accepted"].sum()
    out["target_met_share"] = rows[rows["any_accepted"]].groupby(["model", "target", "rule"])["met"].mean()
    return out.reset_index()


def summarise_draws(draws: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap mean and 95 % percentile interval per model, target, rule and metric."""
    long = draws.melt(id_vars=["draw", "model", "target", "rule"], var_name="metric").dropna(subset=["value"])
    g = long.groupby(["model", "target", "rule", "metric"])["value"]
    return pd.DataFrame({"mean": g.mean(), "ci_low": g.quantile(0.025),
                         "ci_high": g.quantile(0.975)}).reset_index()
