"""Per-rotation test predictions, saved so bootstrap resampling needs no refitting.

For each rotation, model and calibrator we keep the calibrated test probabilities and the
confidence thresholds chosen on the calibration set; for each model, the conformal thresholds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bearing_uq import dataset as D
from bearing_uq import metrics as M
from bearing_uq import splits as S
from bearing_uq.calibration import CALIBRATORS
from bearing_uq.conformal import SplitConformal
from bearing_uq.models import fit, make_models

TARGETS = (0.01, 0.05)
ALPHA = 0.1
CONFORMAL_METHODS = ("marginal", "class_conditional")


def predict_rotation(df: pd.DataFrame, split: dict[str, str], features, seed: int = 0,
                     models: dict | None = None) -> dict:
    parts = {n: D.subset(df, split, n) for n in (S.TRAIN, S.CALIB, S.TEST)}
    X = {n: D.design_matrix(p, features) for n, p in parts.items()}
    y = {n: D.targets(p) for n, p in parts.items()}

    models = models or make_models(seed)
    n_test, k = len(y[S.TEST]), len(D.LABELS)
    probs = np.zeros((len(models), len(CALIBRATORS), n_test, k), dtype=np.float64)
    thresholds = np.zeros((len(models), len(CALIBRATORS), len(TARGETS)))
    qhat = np.zeros((len(models), len(CONFORMAL_METHODS), k))

    for mi, model in enumerate(models.values()):
        fit(model, X[S.TRAIN], y[S.TRAIN])
        p_cal_raw, p_te_raw = model.predict_proba(X[S.CALIB]), model.predict_proba(X[S.TEST])
        for ci, cls in enumerate(CALIBRATORS.values()):
            cal = cls().fit(p_cal_raw, y[S.CALIB])
            p_cal = cal.transform(p_cal_raw)
            probs[mi, ci] = cal.transform(p_te_raw)
            for ti, target in enumerate(TARGETS):
                thresholds[mi, ci, ti] = M.threshold_for_target(
                    p_cal.max(1), p_cal.argmax(1) == y[S.CALIB], target)
        for qi, method in enumerate(CONFORMAL_METHODS):
            cp = SplitConformal(ALPHA, class_conditional=method == "class_conditional")
            qhat[mi, qi] = cp.fit(p_cal_raw, y[S.CALIB]).qhat

    return {"probs": probs, "thresholds": thresholds, "qhat": qhat, "y": y[S.TEST],
            "bearing": parts[S.TEST]["bearing"].to_numpy(dtype=str),
            "models": np.array(list(models)), "calibrators": np.array(list(CALIBRATORS))}


def save(path: Path, pred: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pred)


def load_all(folder: Path) -> dict:
    """Stack all saved rotations: probs (R, M, C, N, K), thresholds (R, M, C, T), qhat (R, M, Q, K)."""
    files = sorted(Path(folder).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {folder}")
    loaded = [dict(np.load(f)) for f in files]
    first = loaded[0]
    for other in loaded[1:]:
        if not (np.array_equal(other["bearing"], first["bearing"]) and np.array_equal(other["y"], first["y"])):
            raise ValueError("Test set differs between rotations")
    return {"probs": np.stack([p["probs"] for p in loaded]),
            "thresholds": np.stack([p["thresholds"] for p in loaded]),
            "qhat": np.stack([p["qhat"] for p in loaded]),
            "y": first["y"], "bearing": first["bearing"],
            "models": list(first["models"]), "calibrators": list(first["calibrators"]),
            "rotations": [f.stem for f in files]}
