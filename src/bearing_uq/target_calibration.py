"""Does calibrating on a few labelled real-damage bearings restore trustworthy confidence?

For one draw: take a source rotation, move n real (target) bearings per class out of the test set,
and compare three calibration sets on the same remaining test bearings:
  source    - the rotation's held-out artificial/source bearings (as in the main experiment)
  target    - only the n real bearings per class
  combined  - both
The classifier is trained on the rotation's training bearings in every case.
"""

import numpy as np
import pandas as pd

from bearing_uq import bearings as B
from bearing_uq import dataset as D
from bearing_uq import metrics as M
from bearing_uq import splits as S
from bearing_uq.calibration import CALIBRATORS
from bearing_uq.conformal import SplitConformal, set_metrics
from bearing_uq.models import fit, make_models

TARGET_ERROR = 0.05
ALPHA = 0.1


def choose_real_calibration(split: dict[str, str], n_per_class: int, rng) -> list[str]:
    """n test bearings per class, chosen at random; at least one per class stays in the test set."""
    chosen = []
    for label in B.CLASSES:
        pool = sorted(c for c in S.codes_in(split, S.TEST) if B.get(c).label == label)
        if n_per_class >= len(pool):
            raise ValueError(f"Not enough {label} test bearings to move {n_per_class}")
        chosen += list(rng.choice(pool, size=n_per_class, replace=False))
    return sorted(chosen)


def _evaluate(p_cal_raw, y_cal, p_te_raw, y_te) -> list[dict]:
    rows = []
    for name, cls in CALIBRATORS.items():
        cal = cls().fit(p_cal_raw, y_cal)
        p_cal, p_te = cal.transform(p_cal_raw), cal.transform(p_te_raw)
        t = M.threshold_for_target(p_cal.max(1), p_cal.argmax(1) == y_cal, TARGET_ERROR)
        sel = M.selective(p_te.max(1), p_te.argmax(1) == y_te, t)
        rows.append({"calibration": name, "accuracy": M.accuracy(p_te, y_te), "ece": M.ece(p_te, y_te),
                     "auto_rate@5%": sel["automation_rate"], "auto_error@5%": sel["automated_error"]})
    for cc in (False, True):
        cp = SplitConformal(ALPHA, class_conditional=cc).fit(p_cal_raw, y_cal)
        m = set_metrics(cp.predict_sets(p_te_raw), y_te)
        rows.append({"calibration": f"conformal_{'class_conditional' if cc else 'marginal'}",
                     "coverage": m["coverage"], "mean_set_size": m["mean_set_size"]})
    return rows


def run_draw(df: pd.DataFrame, calib: tuple, split: dict[str, str], n_per_class: int,
             seed: int, features) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real = choose_real_calibration(split, n_per_class, rng)
    source = S.codes_in(split, S.CALIB)
    test = [c for c in S.codes_in(split, S.TEST) if c not in real]

    def xy(codes):
        part = df[df["bearing"].isin(codes)]
        return D.design_matrix(part, features), D.targets(part)

    X_tr, y_tr = xy(S.codes_in(split, S.TRAIN))
    X_te, y_te = xy(test)
    cal_sets = {"source": xy(source), "target": xy(real), "combined": xy(source + real)}

    rows = []
    for model_name, model in make_models().items():
        fit(model, X_tr, y_tr)
        p_te = model.predict_proba(X_te)
        for set_name, (X_c, y_c) in cal_sets.items():
            for row in _evaluate(model.predict_proba(X_c), y_c, p_te, y_te):
                rows.append({"model": model_name, "calibration_set": set_name, **row})
    return pd.DataFrame(rows).assign(n_per_class=n_per_class, seed=seed,
                                     source_calib="+".join(calib), real_calib="+".join(real))
