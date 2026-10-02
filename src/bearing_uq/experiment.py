"""Main artificial-to-real experiment: train on source, calibrate on held-out source, test on real damage."""

import numpy as np
import pandas as pd

from bearing_uq import dataset as D
from bearing_uq import metrics as M
from bearing_uq import splits as S
from bearing_uq.calibration import CALIBRATORS
from bearing_uq.conformal import SplitConformal, set_metrics
from bearing_uq.models import fit, make_models

TARGET_ERRORS = (0.01, 0.05)
ALPHA = 0.1


def _xy(df, split, name):
    part = D.subset(df, split, name)
    return D.design_matrix(part), D.targets(part)


def _score(probs, y) -> dict:
    return {"accuracy": M.accuracy(probs, y), "macro_f1": M.macro_f1(probs, y),
            "ece": M.ece(probs, y), "brier": M.brier(probs, y),
            "aurc": M.aurc(probs.max(axis=1), probs.argmax(axis=1) == y)}


def _automation(p_cal, y_cal, p_test, y_test) -> dict:
    """Threshold chosen on the calibration set (what a deployment would do) vs. the test-set oracle."""
    out = {}
    for target in TARGET_ERRORS:
        t = M.threshold_for_target(p_cal.max(1), p_cal.argmax(1) == y_cal, target)
        sel = M.selective(p_test.max(1), p_test.argmax(1) == y_test, t)
        oracle = M.threshold_for_target(p_test.max(1), p_test.argmax(1) == y_test, target)
        pct = f"{target:.0%}"
        out[f"auto_rate@{pct}"] = sel["automation_rate"]
        out[f"auto_error@{pct}"] = sel["automated_error"]
        out[f"oracle_auto_rate@{pct}"] = M.selective(p_test.max(1), p_test.argmax(1) == y_test, oracle)["automation_rate"]
    return out


def run(df: pd.DataFrame, split: dict[str, str], seed: int = 0):
    X_tr, y_tr = _xy(df, split, S.TRAIN)
    X_cal, y_cal = _xy(df, split, S.CALIB)
    X_te, y_te = _xy(df, split, S.TEST)

    main, conformal, recall = [], [], []
    for name, model in make_models(seed).items():
        fit(model, X_tr, y_tr)
        p_cal_raw, p_te_raw = model.predict_proba(X_cal), model.predict_proba(X_te)

        main.append({"model": name, "calibration": "raw", "domain": "source_holdout",
                     **_score(p_cal_raw, y_cal)})
        for cal_name, cls in CALIBRATORS.items():
            cal = cls().fit(p_cal_raw, y_cal)
            p_cal, p_te = cal.transform(p_cal_raw), cal.transform(p_te_raw)
            main.append({"model": name, "calibration": cal_name, "domain": "target_real",
                         **_score(p_te, y_te), **_automation(p_cal, y_cal, p_te, y_te)})

        for cc in (False, True):
            cp = SplitConformal(alpha=ALPHA, class_conditional=cc).fit(p_cal_raw, y_cal)
            conformal.append({"model": name, "method": "class_conditional" if cc else "marginal",
                              "nominal_coverage": 1 - ALPHA, **set_metrics(cp.predict_sets(p_te_raw), y_te)})

        pred = p_te_raw.argmax(1)
        recall.append({"model": name, **{f"recall_{lab}": float((pred[y_te == i] == i).mean())
                                         for i, lab in enumerate(D.LABELS)}})

    sizes = {n: len(D.subset(df, split, n)) for n in (S.TRAIN, S.CALIB, S.TEST)}
    return pd.DataFrame(main), pd.DataFrame(conformal), pd.DataFrame(recall), sizes
