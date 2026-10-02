"""In-domain reference: leave one source bearing out at a time.

Each source bearing is predicted by models trained on all other source bearings,
so the score reflects generalisation to an unseen bearing of the same damage origin.
"""

import numpy as np
import pandas as pd

from bearing_uq import dataset as D
from bearing_uq import metrics as M
from bearing_uq.models import fit, make_models


def leave_one_bearing_out(df: pd.DataFrame, codes: list[str], seed: int = 0,
                          features: list[str] = D.FEATURES):
    """Return (summary per model, per-bearing accuracy) on raw probabilities."""
    data = df[df["bearing"].isin(codes)].reset_index(drop=True)
    X, y = D.design_matrix(data, features), D.targets(data)
    summary, per_bearing = [], []
    for name in make_models(seed):
        probs = np.zeros((len(data), len(D.LABELS)))
        for code in codes:
            test = (data["bearing"] == code).to_numpy()
            model = fit(make_models(seed)[name], X[~test], y[~test])
            probs[test] = model.predict_proba(X[test])
        correct = probs.argmax(1) == y
        summary.append({"model": name, "accuracy": M.accuracy(probs, y), "macro_f1": M.macro_f1(probs, y),
                        "ece": M.ece(probs, y), "brier": M.brier(probs, y),
                        "aurc": M.aurc(probs.max(1), correct)})
        for code, g in data.assign(correct=correct, conf=probs.max(1)).groupby("bearing"):
            per_bearing.append({"model": name, "bearing": code, "label": g["label"].iloc[0],
                                "accuracy": g["correct"].mean(), "mean_confidence": g["conf"].mean()})
    return pd.DataFrame(summary), pd.DataFrame(per_bearing)
