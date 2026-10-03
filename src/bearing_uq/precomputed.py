"""Models whose probabilities were computed elsewhere (the 1D-CNN, trained in Colab).

The CNN saves window probabilities keyed by (bearing, recording, window). Here they are attached to the feature
table as a ``row_key`` column, and ``PrecomputedModel`` serves them through the same fit/predict_proba interface as
the scikit-learn models, so every analysis (calibration, conformal prediction, selective automation, bootstrap,
real-bearing calibration) runs unchanged. Use ``KEY_FEATURES`` as the feature list.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bearing_uq import dataset as D

KEYS = ["bearing", "recording", "window"]
KEY_FEATURES = ["row_key"]
PROB_COLS = [f"p_{label}" for label in D.LABELS]


def with_row_key(df: pd.DataFrame) -> pd.DataFrame:
    """The feature table with a unique integer row_key per window."""
    return df.assign(row_key=np.arange(len(df)))


def probabilities(path: Path, df: pd.DataFrame) -> dict[int, np.ndarray]:
    """row_key -> probability vector, for the windows in a CNN output file."""
    probs = pd.read_csv(path, float_precision="round_trip")
    merged = probs.merge(df[KEYS + ["row_key"]], on=KEYS, how="left", validate="one_to_one")
    if merged["row_key"].isna().any():
        missing = merged.loc[merged["row_key"].isna(), KEYS].head(3).to_dict("records")
        raise ValueError(f"{Path(path).name}: windows not in the feature table, e.g. {missing}")
    values = merged[PROB_COLS].to_numpy(dtype=float)
    return dict(zip(merged["row_key"].astype(int), values))


class PrecomputedModel:
    """Serves saved probabilities; fit() does nothing (the model was trained on the right bearings already)."""

    def __init__(self, probs: dict[int, np.ndarray]):
        self.probs = probs

    def fit(self, X, y):
        return self

    def predict_proba(self, X) -> np.ndarray:
        keys = np.asarray(X)[:, 0].astype(int)
        missing = [k for k in keys if k not in self.probs]
        if missing:
            raise KeyError(f"No saved probabilities for {len(missing)} windows (was the model trained on them?)")
        return np.stack([self.probs[k] for k in keys])
